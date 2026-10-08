"""
细粒度业务意图分类器
区分退货/退款/物流/投诉/咨询/订单/闲聊等业务子意图，
让 Agent 在调用工具前就知道优先该用哪个工具，避免盲目试探。

两阶段（hybrid 模式）：
  阶段1 规则：关键词 + 优先级打分，快、免费、可解释
     - 命中且置信度 >= INTENT_LLM_FALLBACK_CONFIDENCE → 直接返回
     - 命中但置信度低 / 未命中 → 进入阶段2
  阶段2 LLM：few-shot prompt 兜底，处理规则覆盖不到的语义模糊问题

类型定义（对齐 settings.INTENT_TOOL_MAP）：
  consult   知识咨询   → search_knowledge
  return    退货        → search_knowledge (先查规则，必要时 create_ticket)
  refund    退款        → query_order
  logistics 物流        → query_order
  order     订单        → query_order
  complaint 投诉        → create_ticket
  chat      闲聊        → none
"""
import time
from typing import Dict, Tuple, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agent.intent_rules import (
    ACTION_HINTS, POLICY_QUESTION_RE, POLICY_QUERY_HINTS, RULE_PATTERNS,
)
from config.settings import (
    INTENT_MODE, INTENT_LLM_FALLBACK_CONFIDENCE, INTENT_TOOL_MAP,
)
from logs.log_config import log
from utils.circuit_breaker import llm_breaker

# 意图规则数据（词表 + 政策句式正则）已外置到 agent/intent_rules.py，
# 本文件只保留算法：打分、政策判定、LLM 兜底、三级级联。


def is_policy_question(question: str) -> bool:
    """判断是否只是「问政策/问规则」，而不是「提出办理诉求」。

    两步：先过办理动作闸门（有动作措辞就不是问政策），
    再看政策词表（词表与句式定义见 agent/intent_rules.py）。
    """
    text = question or ""
    if any(a in text for a in ACTION_HINTS):
        return False
    return any(h in text for h in POLICY_QUERY_HINTS) or bool(POLICY_QUESTION_RE.search(text))

# 意图的中文名（日志/提示用）
INTENT_CN = {
    "consult": "知识咨询", "return": "退货", "refund": "退款", "logistics": "物流",
    "order": "订单", "complaint": "投诉", "chat": "闲聊",
}


class IntentClassifier:
    def __init__(self, llm=None, mode: str = INTENT_MODE):
        self.mode = mode
        self.llm = llm
        self._fallback_chain = self._init_fallback_chain() if (llm and mode != "rule") else None

    def _init_fallback_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个电商客服的意图分类器。从用户问题中识别业务子意图，只返回一个标签。

可选标签及含义：
- chat: 闲聊/打招呼/致谢，不需要处理业务
- consult: 知识咨询（产品参数、保修/维修/发票等售后规则、使用方法等知识类问题）
- return: 退货（用户提出要退货/换货的诉求）
- refund: 退款（用户要退款，或追问退货的钱什么时候到账）
- logistics: 物流/快递查询
- order: 订单状态查询、历史订单查询
- complaint: 投诉、维权、举报

易混淆处的判定口径：
- 问规则 vs 提诉求：句中出现「退货/退款」但在问规则（怎么算、由谁承担、什么条件）→ consult；
  用户在提诉求（我要、帮我办、想退）→ return/refund。
- 退货 vs 退款：问「退货的钱到没到、去哪了」→ refund（问的是钱，不是要寄回商品）。
- 投诉 vs 咨询：抱怨且要说法/赔偿/举报/找领导 → complaint；只是问政策 → consult。
- 招呼与应诺（嗯嗯、好的、多谢、辛苦了）→ chat，不要因为句子短就归 consult。

判定示例：
- 「我要把手机退了」→ return
- 「退货的钱几天能到账」→ refund
- 「我的包裹到哪了」→ logistics
- 「我在你们这儿下过哪些单」→ order
- 「保修期内换屏幕要钱吗」→ consult
- 「客服一直不回我，太气人了」→ complaint
- 「好的，辛苦了」→ chat

注意：只输出一个标签，不要输出任何解释。如果实在无法判断，输出 consult。"""),
            ("human", "用户问题：{question}"),
        ])
        return prompt | self.llm | StrOutputParser()

    # ---------- 阶段1：规则打分 ----------
    @staticmethod
    def _match_score(question: str, patterns: List[Tuple[str, float]]) -> float:
        """按「先长后短」匹配并计分，已被更长关键词覆盖的位置不重复计分。

        否则「订单号」会同时命中「订单号」(3.0) 与「订单」(2.5)，把订单意图分数虚高，
        导致“我要退货，订单号 SOxxx”被判成订单意图，退货主线流程无法触发。
        """
        claimed = []   # 已计分的位置区间 [(start, end)]
        total = 0.0
        for kw, weight in sorted(patterns, key=lambda p: -len(p[0])):
            start = question.find(kw)
            while start != -1:
                end = start + len(kw)
                if not any(s <= start and end <= e for s, e in claimed):
                    total += weight
                    claimed.append((start, end))
                start = question.find(kw, start + 1)
        return total

    def rule_scores(self, question: str) -> Dict[str, float]:
        """只跑规则打分，返回各意图的原始得分（无 LLM、零耗时）。

        与 _rule_classify 的区别：不做归一化。调用方（如 DST 主线判断是否该让位）
        需要的是「是否命中某业务线的强关键词」的绝对量级，归一化后的占比会失真。
        """
        scores: Dict[str, float] = {}
        for intent, patterns in RULE_PATTERNS.items():
            total = self._match_score(question, patterns)
            if total > 0:
                scores[intent] = total
        return scores

    def _rule_classify(self, question: str) -> Tuple[str, float]:
        """规则打分，返回 (意图, 归一化置信度)"""
        scores = self.rule_scores(question)

        if not scores:
            return "consult", 0.1

        # 取最高分意图
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        all_score = sum(scores.values())
        # 置信度 = 最高分占比（区分 "分明是退货" 和 "模糊触及多个")
        confidence = best_score / all_score if all_score > 0 else 0.0
        # 「问政策」而非「要办理」：命中退货/退款词表但整句在问规则时改判知识咨询。
        # 置信度沿用原占比——改判依据同样是这批命中词，不是把握不足的信号，
        # 保持高置信可避免 hybrid 模式再花一次 LLM 兜底（还可能被兜回 return）。
        if best_intent in ("return", "refund") and is_policy_question(question):
            best_intent = "consult"
        return best_intent, round(confidence, 3)

    # ---------- 阶段2：LLM 兜底 ----------
    async def _llm_classify(self, question: str) -> str:
        if self._fallback_chain is None:
            return "consult"
        # 熔断：LLM 连续失败时不再发起兜底调用（意图兜底本身可缺省），
        # 直接回退规则默认意图，避免拖长响应并加重下游压力
        if not llm_breaker.allow_request():
            log.warning("LLM 熔断中，意图兜底跳过 LLM，回退 consult")
            return "consult"
        try:
            result = await self._fallback_chain.ainvoke({"question": question})
            llm_breaker.record_success()
            intent = result.strip().lower()
            return intent if intent in INTENT_TOOL_MAP else "consult"
        except Exception as e:
            llm_breaker.record_failure()
            log.warning(f"LLM 意图兜底失败，回退 consult: {e}")
            return "consult"

    # ---------- 主入口 ----------
    async def classify(self, question: str, force_intent: str = None) -> Dict[str, object]:
        """
        返回 {intent, intent_cn, confidence, primary_tool, source}
        source: rule / llm / fallback
        force_intent: 若给定合法意图，直接复用其值，跳过规则/LLM 分类。
          用途：DST 主线流程（退货/退款/投诉等）进行中时，后续追问不必重新分类
              —— 避免“确认/跟进”等短句触发昂贵的 LLM 兜底（实测约 +7s）。
        """
        start = time.time()
        if force_intent and force_intent in INTENT_TOOL_MAP:
            return self._pack(force_intent, 1.0, "dst_reuse", cost=f"{time.time()-start:.3f}s")

        if self.mode == "rule":
            intent, confidence = self._rule_classify(question)
            return self._pack(intent, confidence, "rule")

        if self.mode == "llm":
            intent = await self._llm_classify(question)
            return self._pack(intent, 1.0, "llm")

        # hybrid：规则优先
        intent, confidence = self._rule_classify(question)
        if confidence >= INTENT_LLM_FALLBACK_CONFIDENCE:
            return self._pack(intent, confidence, "rule")

        # 规则置信度低 → LLM 兜底
        intent = await self._llm_classify(question)
        source = "rule_lowconf" if intent in RULE_PATTERNS else "llm"
        return self._pack(intent, max(confidence, 0.6), source, cost=f"{time.time()-start:.3f}s")

    def _pack(self, intent: str, confidence: float, source: str, cost: str = "") -> Dict[str, object]:
        return {
            "intent": intent,
            "intent_cn": INTENT_CN.get(intent, intent),
            "confidence": confidence,
            "primary_tool": INTENT_TOOL_MAP.get(intent, "search_knowledge"),
            "source": source,
            "cost": cost or f"{0.0:.3f}s",
        }