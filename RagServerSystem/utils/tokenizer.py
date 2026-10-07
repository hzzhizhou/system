"""
企业级中文分词：自定义词典、停用词过滤、容错处理
"""
import jieba
from typing import List
from config.settings import STOP_WORDS
from logs.log_config import log
import re

class ChineseTokenizer:
    """中文分词工具，适配RAG检索场景（支持小写兼容）"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_tokenizer()
        return cls._instance

    def _init_tokenizer(self):
        """初始化分词器：加载词典、停用词、优化配置"""
        DOMAIN_TERM_FREQ = 10000
        domain_terms = [
            # 售后动作
            "退货", "退款", "换货", "维修", "上门取件", "无理由退货", "七天无理由",
            # 费用相关
            "运费", "运费险", "价格保护",
            # 凭证与时效
            "发票", "保修", "保修期", "订单", "订单状态", "签收", "物流",
            # 服务角色
            "客服", "售后",
        ]
        for term in domain_terms:
            jieba.add_word(term, freq=DOMAIN_TERM_FREQ)

        # 2. 加载停用词
        self.stop_words = self._load_stop_words()
        # 3. 开启并行分词（性能优化）
        import sys
        if sys.platform != "win32":
            try:
                jieba.enable_parallel(4)
            except Exception as e:
                log.warning(f"并行分词开启失败：{e}")
        else:
            log.info("Windows 系统，跳过并行分词（使用单线程）")
    def _load_stop_words(self) -> set:
        """加载停用词（全异常兜底）"""
        stop_words = set()
        try:
            # 修正原代码bug：open缺少文件路径参数
            with open(STOP_WORDS, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip().lower()  # 停用词统一小写
                    if word:
                        stop_words.add(word)
            log.info(f"停用词加载完成，数量：{len(stop_words)}")
        except FileNotFoundError:
            # 企业级兜底停用词
            stop_words = {"的", "是", "在", "和", "有", "了", "我", "你", "他", "\n", "\t", " "}
            log.warning("停用词文件不存在，使用兜底停用词")
        except Exception as e:
            stop_words = {"的", "是", "在", "和", "有"}
            log.error(f"停用词加载异常：{str(e)}")
        return stop_words

    def clean_text(self, text: str) -> str:
        """企业级文本清洗：保留中文/英文/数字，去除所有干扰字符"""
        if not text:
            return ""
        text = text.strip()
        # 正则清洗：只保留有效字符（兼容中英文）
        return re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)

    def tokenize(self, text: str) -> List[str]:
        """清洗 → 精确分词 → 去停用词 → 统一小写；**保留重复词**（词频是 BM25 的输入）"""
        clean_text = self.clean_text(text)
        if not clean_text:
            return []
        tokens = jieba.lcut(clean_text, cut_all=False)
        return [
            t.strip().lower() for t in tokens
            if t.strip() and t.strip().lower() not in self.stop_words
        ]

    def tokenize_document(self, text: str) -> List[str]:
        """文档侧分词：保留词频（不去重）。
        BM25 靠「词频饱和（k1）」和「文档长度归一化（b / avgdl）」区分文档，
        而这两项都以词频与真实长度为输入。若在这里去重，TF 恒为 1、文档长度退化成
        "去重后的词表大小"（实测 avgdl 被低估 28%），打分只剩 IDF × 长度归一化。
        后果不只是"同分"，而是**排序倒挂**：实测本项目语料查「退货」时，
        出现 20 次的长块得 0.78 分，而只出现 1 次的短块得 1.24 分——
        越相关的文档被长度惩罚压得越低。保留词频后两者变为 2.40 / 1.26，恢复正确序。
        """
        return self.tokenize(text)

    def tokenize_query(self, text: str) -> List[str]:
        """查询侧分词：去重（保留首次出现顺序）。
        查询是词袋，同一个词重复出现不代表更重要；而 rank_bm25 的 get_scores()会按查询 token 逐个累加，重复词会把该项的权重成倍放大，故查询侧必须去重。
        """
        return list(dict.fromkeys(self.tokenize(text)))

# 全局单例分词器
tokenizer = ChineseTokenizer()