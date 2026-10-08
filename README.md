# 智选商城 · 智能售后客服系统

面向 **「智能手机及配件」电商售后**场景的 **RAG + 对话式 Agent 智能客服**，覆盖
**知识问答 / 订单与物流查询 / 退货退款办理 / 投诉转人工** 全链路。
技术栈为 **LangChain + Chroma + FastAPI + Vue 3**，单仓库（monorepo）交付后端服务与前端界面。

> 主打工程化亮点：统一 Agent（5 工具）、细粒度意图分类、DST 对话状态机、确定性退货/退款主流程、
> 查询分解与多子问门控、置信度门控（绝对余弦相似度）、三态熔断器、输出合规护栏、
> 人工会话 WebSocket、Docker Compose 一键编排。

---

## 目录

| # | 章节 | 内容 |
|---|---|---|
| 一 | [仓库结构总览](#一仓库结构总览) | monorepo 布局：后端分层包 + 前端 + 本地模型 |
| 二 | [技术栈](#二技术栈) | 框架 / 向量库 / 嵌入 / LLM / 监控 / 部署 |
| 三 | [系统架构](#三系统架构) | 分层数据流图 + PlantUML 架构图入口 |
| 四 | [核心能力](#四核心能力) | RAG 流水线 · 统一 Agent · 意图分类 · DST · 人工会话 · 健壮性 · 性能 |
| 五 | [API 一览](#五api-一览) | 30 个端点，按鉴权层级分组（公开 / 认证 / 会话 / 管理） |
| 六 | [环境约定与依赖服务](#六环境约定与依赖服务) | Python 环境 · 依赖服务 · `.env` · MySQL 准备 |
| 七 | [快速开始](#七快速开始) | 入库 · 启动后端 / 前端 · 双端口登录 · 本地模型 |
| 八 | [前端说明](#八前端说明) | 页面路由 · 与后端的约定 · 开发命令 |
| 九 | [测试与校验](#九测试与校验) | 单元测试 · 在线端到端 · 离线评测与验证工具 |
| 十 | [已知边界与优化点](#十已知边界与优化点诚实清单) | 诚实清单：架构边界 + 待优化项 |
| 十一 | [后端分层详解](#十一后端分层详解) | 逐文件说明后端目录 |

---

## 一、仓库结构总览

```
AI大模型与智能体开发/
├── RagServerSystem/            后端：FastAPI + RAG 检索 + 对话式 Agent
│   ├── access/                 接入层：应用装配 · 路由 · 鉴权 · 限流
│   ├── service/                服务层：轮次编排 · 6 条确定性业务分支 · 人工会话 · 响应缓存
│   ├── agent/                  Agent 层：统一 Agent（5 工具）· 意图分类 · DST 状态机
│   ├── generation/             生成层：流式生成 · 合规护栏
│   ├── retrieval/              检索层：查询改写 · 混合检索 · 重排 · 置信度门控
│   ├── ingestion/              数据层：文档加载 · 文本清洗 · 分块 · 向量入库
│   ├── infrastructure/         基础设施：Chroma · Redis · MySQL
│   ├── shared/                 跨层共享：业务常量 · 话术模板 · 请求/响应 DTO
│   ├── evaluation/             评估层：RAGAS 评测（30 题评测集 + 双臂对比，离线）
│   ├── config/settings.py      全局配置（分块 / 检索 / 门控 / 熔断阈值）
│   ├── utils/                  LLM 工厂 · 熔断器 · 安全 · Prometheus 指标
│   ├── data/                   知识库语料（10 个文件：8 篇文档 + 2 份表格）
│   ├── tests/                  pytest 单元测试（178 passed）
│   ├── deploy/ logs/           依赖服务编排（Redis）· 日志配置与运行日志
│   ├── e2e_*.py intent_*.py    开发期工具：在线端到端 / 意图评测与校准 / 入库检索验证（见 §9.2）
│   └── main.py                 服务入口（uvicorn，8000 端口）
├── frontend/                   前端：Vue 3 + Element Plus（Vite）
│   ├── src/views/user/         买家端：ChatView 智能对话主界面 · OrderView 我的订单
│   ├── src/views/admin/        管理端：人工工作台 · 工单 · 订单 · 知识库 · 用户 · 监控
│   ├── src/views/LoginView.vue 登录 / 注册
│   ├── src/api/index.js        axios 封装 + 流式问答 + 人工会话 WebSocket
│   ├── src/router/             路由与登录 / 管理员守卫
│   ├── src/stores/user.js      Pinia：登录态
│   └── vite.config.js          dev 5173，代理 /api（含 ws）→ 127.0.0.1:8000
├── README.md                   本文档（项目唯一说明入口）
├── models/                     本地模型（bge 系列，约 656 MB，不入库）
└── .venv/                      统一 Python 虚拟环境（不入库）
```

---

## 二、技术栈

| 类别 | 选型 |
|---|---|
| 后端框架 | LangChain, FastAPI（单进程 uvicorn） |
| 向量库 | Chroma（异步封装，单机持久化至 `vector_db-sql/`） |
| 嵌入模型 | 默认 DashScope 云端 `qwen3.7-text-embedding`；可切本地 `bge-base-zh-v1.5`（768 维，CPU 推理约 13~17s/次） |
| LLM | 通过 **DashScope OpenAI 兼容端点**（`compatible-mode/v1`）调用，模型名见 `config/settings.py` 的 `LLM_MODEL` |
| 检索 | BM25 + 稠密向量混合检索、**RRF 融合（rank-based，`RRF_K=60`）**、**Cross-Encoder 重排（`bge-reranker-base`）**（`BM25_WEIGHT=0.4 / VECTOR_WEIGHT=0.6` 仅对可选的 `weighted` 融合策略生效） |
| 会话记忆 | MySQL 为准（对话历史逐条落库）+ Redis 热缓存（不可用时回源查库，fail-open） |
| 结构化持久化 | MySQL（订单、工单/转人工、用户账号、DST 会话状态、会话与消息） |
| 监控 | Prometheus `/metrics`、健康检查 `/health`、接口限流 |
| 部署 | Docker Compose 一键编排（app + MySQL + Redis） |
| 测试 | pytest（单元，178 passed）+ 在线端到端脚本 |
| 前端 | Vue 3.5（`<script setup>`）, Element Plus 2.14, Vue Router 5, Pinia 4, Vite 8, axios |
| 前端流式 | 原生 `fetch` + `ReadableStream`（问答）；WebSocket（人工会话） |

---

## 三、系统架构

采用**分层 + 可观测**架构，数据单向流动，各层职责单一、可替换。

```
┌─────────────────────────────────────────────────────────────┐
│  用户交互层    Vue 前端 / 直接调用 HTTP API                  │
├─────────────────────────────────────────────────────────────┤
│  接入层（FastAPI /metrics /health /限流 /鉴权 /缓存）        │
├─────────────────────────────────────────────────────────────┤
│  Agent + DST   意图分类 · DST状态机 · 确定性业务主流程        │
├─────────────────────────────────────────────────────────────┤
│  生成层        流式生成 · 合规护栏                          │
├─────────────────────────────────────────────────────────────┤
│  检索层        路由 · 查询改写 · 混合检索 · RRF 融合 · 精排    │
│                · 查询分解（多问题→逐子问检索）                │
│                · 置信度门控（vector_score 绝对相似度）        │
├─────────────────────────────────────────────────────────────┤
│  数据层        文档加载 · 文本清洗 · 自适应分块 · 向量入库    │
├─────────────────────────────────────────────────────────────┤
│  基础设施      Chroma · DashScope(Embedding/LLM) · Redis     │
│                · MySQL(文档状态/订单/工单/用户/会话/DST)       │
└─────────────────────────────────────────────────────────────┘
```

详细架构图见 [RagServerSystem/docs/architecture.puml](RagServerSystem/docs/architecture.puml)（PlantUML），
文字说明见 [RagServerSystem/docs/architecture.md](RagServerSystem/docs/architecture.md)。

---

## 四、核心能力

### 4.1 RAG 检索流水线（`retrieval/service.py`）

- **自适应分块**：默认 `recursive`（按 Markdown 标题结构 + 768/128 递归切分）；
  `combined_splitter`（语义父块 2000 + 递归子块 400）保留为可选策略，父子回填仅 `/rag/stream` 生效。
- **混合检索**：BM25 + 向量双路召回，融合策略由 `FUSION_METHOD` 二选一——默认 **RRF（rank-based）**，
  按名次累加 `1/(RRF_K + rank)`（`RRF_K=60`），与两路分数尺度无关、无需调权重；`weighted` 为备选，
  两路分数各自 Min-Max 归一化后按 `BM25_WEIGHT=0.4 / VECTOR_WEIGHT=0.6` 加权。融合层与重排层**不串联**。
- **Cross-Encoder 重排**：融合后取前 `RERANK_CANDIDATES=20` 条送本地 `bge-reranker-base`
  （`RERANK_MODE="cross_encoder"`）逐条打分，按分降序返回前 `RERANK_TOP_N=5` 条。模型缺失或推理失败
  **降级为融合序并告警**，不会让检索整体失败；置 `RERANK_MODE="none"` 可关闭重排、直接返回融合序（可灰度/排障）。
- **查询分解（一次提多个问题）**：`split_sub_questions()` 按 `？/?/；/换行` 与显式连接词
  （另外/还有/以及/顺便…）做**确定性切分**（纯函数、可单测），拆不出多段但看着像多问题时才由 LLM 兜底拆一次；
  随后**逐子问独立检索**（各子问独占 `RERANK_TOP_N` 名额，不再互相挤占）、结果**交错合并去重**
  （防生成层按顺序截断上下文时丢掉后一个子问）。实测 4 个合问的子问覆盖率由 2/6、3/6、1/9、3/6 提升到
  6/6、6/6、9/9、6/6，端到端不再出现「第二问静默消失」。
- **多子问门控（答不上的不拖累能答的）**：逐子问算各自门控后，**先剔除 `low` 的子问，再在剩余可答子问间取最弱**
  作为整轮门控；若全部子问都 `low`，仍保持整轮转人工（不放松）。修前「你们有哪些在售手机？iPhone 15 多少钱？」
  取最弱子问得 0.54(low)，连能答的「iPhone 15 4999 元」也一并被放弃；修后剔除无资料子问，整轮 0.62(medium) 正常作答。
  代价：被剔除子问的资料一并丢弃，不做「这部分没查到」的显式说明（安全但略不透明，属已知取舍）。
- **文档类别预过滤（跨域取并集）**：检索前用 `infer_doc_categories()` 按问题关键词推断 `doc_category`
  （类别来源为**文件名**派生，见 `ingestion/loader/mysql_data_loader.derive_doc_category`）缩小范围。
  **返回全部命中类别而非优先级最高的一个**——「运费险能赔多少钱」同时命中 `product`（"多少钱"）与
  `faq`（"运费"），只留 `product` 会把真正答到问题的 `平台FAQ(PLAT-013)` 整类排除，实测置信度被压到
  0.53(low) → 用户得到「没查到资料」。多类时用 `$in` 取并集：单类等值 / 多类并集 / 无命中不过滤 / 显式 `filter_metadata` 优先。
- **置信度门控（基于绝对相似度）**：`vector_score = 1 - distance/2`（即**余弦相似度**；Chroma 默认返回平方 L2 距离且库内向量已单位化）
  做门控，阈值 `high ≥0.80 / low <0.55`，margin `0.08`（top 与第二名差距过小则 high 降级 medium），
  低置信**自动转人工**。不使用批次内归一化分数（全差批次 top 也会被拉到 1.0）。阈值分不开的
  「模型只吐无相关信息」情形另由**空答复兜底**接管：流式产出先缓冲，判定为占位答复
  （`is_empty_answer` / `could_be_empty_answer`）即整段换成 `KNOWLEDGE_NO_ANSWER`
  （「没有查到可靠的资料…」）确定性话术；内部串「无相关信息」不再原样透给用户。
- **表格入库**：`.csv` 整表、`.docx/.xlsx` 内表格统一转成**带表头的 Markdown 表格块**入库
  （`TABLE_CHUNK_SIZE=1000`，表头 + 约 5~6 行），避免表头与数据行分离、一行多列被拆进不同块。

### 4.2 统一对话式 Agent（`agent/service.py`）

- 单 Agent 编排 5 个工具：`search_knowledge`、`query_order`（按订单号）、`list_my_orders`
  （按登录账号列出名下订单）、`query_ticket`、`create_ticket`。
- 不接入联网搜索工具：模型凭参数记忆「假装联网」会编造资讯与来源，外部资讯类问题统一如实告知并转人工。
- `search_knowledge` **复用 `RetrievalService.retrieve()` 完整流水线**（路由→改写→查询分解→混合检索→rerank→门控），
  因此置信度门控与重排同样生效。
- 所有调用统一 OpenAI 兼容协议（原生 ChatTongyi 无法识别 flash 类模型名）。
- **投诉意图不进 ReAct 工具循环**：直接建单转人工，避免多轮工具调用超时与情绪激化。

### 4.3 细粒度意图分类（`agent/intent_classifier.py` + `agent/intent_rules.py`）

- 规则优先 + LLM 兜底（`INTENT_MODE=hybrid`），识别 7 类业务子意图：
  `consult / return / refund / logistics / order / complaint / chat`。
- **规则数据与算法分离**：词表与句式正则集中在 `agent/intent_rules.py`
  （`RULE_PATTERNS` / `POLICY_QUERY_HINTS` / `ACTION_HINTS` / 合并后的 `POLICY_QUESTION_RE`），
  `intent_classifier.py` 只保留加权打分与级联逻辑。调规则只改前者，改算法只改后者。
- **置信度语义**：`conf = best_score / all_score`（无量纲占比）；`conf ≥ 0.5`
  时规则直判，**LLM 兜底完全不触发**（`INTENT_LLM_FALLBACK_CONFIDENCE=0.5`）。
  这意味着词表内任何不自洽的权重都会造成**永久误判**，因此规则表需要在评测集上持续回归。
- **「问政策」≠「要办理」**：`退货/退款/换货` 同时出现在两类问句里——「退货运费由谁承担」「退款要几天到账」
  是**问规则**（走知识检索直接作答），「我要退货」「帮我退款」才是**提诉求**（进退货/退款确定性主流程）。
  词表权重分不开，故按「**先过 `ACTION_HINTS` 动作闸门、再查问法守卫**」判定：
  1. 政策词表 `POLICY_QUERY_HINTS`（由谁承担 / 怎么算 / 多久 / 流程 / 是否…）
  2. 合并正则 `POLICY_QUESTION_RE`，内含三个互斥分支（各自保留 `$` 锚点，一次 `search` 等价于拆开三次）：
     - 条件类（`支持|能|可以|是否…吗`：「支持7天无理由退货吗」）
     - 评价类（`麻烦|方便|难|好退…` + 句末疑问或 `A不A` 重叠式：「退货麻烦吗」「退款麻不麻烦」）
     - 疑问句式（`怎么退/怎么换/如何退`、`哪天/几时/何时`、`才能`、`要不要/需不需要`、
       以 `退/换/运费…吗么嘛呢` 收尾：「我要退款怎么弄」「运费谁出」）

  任一命中即把 `return/refund` 降级为 `consult`，改判沿用原置信度（不再多花一次 LLM 兜底，否则可能被兜回 return）。
  疑问句式**刻意不放裸「怎么」**，否则既有用例「退款进度怎么样了」（期望 `refund`）会被误降级。
- **两处「永久误判」的修复**（因为 `conf ≥ 0.5` 时 LLM 兜底根本不触发，只能靠词表自洽）：
  - 补 `logistics` 的 `("发货了", 3.0)`：「这个订单发货了吗」里「订单」2.5 > 「发货」2.0，
    conf=0.556 已过闸门 → 断成 `order`。补长短语后（长词优先，覆盖同句的「发货」不再重复计分）logistics 3.0 > order 2.5。
  - 补 `refund` 的 `("退货的钱", 3.5) / ("退的钱", 3.5)`：「上次退货的钱去哪了」问的是钱到没到，
    「退货」3.0 独占命中 → `return` conf=1.0 被直判。补后 refund > return。
- **「售后规则问句」≠「投诉」**：`RULE_PATTERNS["consult"]` 收录「描述不符」（对应 PLAT-009 售后规则）。
  此前该表述规则零命中 → LLM 兜底判成 `complaint` → 直接建工单，用户拿不到任何政策说明
  （实测「收到的东西和描述不符怎么办」从 `complaint/LLM` 改判 `consult/rule`）。
- **手机售后语汇**：`consult` 补齐 `维修/碎屏/进水/激活/序列号/发票/开票/配件/以旧换新/延保/网点/拆封` 等。
  这批问句此前一条规则都不命中（conf=0.1）→ 每次都白花一次 LLM 兜底（实测约 +7s）；补词后由规则直判，
  只省延迟、不改变意图归属。
- **LLM 兜底 prompt 带 few-shot**：system prompt 含 4 条「易混淆处的判定口径」
  （问规则 vs 提诉求、退货 vs 退款、投诉 vs 咨询、招呼与应诺→chat）+ 7 条判定示例
  （刻意用与评测集不同的句子，避免把评测集答案喂进去）。
- **实测（81 条评测集：A 组 46 条规则重合 / B 组 35 条同义改写）**：
  `rule` 模式整体 76.5%（A 组 100%，B 组 45.7%，refund 召回 100%）；
  `hybrid` 模式两轮均 81/81；LLM 兜底占比 25%；平均 0.24~0.35s/条。

### 4.4 DST 对话状态跟踪（`agent/dst/`）

- 结构状态 `{intent, slots, stage, tool_call_count, answer}`，**MySQL 持久化，重启不丢**。
- 阶段状态机：`COLLECTING → CONFIRMING → EXECUTING → DONE`。
- **确定性退货/退款主流程**：收集缺漏槽位 → 核对确认 → 登记工单，全程不依赖 LLM 工具循环，
  杜绝泛化成「查物流 / 破损鉴定」，并避免投诉场景死循环卡死（超时问题根因）。
- **槽位提取 = LLM 主 + 规则辅**，三层协作（`slot_extractor.py` / `slot_schema.py`）：
  1. `_rule_extract` 用正则兜底补漏（`setdefault`，不覆盖 LLM 结果）；
  2. pattern 校验拦截 LLM 幻觉（如把订单号里的数字当金额、把手机号当订单号）；
  3. `merge_new_slots` 合并后自愈清理（剔除不合法槽位）。
  `reason` 词表按「智能手机及配件」定位重写（含碎屏/进水/无法开机/闪退/电池鼓包等高频故障，
  以及「七天无理由」「7天无理由」两种写法），已删除服装品类的尺码词与裸「破」等噪声。
- **归还/退款轨迹按工单类别渲染**：`RETURN_STAGES` 带 `required` 字段，`stage_flow(category)` 按
  `return`（6 步）/ `refund`（4 步）出不同时间线，退款类工单不再显示「寄回商品 / 商家验收」等无关待办。
- **幂等防护**：同订单同账号已有在办工单时不重复建单。
- **归属校验前置**：拿到 `order_id` 后立刻校验归属，非本人订单直接拒绝并清空 `order_id`；
  越权访问返回 `ORDER_FORBIDDEN` 且**不泄露任何订单字段**（客户姓名、金额、物流单号）。
- **金额与订单号用正则强校验**（DST 槽位层面拦截 LLM 幻觉）。

### 4.5 人工会话（转人工）

- AI 主动升级转人工时**必须建单**并写入开场白，工单才会出现在客服工作台待处理列表。
- 转人工提示话术统一为：「正在为您转接人工客服（工单号 {ticket_id}），请稍候……」——
  **不得**在客服实际接入前声称「已接入人工客服」。
- 客服接手时可拉取转人工前的机器人会话记录（`/tickets/{id}/context`），无需用户重复描述。
- 下行通道为 WebSocket `/ws/handoff/{ticket_id}`（令牌放 URL query，因浏览器 WS 构造器不支持自定义 header）；
  发送消息仍走 HTTP POST。断线重连用 `after_id` 增量补消息。

### 4.6 健壮性与安全

- **三态熔断器**（`CLOSED/OPEN/HALF_OPEN`）：为 LLM、Redis 独立配置
  （`CIRCUIT_FAILURE_THRESHOLD=5` / `CIRCUIT_RECOVERY_TIMEOUT=60` / 半开探测 1 次），熔断自动降级。
- **合规护栏**（`OUTPUT_GUARD_*`，客服场景必备）：敏感词过滤、承诺校验
  （拦截「全额退款 / 免费换新 / 双倍赔偿 / 保证 X 天到货」等未授权承诺）、金额事实核查
  （`¥\d+` 必须有依据）。
- **接口限流**（60 次/分/Key，`RATE_LIMIT_PER_MINUTE`）。
- **鉴权分层**：对话类接口 `Depends(require_user)`（未登录 401）；管理类接口
  `Depends(require_admin)`（非管理员 403）；`/health` 与 `/metrics` 公开。
- **跨账号隔离**：订单/工单查询一律按登录账号过滤（含 `AND user_id=?`），admin 的工单对
  customer/visitor 不可见。
- **路径穿越防护**：删文档类操作校验路径落在受管 `data/` 目录内（`Path.resolve()` + `is_relative_to()`）。
- **无据可答诚实拒答**：知识库无可靠依据时输出「没有查到可靠的资料，不敢给您不准确的答复」并转人工
  （`should_escalate=True`、gate-level=low），**杜绝编造**。

### 4.7 性能优化

- 对话 / Agent 流式输出（SSE），HTTP 连接与流式生成解耦：客户端中途断开，该轮历史仍完整落库
  （RAG/Agent 两条入口同一套生产者骨架，历史在生产者任务内直接写入，不用 `BackgroundTasks`）。
- 意图分类 `force_intent` 复用 + 确认轮跳过槽位提取 LLM，**主线内追问零额外 LLM 开销**
  （确认轮由约 7s 降至约 0.2s）。
- RAG 同问结果缓存（内存 LRU，`CACHE_MAXSIZE=1000` / `CACHE_TTL=300s`，命中秒回）；缓存命中分支
  同样返回 `X-Gate-*` 头，保证客户端能拿到检索元数据。
- 异步检索线程池按 CPU 核数自适应（BM25 CPU 密集 = cores；向量 IO 密集 = cores×2）。

---

## 五、API 一览

后端路由**不带 `/api` 前缀**（前端由 Vite 代理把 `/api` 去前缀后转发）。共 30 个端点。

### 公开（无鉴权，供监控）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查：`redis / vector_store / mysql / llm` 四项组件状态 |
| GET | `/metrics` | Prometheus 指标 |

### 认证（`access/routes/auth.py`）

| 方法 | 路径 | 鉴权 | 说明 |
|---|---|---|---|
| POST | `/auth/register` | — | 注册（首次启动自动创建演示账号） |
| POST | `/auth/login` | — | 登录，返回 `token` + `user` |
| POST | `/auth/logout` | 登录 | 注销并失效令牌 |
| GET | `/auth/me` | 登录 | 当前登录用户 |

### 会话类（`Depends(require_user)`，未登录 401）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/rag/stream` | RAG 流式问答 |
| POST | `/agent/stream` | Agent 流式对话（含 DST 状态跟踪与确定性主流程）——前端唯一使用的对话入口 |
| GET | `/history/sessions` | 会话列表（仅本人） |
| GET | `/history/messages` | 指定会话的历史消息 |
| DELETE | `/history/sessions/{session_id}` | 删除会话 |
| POST | `/history/sessions/{session_id}/close` | 结束会话 |
| GET | `/my/orders` | 本人名下订单列表（只读，最近下单在前） |
| POST | `/handoff/request` | 用户主动转人工（已有进行中会话则复用，否则建单开启） |
| GET | `/handoff/active` | 当前进行中的人工会话工单（无则 `ticket: null`） |
| GET | `/tickets/{ticket_id}/messages` | 会话消息（`after_id` 支持断线增量补消息） |
| GET | `/tickets/{ticket_id}/context` | 转人工前的机器人会话记录 |
| POST | `/tickets/{ticket_id}/messages` | 人工发送消息 |
| POST | `/handoff/{ticket_id}/close` | 关闭人工会话 |
| WS | `/ws/handoff/{ticket_id}?token=…` | 人工会话下行通道（令牌走 URL query） |

### 管理类（`Depends(require_admin)`，非管理员 403）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/tickets` | 工单列表（分页，可按状态筛选） |
| GET | `/tickets/{ticket_id}` | 工单详情（含处理备注） |
| GET | `/orders` | 后台订单列表（分页，可按状态筛选） |
| GET | `/users` | 用户列表（关键词 + 分页） |
| DELETE | `/users/{user_id}` | 删除用户（不能删自己） |
| POST | `/knowledge/upload` | 上传文件到 `data/` 后触发统一增量入库 |
| GET | `/knowledge/documents` | 知识库文档列表（入库状态） |
| GET | `/knowledge/documents/{doc_id}/chunks` | 某文档的分块列表 + 从向量库取回每块正文 |
| DELETE | `/knowledge/documents/{doc_id}` | 删除文档（清向量库分块 + 删 MySQL 状态 + 移除源文件） |
| GET | `/handoff/sessions` | 客服工作台会话列表（带最后一条消息，按最新倒序） |

请求体示例：

```json
// POST /agent/stream
{ "question": "订单 SO20241120005 到哪了", "session_id": "u_001", "api_key": "sk-..." }
```

---

## 六、环境约定与依赖服务

### 6.1 统一 Python 环境

全仓库共用**唯一**虚拟环境 `AI大模型与智能体开发/.venv`，后端与前端脚本共用，不再单独建 venv。
密钥由模板复制后填值（`.env` 不入库），且以 `BASE_DIR` 显式绑定路径加载，与进程启动时的工作目录无关。

| 项目 | 约定 |
|---|---|
| 环境位置 | 仓库根 `.venv`（不入库） |
| 解释器 | `.venv\Scripts\python.exe` |
| 依赖清单 | `RagServerSystem/requirements.txt` |
| 密钥文件 | `RagServerSystem/.env`（由 `.env.example` 复制） |

首次安装 / 重装：

```powershell
cd "d:\RAG-Windows\AI大模型与智能体开发"
python -m venv .venv                                    # 已存在可跳过
.\.venv\Scripts\Activate.ps1
pip install -r RagServerSystem\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

日常运行两种方式：

```powershell
# 方式 A：先激活环境，再运行脚本
.\.venv\Scripts\Activate.ps1
cd RagServerSystem; python main.py

# 方式 B：不激活，直接用解释器绝对路径（脚本/CI 更稳，不依赖 shell 是否已激活）
& "d:\RAG-Windows\AI大模型与智能体开发\.venv\Scripts\python.exe" "d:\RAG-Windows\AI大模型与智能体开发\RagServerSystem\main.py"
```

> PowerShell 注意：路径以 `.` 开头时**必须加调用运算符 `&`**（`& .\.venv\Scripts\python.exe`）。
> 直接写 `.venv\Scripts\python.exe` 会被当成模块名，报 `The module '.venv' could not be loaded`。

IDE（VS Code / Cursor）：把 Python 解释器设为
`d:\RAG-Windows\AI大模型与智能体开发\.venv\Scripts\python.exe`，各子目录的运行 / 调试即共用同一环境，
避免「找不到 langchain_chroma」这类问题。

### 6.2 依赖服务

| 服务 | 必需性 | 说明 |
|---|---|---|
| MySQL 8 | **必需（硬依赖）** | 订单 / 工单 / 用户 / 会话历史 / DST 状态 / 文档入库状态；连不上则相关接口直接失败，**不做降级** |
| Redis | 可选 | 会话历史热缓存；未启动时回源 MySQL，功能不受影响（fail-open）。若 `requirepass` 需配 `Redis_password` |
| DashScope API Key | **必需** | LLM 与 Embedding（走 OpenAI 兼容端点） |

### 6.3 配置 `.env`

```powershell
cd RagServerSystem
Copy-Item .env.example .env
# 编辑 .env：DASHSCOPE_API_KEY、MySQL 连接信息、Redis_password（如启用）
```

`.env` 关键项：

```env
DASHSCOPE_API_KEY=sk-xxxx
Redis_password=xxxx        # Redis 开启 requirepass 时必填
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=rag_user
MYSQL_PASSWORD=xxxx
MYSQL_DATABASE=rag_db      # 业务表与文档入库状态共用此库
```

### 6.4 准备 MySQL（必需）

```sql
CREATE DATABASE IF NOT EXISTS rag_db DEFAULT CHARACTER SET utf8mb4;
CREATE USER IF NOT EXISTS 'rag_user'@'%' IDENTIFIED BY 'xxxx';
GRANT ALL PRIVILEGES ON rag_db.* TO 'rag_user'@'%';
FLUSH PRIVILEGES;
```

业务表（`orders` / `tickets` / `ticket_messages` / `users` / `conversations` / `messages` /
`conversation_state`）与文档入库状态表（`documents` / `doc_chunks`）都在首次启动时自动建。

**数据一致性约定**：MySQL 状态表与 Chroma 内容始终保持一致，增量去重与删除对齐依赖它。
历史以 MySQL 为权威存储（`conversations` + `messages`），Redis 仅作最近 N 条的热缓存。

---

## 七、快速开始

### 7.1 构建知识库

将文档放入 `data/` 目录后（支持 `.txt / .md / .pdf / .docx / .xlsx / .csv`）：

```bash
python -m ingestion.service
```

统一入库服务 `DataLayer.ingest()` 两种模式：

| 模式 | 行为 | 是否需要重启服务 |
|---|---|---|
| `incremental`（默认） | 按 MySQL 状态比对，只处理新增/修改/删除的文件；删除旧 chunk 后重加，**不动集合** | **不需要**——运行中的进程能直接读到新段 |
| `full` | 先 `aclear()` 清空向量库 + `reset()` 清空 MySQL 状态，再全量重建（`__main__` 走此模式） | **需要重启**（集合被删除重建后，进程持有的 handle 不自动刷新，会一直返回空内容） |

### 7.2 启动 API 服务

方式一：本地直启（需自备 MySQL/Redis）

```bash
python main.py          # 默认 8000 端口
```

方式二：Docker 一键编排（app + MySQL + Redis，无需本机装数据库）

```bash
cp .env.example .env    # 填 DASHSCOPE_API_KEY；Redis_password 与 deploy/redis/conf/redis.conf 的 requirepass 一致
docker compose up -d --build
```

- 首次启动 app 容器检测到向量库为空会**自动入库**（容器入口 `docker/entrypoint.py`），
  `docker compose logs -f app` 看到 `Uvicorn running` 即就绪；
- MySQL 暴露在宿主机 **3307**（避开本机已有 3306），应用始终走容器内网络；
- 更新 `data/` 语料后重建索引：`docker compose run --rm app python -m ingestion.service`。

### 7.3 启动前端

```bash
cd frontend
npm install
npm run dev             # 用户端 → http://127.0.0.1:5173
npm run dev:admin       # 管理端 → http://127.0.0.1:5174（可选，见下）
```

Vite 把 `/api` 代理到 `http://127.0.0.1:8000`（含人工会话的 `/api/ws/handoff/...` WebSocket），
所以**必须先启动后端**。后端不托管 `frontend/dist`，生产部署需自行用 nginx 托管该目录。

**演示账号**（首次启动后端时自动创建）：`admin/admin123`（管理员）、
`customer/customer123`（买家，3 张种子订单挂其名下）。

### 7.4 同时登录用户与管理员（双端口）

登录态存在 `localStorage` 的 `auth_token` / `auth_user` 两个 key 上，而 **localStorage 按 origin
（协议 + 主机 + 端口）隔离**。同一端口下所有标签页共享同一份存储，所以第二个账号登录会覆盖第一个——
表现为「切换角色后被路由守卫送回原侧」。这是用 localStorage 存 JWT 的网页应用的通用行为，非本项目缺陷。

因此**端口不同即 origin 不同**，起两个 dev server 就能在同一浏览器里同时持有两套独立登录态：

```powershell
npm run dev          # 终端 A：http://localhost:5173  → 登 customer，开用户聊天页
npm run dev:admin    # 终端 B：http://localhost:5174  → 登 admin，开 /admin/tickets
```

两个实例跑的是同一份前端代码（5174 不是单独的管理端构建产物），只是用 admin 账号登录后会被路由守卫
送到 `/admin/tickets`；两个端口的 `/api` 代理与 WebSocket 都指向同一个后端 8000。

注意点：

- `dev:admin` 带 `--strictPort`：5174 被占用时直接报错，而不是悄悄换到 5175（端口变了登录态又会串）。
- 换角色前记得在对应端口**登出**旧账号，否则该端口仍是旧身份。
- 不想多起一个 server 时，也可用**无痕窗口 / 另一个浏览器**（不同 profile → 独立 localStorage）。

### 7.5 本地模型

`models/` 不入库（约 656 MB）：本地 embedding 模型（`bge-base-zh-v1.5`、`bge-small-zh-v1.5`）与
重排模型（`bge-reranker-base`，即 `RERANK_MODEL_PATH` 默认值；另有 `bge-reranker-v2-m3` 备选）。
当前 `EMBEDDING_BACKEND` 默认为 `"dashscope"`（云端约 0.5s），嵌入模型仅在切回 `"local"` 时使用；
重排模型在 `RERANK_MODE="cross_encoder"`（默认）时使用，两者均为 CPU 本地推理。
模型路径由 `config/settings.py` 中的**绝对路径**指定，换机器需同步修改。

---

## 八、前端说明

### 8.1 页面与路由

| 路径 | 页面 | 说明 |
|---|---|---|
| `/` | `views/user/ChatView.vue` | 买家端智能对话主界面（SSE 流式，需登录） |
| `/my/orders` | `views/user/OrderView.vue` | 我的订单（只读：商品/金额/状态/物流 + 申请退货） |
| `/login` | `views/LoginView.vue` | 登录 / 注册 |
| `/admin/workbench` | `views/admin/WorkbenchView.vue` | 人工客服工作台（WebSocket 实时收发） |
| `/admin/tickets` | `views/admin/TicketView.vue` | 工单管理（进入管理端的默认页） |
| `/admin/orders` | `views/admin/OrderView.vue` | 订单管理 |
| `/admin/kb` | `views/admin/KbDocsView.vue` | 知识库文档管理 |
| `/admin/users` | `views/admin/UserView.vue` | 用户管理 |
| `/admin/monitor` | `views/admin/MonitorView.vue` | 监控（后端 `/metrics` 指标） |

路由守卫（`src/router/index.js`）：`requiresAuth` 未登录跳 `/login` 并带回跳地址；`requiresAdmin`
非管理员提示后回首页；已登录再访问 `/login` 按角色送回对应首页。

### 8.2 与后端的约定

- axios 实例 `baseURL = '/api'`；请求拦截器自动加 `Authorization: Bearer <token>`；
  响应 401（登录/注册接口本身返回 401 属正常校验，不跳转）清空令牌并跳登录页。
- 开发环境由 Vite 代理把 `/api` **去掉前缀**后转发到 `http://127.0.0.1:8000`，因此后端路由不带
  `/api`（如 `/agent/stream`）。人工会话的下行 WebSocket `/api/ws/handoff/{ticketId}` 也走该代理
  （需 `ws: true`，否则握手被代理吞掉），发送仍走 HTTP POST。
- 令牌存 `localStorage`（`auth_token` / `auth_user`）。浏览器 WebSocket 构造器不支持自定义
  header，因此令牌放在 URL query 上。

### 8.3 开发命令

```bash
npm install
npm run dev        # 用户端 → http://127.0.0.1:5173
npm run dev:admin  # 管理端 → http://127.0.0.1:5174（可选）
npm run build      # 产物 dist/
npm run preview
```

Node 版本要求 `^22.18.0 || >=24.12.0`（见 `package.json` 的 `engines`）。

---

## 九、测试与校验

### 9.1 单元测试与在线端到端

```bash
# 单元测试（核心纯逻辑：意图分类/查询分解/熔断/槽位/状态机/历史持久化与归属隔离等）
# 注意：历史持久化用例会真连 MySQL（连不上则自动 skip），并在前后清理测试专用会话行
python -m pytest tests -q          # 178 passed

# 在线端到端（需服务已启动、MySQL/Redis 可用、DashScope key 有效）
python e2e_online_test.py          # S0 登录 + S1-S5 全部通过
python e2e_business_eval.py        # 真实业务问题评测集（53 题，/agent/stream，灰盒判定）

# 健康检查
curl http://127.0.0.1:8000/health
```

`tests/` 覆盖：意图分类、查询分解与类别推断、DST 状态机与槽位 schema、熔断器、
订单归属校验、我的订单、工单工具转人工、人工会话路由、会话历史存储、流式前导剥离。

### 9.2 离线评测与验证工具

以下是**开发期工具，服务运行时不依赖**（`main.py` 不 import 它们；`.dockerignore` 已排除其中部分）：

| 脚本 | 用途 | 前置 | 何时该跑 |
|---|---|---|---|
| `intent_eval.py rule` | 意图规则层评测：81 条标注集（A 组 46 规则重合 / B 组 35 同义改写），**秒级、零 LLM 费用** | 无 | 改 `intent_rules.py` 词表后回归 |
| `intent_eval.py` | 追加 hybrid 模式（真实调 LLM），输出分流比与「兜底救回 / 改错」明细 | 无 | 改 LLM 兜底 prompt 后 |
| `intent_calibration.py` | 置信度校准：可靠性表 + ECE + 兜底阈值扫描（复用 `intent_eval.CASES`） | 无 | 调 `INTENT_LLM_FALLBACK_CONFIDENCE` 时 |
| `verify_dataset.py` | 离线检索命中验证（传 `fresh` 会**清空重建**向量库） | 无 | 改知识库 / 分块策略后 |
| `e2e_online_test.py` / `e2e_business_eval.py` | 在线端到端（S0-S5 / 53 题业务评测） | **服务须在 8000 运行** | 部署验收 / 改业务逻辑后 |
| `evaluation/service.py` | RAGAS 离线评测：30 题评测集（`evaluation/eval_set.json`）跑**双臂对比**（关闭 / 开启 Cross-Encoder 重排），输出四项指标 + 平均检索耗时 | 向量库已建、判分模型 key 有效 | 改检索 / 重排 / 生成层后量化效果 |

> 「意图识别准确率多少」这类必须能报数字的问题，证据来自 `intent_eval.py` 与 `intent_calibration.py`
> （实测：`rule` 76.5% / `hybrid` 81/81 / ECE 约 0.047 / LLM 兜底率 25%）；断言「某个句式必须判对」
> 的单测不能替代它——单测用例是为规则量身写的，只反映「回归没坏」而非泛化能力。

---

## 十、已知边界与优化点（诚实清单）

以下为当前实现的可接受局限与后续优化方向，面试或评审时可主动说明取舍与改进路径。

**架构 / 部署边界**

- 订单、工单、DST 为**自建 MySQL 表的 mock 数据**（文档入库状态另用同一 MySQL 库的
  `documents`/`doc_chunks`），未对接真实 ERP/OMS。
- Chroma 为**单机向量库**，生产建议迁移 Milvus / Elasticsearch 集群。
- 服务为**单进程 uvicorn**，未配置多 worker 与压测指标（可用 `gunicorn -w N` 扩展）。
- Docker 编排为**单机 Compose**：内存限流/缓存在多实例下不共享，分布式部署需换 Redis 计数与共享缓存；
  容器内 MySQL 访问层仍是 pymysql 短连接（业务库）+ aiomysql（文档状态）两套并存，未做连接池统一。
- `config/settings.py` 中本地模型路径写死为**绝对路径**，换机器需同步修改。

**已识别的性能 / 准确性优化点**

- **Embedding 吞吐**：切回本地 CPU `bge-base-zh-v1.5` 时单次嵌入约 13~17s，入库偏慢；
  当前默认已改用 DashScope 云端 embedding（约 0.5s）。
- **置信度门控阈值**：采用 `vector_score = 1 - distance/2`（余弦相似度）绝对分
  （`high ≥0.80 / low <0.55`）+ 排序 margin(0.08)。旧实现用 `1/(1+distance)` 是量纲错配，
  会把 cos 0.34~0.79 压成 0.43~0.73，high/low 同时失效，已修正；阈值仍需要更多业务样本持续调优。
- **检索延迟**：纯 CPU 环境下 **Cross-Encoder 重排是主要耗时项**——平均单题检索耗时由关闭重排时约 0.4s
  升至开启后约 11s（含首次模型懒加载约 20~25s）。已用流式输出 + 同问缓存缓解，进一步可引入缓存层、
  向量索引优化，或将重排切到 GPU / 更小模型（`RERANK_MODE="none"` 可临时关闭重排）。
- **查询分解的边界**：只按强分隔符 + 显式连接词切分，**逗号分隔的多问题**
  （「电池多大，运费怎么算？」）不拆（逗号作分隔会把单句切碎）；子问过多时上下文仍会被
  `MAX_CONTEXT_TOKENS` 截断（已用交错合并保证每个子问先保头部）。
- **多子问门控的取舍**：`low` 子问被剔除后其资料一并丢弃，用户只会拿到可答子问的答复，
  **不会**得到「这部分没查到」的显式说明（比裸转人工体验好，但信息不透明）；
  若全部子问都 `low` 则整轮转人工，不放松——宁可让用户多问一次，也不拿弱证据糊弄。
- **文档类别预过滤的取舍**：`$in` 并集会让跨域问题的候选池略微变大（换取「不漏掉答案所在文档类」），
  索引继续增长后建议按类别分库/分 collection 而非依赖 metadata 过滤。
- **`combined_splitter` 未作默认**：它属于可选策略，收益依赖「父块显著大于子块」，需要万字级长文档才体现。
  当前语料为 8 篇短文 + 2 份表格（约 40KB），实测父块 avg 425 / 子块上限 400，18 个父块里 9 个只切出
  1 个子块，分层已退化为重复存储；且父子回填（`parent_cache.json`）目前只在 `/rag/stream` 生效，
  前端调用的 Agent 路径直接使用命中的块内容。换成长文档语料后可切回该策略。
- **`问政策 vs 要办理` 的判定边界**：靠「动作闸门 + 合并问法正则」判定，整句不含办理动作措辞。
  已知剩余边界：①**主线流程进行中**（DST `stage=COLLECTING/CONFIRMING`，如先说了「我要退货」
  再问「退货麻烦吗」）会复用主线意图，评价类问题不会切走——回「请补充订单号」引导，
  DST 状态不丢、方向无害但不够聪明；要让它在不丢主线状态的前提下答政策，需在主线 handler 内联知识检索，
  属于后续优化；②词表外的表述（如「退个货折腾吗」）仍可能落进退货主流程，用户补充诉求即可继续。
- **意图规则表需持续回归**：因 `conf ≥ 0.5` 时 LLM 兜底不触发，规则表内任何不自洽的权重都会造成
  永久误判（已修两例）。新增词条后应跑 `intent_eval` 回归，避免引入新的权重冲突。

面试若被问「哪里还能做得更好」，重点讲这三条（Embedding 切换云端、门控阈值调优、
分布式向量库/多 worker），能体现对生产化瓶颈的真实判断。

---

## 十一、后端分层详解

```
RagServerSystem/                 # 每一层 = 一个同名顶层包，目录形状即架构图
├── access/                     # 接入层：FastAPI 装配 · 路由 · 鉴权 · 限流
│   ├── app.py                  # 应用装配 + /rag/stream、/agent/stream、/health、/metrics、/history/*
│   ├── routes/                 # auth（登录/注册/用户管理）· admin（工单/订单/知识库）· handoff（人工会话）· user_orders（我的订单）
│   ├── middleware.py           # 限流 + Prometheus 指标中间件
│   └── history.py              # 会话 ID 作用域 + 历史读写助手
├── service/                    # 服务层
│   ├── turn_service.py         # 轮次编排（Agent 调度 / RAG 流式 + 会话历史读写）
│   ├── handlers/               # 6 条确定性分支（投诉/进度/订单/名下订单/退货退款/知识）+ context
│   ├── handoff_service.py      # 人工会话生命周期（建单/复用/写开场白）
│   └── cache.py                # RAG 响应缓存（LRU + TTL）
├── agent/                      # Agent + DST
│   ├── service.py              # 统一 Agent（5 工具）
│   ├── intent_classifier.py    # 细粒度意图分类（打分与级联）
│   ├── intent_rules.py         # 意图规则数据（词表 + 合并后的问法正则）
│   ├── tools/                  # search_knowledge / order_query / ticket_query / ticket_tool
│   └── dst/                    # DST 状态机（slot_schema / slot_extractor / dst_manager / order_validator）
├── generation/
│   └── service.py              # 生成层：流式生成 + 合规护栏
├── retrieval/                  # 检索层：router/BM25/向量/混合/重排序/查询增强
│   └── service.py              # RetrievalService（查询分解 → 逐子问检索 → 置信度门控）+ infer_doc_categories
├── ingestion/                  # 数据层：加载 · 清洗 · 分块 · 向量入库
│   ├── service.py              # 统一入库服务（全量/增量，DataLayer）
│   ├── loader/                 # 文档加载（file_watcher / mysql_data_layer / mysql_data_loader / table_loader）
│   └── splitter/               # 5 种分块策略 + 工厂
├── shared/                     # 跨层共享：constants（确定性业务规则 + 多问题拆分）+ reply_templates（话术）+ schemas（请求/响应 DTO）
├── evaluation/
│   └── service.py              # 评估层：RAGAS 评估
├── infrastructure/
│   ├── vector_store/           # Chroma 异步封装
│   ├── EmbeddingService/       # 嵌入服务
│   ├── redis/                  # Redis 连接池（热缓存 / 计数等）
│   ├── mysql_history.py        # 对话历史后端（MySQL 为准 + Redis 热缓存）
│   ├── base_chat_history.py    # 历史后端基类（含查询改写）
│   ├── chat_history_factory.py # 历史后端工厂
│   ├── mysql_store.py          # 订单/工单/用户/DST/会话消息 持久化（pymysql）
│   └── sql/                    # MySQL 文档入库状态（aiomysql）
├── utils/                      # llm_factory / circuit_breaker / security / metrics / 线程池 等
├── config/settings.py          # 全局配置（分块/检索/门控/LLM/DST/熔断阈值）
├── logs/log_config.py          # 日志配置（命名 logger → 落 logs/Log_server/<module>/）
├── tests/                      # pytest 单元测试（178 passed）
├── e2e_online_test.py          # 在线端到端（S0-S5）
├── e2e_business_eval.py        # 真实业务问题评测集
├── docker/entrypoint.py        # 容器入口（向量库为空时先自动入库再起服务）
├── Dockerfile                  # 应用镜像（python:3.12-slim + CPU torch）
├── docker-compose.yml          # 一键编排（app + MySQL + Redis）
└── main.py                     # 服务入口
```