# WORKLOG — 工作日志

> 每轮迭代在文末追加一节：时间戳、完成内容、测试结果、问题、下一步。

## iter0 — 2026-09-26 01:30–02:00（主会话完成）

**完成：**
- 环境确认：Windows + Python 3.13.2 + git 2.49；网络可用；机器有 GPU 但本任务全程 CPU（用户明确要求不碰 GPU）；环境变量无 LLM API key，故 LLM 判分今晚只做干跑路径。
- 缺口取证（详见 GAP_PROOF.md）：GitHub API 实测 "chinese text chunking"/"中文 分块 RAG" 相关仓库最高 2 star；jieba 35,169 star 停更于 2024-08；RAGFlow 91,298 star 为引擎非库；PyPI `worddael`/`chinese-chunker`/`zh-chunk` 均未注册。
- 包实现（src/worddael/）：
  - `types.py` Chunk（text/start/end/seq/strategy/meta，偏移不变量写进 docstring）
  - `counters.py` CharCounter 启发式 + JiebaCounter/TiktokenCounter 可选
  - `sentences.py` 中文句界（。！？；… + 引号闭合；ASCII `.` 不切）+ markdown 分节（heading path、代码围栏原子）
  - `chunkers.py` Token/Sentence/Recursive/Markdown/Semantic 五策略，统一偏移保证；overlap 为字符级回看，语义切分带 min_chars 防碎片
  - `io_utils.py` chunk_file/to_jsonl；`cli.py` worddael 命令
  - `eval/` 内置 8 篇原创中文语料 24 问、纯 Python BM25、recall@k、LLMJudge（干跑/实跑两态）、report 生成器（含 fixed-window 基线与 toy 语义）
- 测试：**70 passed**（含 8 seed × 4 策略的偏移/覆盖模糊测试）；ruff 0 error。
- 工具链：venv 在 `.venv/`（Git Bash 用 `./.venv/Scripts/python.exe`）；`pip install -e ".[jieba,dev]"` 已装好。

**诚实记录：**
- 内置迷你语料各策略 recall@3 均为 0.958（24 问对 23），**无区分度**，只能作冒烟测试；有区分度基准是 iter2 的目标。README/GAP_PROOF 已如实声明，禁止把该数字当质量宣传。
- `python -m worddael.eval.report` 目前有 jieba 的加载日志输出，属正常。
- 尚未 git 提交（由 iter0 收尾统一 init+commit）。

**给下一轮（iter1）的提示：**
- 先读 ROADMAP.md 找第一个未勾选项；提交信息格式 `iterN: ...`；测试命令见上；不要装 >50MB 的包。

**iter0 补充（02:10，主会话收尾前最后验证）：**
- k=1 复测（max_chars=200）：fixed-window 0.917 > sentence/token 0.958？——实际为 sentence=token=0.958 > fixed=0.917 > recursive=0.875 > semantic-toy=0.833。**k=1 下 recursive 输给 fixed-window**。这说明 8 篇迷你语料不但无区分度、还可能误导策略评价；iter2 扩基准时务必用更大文档（≥1000 字/篇）与更细预算（max_chars≤150）再测，并在 results.md 里同时报告 k=1 与 k=3。之前"各策略 0.958 打平"的说法只对 k=3 成立，特此更正记录。

## iter1 — 2026-09-26 03:00–03:06（夜间会话第 1 轮）

**完成：**
- 新增 `src/worddael/embedders.py`：
  - `HashingEmbedder`：字符 trigram + crc32 哈希到定长向量。**修正了 iter0 遗留的隐患**——report.py 原 toy embedder 用 Python 内建 `hash()`，跨进程随机（PYTHONHASHSEED），换成 crc32 后跨进程确定。
  - `OpenAICompatibleEmbedder`：任意 OpenAI 兼容 /embeddings 端点；key 解析惰性（显式传入 > 环境变量 `api_key_env`）；batch_size 分批；HTTPError/URLError 包装为 `EmbedderError`（带状态码与响应片段）；空输入不发请求。
  - 两者均可直接作 `SemanticChunker(embed_fn=...)`。
- 测试 +12：mock 全覆盖（批量切分与顺序、鉴权头来自环境变量、显式 key 优先、缺 key 报错、空输入零请求、HTTP 500 包装、响应长度不符、Hashing 确定性/归一化/相似度排序、与 SemanticChunker 的胶水）。**82 passed，ruff 0 error。**
- `eval/report.py` 的 toy embedder 改为复用 `HashingEmbedder`（删除重复实现）。
- README 语义切分示例更新（含 Zhipu endpoint 示例）；SemanticChunker docstring 指向 embedders 模块。

**问题与修复：**
- 测试暴露一个真实设计缺陷：`__call__ = embed` 类级绑定导致实例级覆盖 `.embed` 无效（mock 失效即症状）。已改为运行时委托 `__call__ -> self.embed(...)`。这类"实例覆写生效"的语义对用户 monkeypatch/子类化都重要。

**下一步：** iter2（把 8 篇迷你语料扩成 ≥25 篇 ≥75 问的有区分度基准，k=1/k=3 双口径，结果如实写入 benchmarks/results.md）。

## iter2 — 2026-09-26 03:05–03:14（夜间会话第 2 轮）

**完成：**
- 语料扩容：8 篇 → **25 篇原创中文文档 / 75 问 / 9,044 字**（新增健身房、相机、考勤、多肉、高铁、游泳、退换货、图书馆、养猫、面试、灭火器、咖啡、汽车保养、签证、婚礼、天文、仓库消防 17 篇）。全部原创自写，零版权风险；gold 逐字校验由新增的 `test_corpus_integrity` 永久守护。
- report.py 升级：`full_report()` 双口径（k=1 + k=3）+ 语料元数据 + 每策略失败清单；`ends_on_punct_frac` 数值化。
- 真实结果已写入 `benchmarks/results.md`（max_chars=150 / overlap=30）：
  - **k=1 出现区分度：recursive 0.947 > fixed-window 0.907 = sentence 0.907 > token 0.880 > semantic-hash-toy 0.680**
  - k=3 时各策略趋同（0.96–0.973），说明块数更少时结构优势被 top-3 容错吸收——两条曲线放在一起才有信息量。
  - toy 语义切分 0.68 印证其"仅用于跑通管线"的定位。
- 有个诚实修正：iter0 补充里"k=1 下 recursive 输给 fixed-window"是小语料（8篇、文档短）的失真；扩容后 recursive 在 k=1 领先。教训：**基准规模直接改变结论，任何单一小样本上的结论都不可信**。

**问题与修复：**
- `ends_on_punct` 最初统计 c.text[-1]，被偏移精确切分落在块尾的换行符拉低（recursive 只有 0.369）；改为 rstrip 后统计（这才是对 RAG 有意义的口径），recursive 恢复到 ~0.99，fixed-window 维持 ~0.36，结构性差异清晰。
- hospital 文档一处 gold 与原文不连续，integrity 测试当场抓住并修复——该测试的价值得到验证。

**下一步：** iter3（LLM 判分实跑引擎：限速/重试/JSONL 断点续跑/成本熔断，unittest.mock 测 HTTP，写 docs/eval_guide.md）。

## iter3 — 2026-09-26 03:15–03:25（夜间会话第 3 轮）

**完成：**
- 新增 `src/worddael/eval/judge_runner.py`：批量判分引擎
  - `JudgeRunner`：rpm 进程内限速（墙钟）、指数退避重试（耗尽后记 error 记录，不丢数据）、JSONL 断点续跑（按 pair key 跳过已完成，容忍被杀进程的残行）、成本上限熔断（实跑模式下按启发式估算，超限安全停止）
  - `build_pairs`：策略×文档×问题×块 的判分对展开
  - `summarize`：按策略聚合 {n, errors, mean_score}
  - CLI `python -m worddael.eval.judge_runner`：默认干跑（无 key 可用），--live 需 key
- `docs/eval_guide.md` 评测指南：干跑→估价→实跑三段式、参数语义表、解读纪律（干跑实跑不得混用、LLM 判分偏差需与 BM25 recall 交叉验证、报数字必须带语料规模）
- 测试 +10（全部 fake 注入，零网络零 sleep）：续跑跳过、限速 sleep 序列、退避 2s/4s、重试耗尽记 error、成本熔断停在第 1 次调用、残行容忍、pairs 展开、汇总聚合、参数校验。**93 passed，ruff 0 error。**
- 真实冒烟（干跑，无 key）：fixed-window 全语料 219 对判定成功，mean_score 2.406（词法重叠启发式，仅验证管线）。

**下一步：** iter4（竞品取证深化 + docs/status_quo.md 坏例子 + docs/launch_checklist.md 发布清单）。

## iter4 — 2026-09-26 03:19–03:30（夜间会话第 4 轮）

**完成：**
- 竞品数据刷新（GitHub API，2026-09-26）：jieba 35,169★/2024-08 停更；LangChain 147,050★；RAGFlow 91,301★；"chinese chunk" 搜索头部仍是 funNLP 等泛 NLP 库——**库级占位者依旧为零**。GAP_PROOF.md 已更新并附复现命令。
- 新增 `tools/status_quo_experiment.py`（可复现取证脚本）：同一段 202 字制度文档、同一 80 字预算，对比 LC 默认参数 / LC 手工中文分隔符 / worddael。真实结果：
  - LC 默认：把「第三章」切成孤立 3 字标题块；答案「每天上限八十元」被腰斩成两半；50% 块从句中开始。
  - LC 手工中文分隔符：改善但出现「。市内交通费…」句号开头的悬垂块（keep_separator 语义），偏移量仍需自算。
  - worddael：句尾标点 100%、句中开始 0%、4 个测试答案全部完整。
- `docs/status_quo.md`：取证正文（含诚实说明：单文档不是统计结论、LC 可配置好、我们只论证默认路径的坑与配置的隐性成本）。
- `docs/launch_checklist.md`：发布渠道顺序（知乎→掘金→V2EX→Reddit→HN→awesome 清单）、chonkie/LangChain 上游贡献动作、48 小时响应纪律。
- 测试 93 passed（本轮无新代码路径，纯文档+脚本），ruff 干净。

**下一步：** iter5（`compare` 子命令、CHANGELOG 更新、README 链接 results.md、tag v0.1.0-rc1）。

## 打磨轮 — 2026-09-26 03:24–03:27（条目全部完成，按预案进入质量打磨）

**完成：**
- 新增 `tests/test_polish.py`（7 个边界测试）：无标题纯文本走 markdown 策略、max_chars=1 退化预算不挂死且精确平铺、纯换行输入、jieba 计数器真实路径、工厂函数全策略偏移不变量、chunk_file 显式策略覆盖、seed 100–115 扩展模糊扫描。
- 打磨测试当场抓到两个真实 API 缺口并修复：
  1. `chars_for_tokens` 只存在于 CharCounter，TokenChunker 的 overlap 路径换 jieba/tiktoken 计数器即崩 → 提升为 counters.py 的共享函数，任意计数器可用；
  2. `chunk()` 工厂对 token 策略不认 `max_chars` → 现在全策略统一预算参数名（token 策略下按 token 解释并在 docstring 声明）。
- 另修正一个错误断言：overlap 块的预算上限 = 内容预算 + 回看量（与 test_offsets 既有约定一致），不是内容预算本身。
- **103 passed，ruff 0 error。**

## wrap-up — 2026-09-26 03:27

- 全量终验：103 passed / ruff 0 error / git 历史完整（每迭代两个提交：代码 + 文档）。
- REPORT.md 已写给用户（完成清单、测试状态、诚实未完成项、下一步行动）。
- 夜间会话到此结束，未开始任何新功能。

## iter6 — 2026-09-26 23:16–23:35（第二阶段，用户指示继续迭代）

**完成：**
- 新增 `src/worddael/parent_child.py` `ParentChildChunker`（策略名 parent-child，已注册进 STRATEGIES/get_chunker/chunk 工厂）：
  - 子块用 RecursiveChunker（默认 150 字、overlap 30），父块 600 字；子块绝对偏移、平铺原文、不变量与所有策略一致
  - `chunk_families() -> (parents, children)`；子块 meta.parent_seq / 父块 meta.child_seqs 双向链接
  - 校验：child_max_chars 必须 < parent_max_chars（退化家庭直接报错）
- `recall_at_k_parent_level`：检索在子块上、判分在父块上（模拟真实小-大 RAG）
- full_report 新增「小-大检索对照」小节，真实结果（25 篇/75 问，子块 150/父块 600）：
  - **k=1：parent-child 0.987 vs flat 0.960（+2.7pp）；k=3：双方 0.987 持平**
  - 解读：方向符合理论（父块判分目标更宽容），但幅度小且 k=3 吸收了差异——如实呈现，不做夸大宣传
- 测试 +7：子块平铺与不变量、父块平铺、双向链接一致性、chunk() 一致性、注册表、退化校验、父块级 recall 两条性质。**110 passed，ruff 0 error。**
- README/CHANGELOG 更新（rc2）。

**下一步：** iter7（VectorRetriever + 分块×检索交互 + 计数器校准 + windows CI）。

## iter7 — 2026-09-26 23:22–23:45（第二阶段）

**完成：**
- `VectorRetriever`：纯 Python 余弦检索器，与 BM25 同 search 接口；`recall_at_k` 增加 `retriever_factory` 参数（默认 BM25 完全向后兼容）。
- full_report 新增两节：
  - 「分块 × 检索方式交互」：各策略在 BM25 vs toy 向量检索下的 recall@1（0.92 vs 0.48——玩具向量差距大，**已明确标注仅验证基础设施**，真实 embedding 数字待模型接入）
  - 「计数器校准」：CharCounter 相对 jieba 词计数平均绝对相对误差 78.0%（最大 90.8%，25 篇）。解读如实：jieba 词数 ≠ BPE token，此对照只量化偏差量级；tiktoken 参照有条件渲染（本机安装失败——Python 3.13 依赖冲突，优雅降级路径生效）
- CI 矩阵加 windows-latest（本库的开发机就是 Windows，RGB 双平台保障）。
- format_markdown 支持自定义列（修掉交互表复用时的 KeyError）。
- 测试 +9：检索器排序性质（trigram 共享构造）、零相似度过滤（确定性正交向量，绕开哈希碰撞）、vector factory 召回、交互报告形状、校准（同计数器零误差/jieba 参照/空输入）、默认 BM25 兼容。**119 passed，ruff 0 error。**

**教训记录：** 哈希碰撞会让"不同文本零相似度"的断言在小维度上失效（0.079 的假相似），这类测试要用确定性向量而不是碰运气。

**下一步：** iter8（生成式 QA 扩充脚手架 qa_gen）。

## iter8 — 2026-09-26 23:30–23:50（第二阶段）

**完成：**
- 新增 `src/worddael/eval/qa_gen.py` 生成式 QA 扩充脚手架：
  - `QAGenerator`：LLM 从文档生成 (question, gold)；质量契约硬校验——gold 必须逐字存在于原文，无效生成丢弃并计数（不静默保留）；问题按文档去重
  - `QAGenRunner`：JSONL 断点续跑（按 doc_id）、指数退避重试、错误记录成行
  - 干跑模式：确定性启发式抽句成对（含 method=dry-run 标记），零 key 跑通全管线
  - CLI `python -m worddael.eval.qa_gen --docs-dir DIR --out qa.jsonl`
- 测试 +6：干跑确定性与 gold 契约、live 模式校验（坏 gold 丢弃/去重）、坏响应抛错、续跑聚合、重试耗尽记录、目录扫描。**125 passed，ruff 0 error。**
- 真实冒烟（干跑）：内置语料前 5 篇 → 10 对 kept，0 dropped，gold 全部逐字有效。

**第二阶段下一步（醒来后/有 key 后）：** ①提供 LLM key → 跑真实 qa_gen 把基准扩到百篇级 + 首次真实 LLM 判分；②接真实 embedding 模型重跑语义切分与向量检索列；③创建 GitHub 仓库 push。

## R2 — 2026-09-26 23:50–23:58（红队驱动，可解释切分）

**角色质疑（开发者）**：切分器是黑盒——RAG 质量出问题时无法回答"这块为什么在这里被切断"。查证：全网无"按切分原因标注边界"的现成工具（搜索确认），chonkie 1.7.0 README 亦无此功能。空位成立。

**改动：**
- 新增 `worddael/explain.py`：后验边界成因分类器——从边界处真实字符推断规则（paragraph/newline/sentence/clause/budget/text_end）。**后验设计是刻意的**：解释永远和实际切法一致，不存在"账本与实现漂移"的 bug 类别。
- 全部六个切分器贯穿 `explain=True`（默认 False 零开销，meta 不加键）；MarkdownChunker 覆盖 section 边界为 `heading`；SemanticChunker 精确追踪 `semantic_drop` vs `budget`；ParentChildChunker 透传规则到子块 meta。
- 11 个新测试：规则与源字符一致性（不只查键存在）、start_rule 链式等于前一块 end_rule、fuzz seed 40–51 扫描、markdown heading、semantic drop、默认无键。

**复盘：** 分类器第一版有优先级 bug——先查句号后查换行，导致"句号+空行"的段落切断被误判为 sentence。测试的精确断言（end@10 必须是 paragraph）当场抓住。教训：边界分类这类"优先级敏感"的逻辑必须用人工构造的精确位置测试，不能只靠 fuzz。
**136 passed / ruff 0。**

## R3 — 2026-09-27 00:00–00:12（标点卫生 + 句界对齐 overlap + 评测消歧）

**角色质疑（开发者）**：①块首悬垂逗号/句号（"，每天上限…"）、块尾未闭合引号——RAG 展示与语义都是伤；②overlap 从半句中间开始；③find_gold_chunk 只认首次出现，gold 重复出现时评测误判。

**改动：**
- `_hygiene_pass`：内部平铺边界上的标点卫生保证——块尾不开口标点、块首不闭口标点。**单向左移实现**：把违例标点并给左块（左块以标点结尾合法），块只缩不涨、预算天然不破、平铺保持。默认启用。
- `_snap_overlap_start`：overlap 窗口起点对齐最近句界（窗口内无句界则跳过悬垂闭口标点）。
- `find_gold_chunks`（复数）：gold 任意出现所在块命中即算；find_gold_chunk 保留向后兼容。
- 40 个新测试（12 seed × 3 策略 fuzz + 精确构造用例）。

**复盘（红队抓自己）：** 第一版实现后 fuzz 立刻抓到 10 处泄漏——根因是 `_apply_overlap` 里 clamp（e-max_chars / prev_s+1）在 snap **之后**把起点压回标点。修复：clamp 后补一次闭合标点跳过。教训：**一条保证链上的每个后处理步骤都要重新验证前序保证**，"我们做过 snap"不等于"保证成立"。预算类测试无一放宽（左移设计的红利）。
**176 passed / ruff 0；基准数字随卫生微调已重生成（fixed-window 0.92 领先 sentence 0.907@k=1——诚实呈现）。**

## R4 — 2026-09-27 00:13–00:20（生态适配 + 一键演示）

**角色质疑（PM）**：用户活在 LangChain 生态里，没人为了切分器换框架；装完到"看见价值"的距离太长。

**改动：**
- `worddael/adapters.py` `LangChainChunker`：继承 langchain `TextSplitter`（可选导入，未安装降级为普通类），split_text/create_documents 即插即用——worddael 变成 RecursiveCharacterTextSplitter 的 drop-in 中文升级。真实 langchain-text-splitters 测试（非 mock）。
- CLI `worddael demo`：一条命令看懂项目——内置样例 + 偏移量/句尾标记/end_rule（可解释切分直接上屏）+ 策略对比表。终端即门面。
- 测试 +6（内容无损、TextSplitter 子类与 create_documents、markdown 策略、demo 出口契约）。

**复盘：** markdown 适配测试先失败后明白——overlap 模式下 join > 原文是**设计语义**（上下文重复），测试要按语义断言而不是想当然"无损"。这个案例值得写进 FAQ：overlap=0 才有拼接无损保证。
**181 passed / ruff 0。**（更正：初稿误写 186）

## R5 — 2026-09-27 00:16–00:26（真实语料 dogfooding + 性能）

**角色质疑（老板）**：性能从未量化就是裸奔；自建语料说服力弱。查证：Gutenberg pg24264 为《紅樓夢》全本公版（2.6MB 原始文件）。

**改动：**
- `tools/dogfood.py`：自动下载/剥离头尾 → 全管线跑 90.6 万字繁体真实小说 → `docs/dogfood.md` + `dogfood_stats.json`。语料不入库（.gitignore，脚本可复现）。
- **真实数字：recursive 吞吐 24.6 MB/s；2350 块 / 0 超预算；边界成因分布 newline 90%（该语料"一段一行"排版所致——explain 字段价值实证：解释跟语料走）；标点卫生违例 4 → 0。**
- 卫生修复：检查跳过空白看真实字符（"《+换行"被换行掩蔽的漏网场景）；closer 吸收设 +2 上限（该方向必然让左块微涨，测试如实放宽并注释）。
- perf floor 测试（0.1 MB/s，防退化非跑分）+ markdown 表格"整行原子"行为锁定测试。
- 顺手修正 R4 worklog 测试数笔误（186→181）。

**复盘：** 红队"性能未量化"一条逼出了两个收获——真实吞吐数字（24.6 MB/s 站得住）和卫生保证在真实语料上的 4 例漏网（自建语料永远测不出来）。**真实语料的最大价值是抓自建语料的盲区。**
**183 passed / ruff 0。**

## R6 — 2026-09-27 00:22–00:30（红队第二轮：安全 + 运维）

**角色质疑（安全工程师）**：S1 标题正则 ReDoS？→ 三条病态输入实测 <10ms，无回溯爆炸，回归测试锁定。S2 语料内容进 prompt 的注入风险 → 行业级通病，库内做了能做的（gold 硬校验），其余如实写进 eval_guide 安全注意事项——不装看不见。S3 供应链：核心零依赖，可选 extra 隔离。S4 无发现。

**角色质疑（运维）**：O1 jieba 首加载污染 stderr → setLogLevel 修复。O2 大文件全量物化 → 文档化 + backlog，不伪装支持。O3 Windows \r\n 差异 → 卫生/分类器空白处理已覆盖 \r，windows CI 矩阵保障。O4 跨运行确定性 → 系统性排查（crc32/保序 dict/set 无序遍历）无发现。

**文档：** docs/architecture.md（模块关系/五条碰不得的不变量/扩展点/non-goals）；README 加 Non-goals 小节（明确不碰 Neural/Slumber 红海——老板角色的战略决议落地）。

**复盘：** 第二轮换角色后攻击面完全不同——第一轮抓正确性，第二轮抓"声明与事实的差距"（做了没有、说了是否算数）。红队要有固定轮换的角色清单，单一视角会漏。
**184 passed / ruff 0。**（更正：初稿误写 190）

## R7 — 2026-09-27 00:25–00:28（文档真实性审计）

**角色质疑（技术写作）**：文档里的每个示例和数字都是隐性承诺——`--overlap 50` 依赖 argparse 前缀匹配（脆弱）、"187 项测试"随开发必然过期、示例代码没人验证会静默腐烂。

**改动：**
- README 修 `--overlap` → `--overlap-chars`（不依赖缩写匹配）；测试数改为"以 CI 为准"（去掉会过期的硬编码）。
- 新增 `tests/test_docs_examples.py`（5 测试）：README 三个代码片段 1:1 镜像验证（recursive 断言、semantic+embedder、CLI 一行命令）；策略表与 STRATEGIES 注册表一致性；README 引用的文档文件全部存在。
- **从此文档示例有测试守着——改 API 不同步改文档会挂 CI。**

**复盘：** "文档腐烂"是纯写作角色抓不出来的，必须让写的作用域进入 CI。本收益来自角色扮演触发的审计视角，而非新技术。
**189 passed / ruff 0。**

## R8 — 2026-09-27 00:27–00:30（性能扩展性 + 热路径）

**角色质疑（性能工程师）**：24.6 MB/s 是单点数字——扩展性曲线呢？100KB 快不代表 10MB 快（_merge_spans/卫生遍历有隐藏平方项吗）。

**实测：** 100KB→400KB→1.6MB 耗时 3.88x/3.60x（4x 数据），曲线亚线性偏快（大文本摊薄固定开销）——**无平方项**。加入 `test_scaling_stays_near_linear` 回归测试（4x 数据 ≤5x 时间，CI 宽松阈值防误报）。

**改动：** 递归切分器的四个分隔符正则从字符串改为模块级预编译（re 内部缓存本会兜底，但预编译消除了首次查找与缓存失位风险——零风险卫生工程）。

**复盘：** 性能质疑的标准动作是"先测曲线再谈优化"——实测线性后本轮**拒绝了一处不必要的优化冲动**（不做更激进的缓存/数据结构改造），避免为不存在的瓶颈引入复杂度。
**190 passed / ruff 0。**

## R9 — 2026-09-27 00:30–00:35（竞品差异证据化）

**角色质疑（老板）**：对比表里"chonkie 无父子块"是转述 README——证据等级不够，万一人家有隐藏实现，旗舰差异点当场塌方。

**查证：** 直接 `pip install chonkie`（1.7.0）检查公开 API 面：chunker 子模块导出 Cloud/Code/Late/Neural/Recursive/Semantic/Sentence/Slumber/Token 九类，**无 parent/family/hierarchy 任何命名**；Chunk 数据字段（id/text/start_index/end_index/token_count/context/embedding/metadata）无父链接。证据已写入 status_quo.md 并附一行复现命令。

**复盘：** 差异化声明必须降到"安装后一条命令可验证"的证据等级。README 转述是二手的，安装实测是一手的。
**（无代码改动，纯证据升级）**

## R11 — 2026-09-27 00:40–00:45（全角标点/特殊空白加固）

**角色质疑（语言数据工程师）**：dogfood 用的紅樓夢是繁体重排本——正文用全角句点 '．' 做句末，当前 SENT_END 不含它，句界会漏切；空白处理是 ASCII 字面量（" \t\r\n"），全角空格 \u3000 与 NBSP 会掩蔽卫生检查和解释分类。

**改动：**
- SENT_END 增加 '．﹖﹗‥⋯'（全角句点 + 变体），docstring 注明来源（繁体重排本实测）。
- 卫生助手与 explain 分类器的空白判断全部改为 `str.isspace()`（覆盖全角空格/NBSP 等 Unicode 空白）。
- 分类器顺带修一个真 bug：newlines==0 时误用带空白的 text[j] 而非回溯后的 text[k] 判句读——全角空格跟随句号时解释会错成 budget。
- 测试 +6：全角句点切分、\u3000 掩蔽下卫生仍生效、全角空格串边界解释正确。
**205 passed / ruff 0；dogfood 重跑正常。**

**复盘：** 真实语料的价值在本轮再次兑现——'．' 这个发现只能来自非合成文本。自建语料是舒适区，真实语料是照妖镜。

## R12 — 2026-09-27 00:36–00:45（研究轮：评测协议对齐业界 + precision/union）

**查证：** 检索业界分块评测实践——LegalBench-RAG 的 gold-span 协议（span 级 precision+recall）、三层评测架构（确定性检索指标 → LLM judge → 人工抽检，与本库设计同构）、故障分解方法（chunking/retrieval/generation 归因）。

**改动：**
- `recall_at_k` 增加 **precision@k**（top-k 中相关块占比，微平均）与 **union_recall**（严格判定未命中、但 top-k 块"合起来"覆盖 gold——即 gold 被边界切成两半且两半都被检回——单独计数，绝不混入严格数字）。
- 协议修复：`find_gold_chunks` 只记**完整包含** gold 的块（跨块 gold 不算严格命中，交给 union）；原实现对跨块 gold 误记覆盖起点的块，严格召回虚高。
- 基准报告表格加 precision@k / union_recall 两列并重生成。

**复盘：** 这轮红队抓到的是**评测协议自身的正确性**——比上一轮的"实现 bug"更隐蔽：协议语义（严格 vs 宽松的边界）不定义清楚，数字就不可信。定位：n=75 的小语料上 precision≈recall（每问最多 1 个相关块），precision 的区分力要等语料和 k 变大——如实记录。
**204 passed / ruff 0。**

## 插曲（00:42）— 误提交事故与修复

用户在自己机器上活动，往仓库目录放了 baidu1.html/bing1.html/parse_baidu.py，被本轮 `git add -A` 误扫进 R12 提交。处置：`git reset --soft` + restore --staged 后重新提交（用户文件保留磁盘、未跟踪）；`.git/info/exclude`（仓库本地排除，不动用户 .gitignore）防止再犯。**教训：共享机器上的"git add -A"必须换成显式路径或 info/exclude。**

## R13 — 2026-09-27 00:42–00:46（方法论对齐 + 繁体专项）

**查证：** 三层评测架构（确定性指标→LLM judge→人工抽检）与 LegalBench-RAG gold-span 协议是业界共识——与本库现有设计同构，写进 eval_guide 第 7 节作为方法论定位（含 backlog：oracle 上界归因分析、judge 校准待 key）。

**改动：**
- `tests/test_traditional.py` +4：繁体全角句点'．'切分、「」引号闭合归属、繁体文本偏移不变量、（语料存在时）紅樓夢全本管线守卫——**dogfood.md 里记录的"繁体未专项断言"局限正式关闭**。
- eval_guide 增补方法论定位与数字纪律。
**208 passed / ruff 0。**

## R14 — 2026-09-27 00:47–00:52（FAQ + 打包验证）

**角色质疑（技术支持/发布工程）**：R4 发现的"overlap 拼接比原文长"将来必然被用户当 bug 报——不写 FAQ 就得重复回答；PyPI 发布前没验证过 wheel 内容。

**改动：**
- README FAQ 四条（overlap 语义 / 块尾换行归属 / 预算 +2 卫生吸收 / 测试数以 CI 为准）——全部是本轮迭代中真实出现过的疑问，答案与实现一一对应。
- `pip wheel .` 构建验证 + wheel 内容断言（含 py.typed、eval 子包、parent_child）；dist/ 进 info/exclude 不入库。
**208 passed / ruff 0。**

## R17 — 2026-09-27 01:00（cProfile 热点审计）

**角色质疑（性能工程师）**：预编译正则后热点在哪？有没有"笨热点"（循环内编译、意外平方）？

**实测：** 1.6MB / 0.094s（profiler 下 ~17 MB/s）。热点依次：_split_keep 正则扫描（0.037s 累计）、_split 递归（238 次调用）、overlap/merge——**全部符合预期，无循环内编译、无平方路径**。31 万次函数调用对 160 万字符属健康。结论：拒绝优化（没有值得优化的目标），审计数字记档。

**复盘：** 性能审计的正确产出常常是"不需要优化"的书面证据——它把后续贡献者的优化冲动引向真正的瓶颈。

## R18 — 2026-09-27 01:31（稳定性轮）

三连跑全绿（209×3，耗时 2.5s 级稳定无 flake）；报告确定性复检 ✓；ruff 0；工作区干净。无发现即无改动——守成轮的意义是证明上一轮的保证仍然成立。

## R19 — 2026-09-27 02:22–02:26（eval_guide worked example）

**角色质疑（技术写作 + 数据科学家）**：评测指南讲了"怎么跑"没讲"怎么读"——用户拿到表格第一个问题是"precision@3 为什么只有 0.33"。

**改动：** eval_guide 增第 8 节 worked example：用当前真实数字演示四步读法（recall 双 k 连读→precision 天花板成因→union 差值=切碎答案的直接度量→失败清单归因起点）。顺带重生成 results.md（上轮 time_ms 移除后的首次再生成，表格已无计时列，0 处 time_ms）。

**复盘：** "教读数"比"给数字"值钱——union 与 strict 的差值被我们定义为"切碎答案的度量"，这个读法本身就是可传播的方法论贡献。
**208 passed / ruff 0。**

## R20 — 2026-09-27 03:08–03:12（全文审读 + TODO 扫描）

**角色质疑（技术写作/审计员）**：文档里的数字会腐烂——status_quo.md 还写着卫生修复前的 0.947。

**改动：**
- 全库 TODO/FIXME/XXX/HACK 扫描：**零残留**。
- status_quo.md 旧数字 0.947 → 0.96（与当前 results.md 一致）；实验 A/B/C 数字重跑验证仍然成立（实验脚本确定性）。
- **新守卫测试**：benchmarks/results.md 必须能从当前代码逐字节重生成（利用确定性保证）——从此"改了切分行为忘更新基准"会直接挂 CI。
**210 passed / ruff 0。**

**复盘：** 确定性保证（R15）在本轮变成了审计工具的基础——没有逐字节确定性，就不可能写"results.md 必须可重生成"的测试。保证之间开始互相增值。

## R21 — 2026-09-27 04:05–04:08（CHANGELOG rc3 + 版本号）

CHANGELOG 收敛为 rc3 单节（晚间 rc2 内容并入），版本号 pyproject/__init__ 同步 0.1.0rc3，CLI --version 实测输出正确。210 passed / ruff 0。守成轮，无新功能。

## R22 — 2026-09-27 05:01–05:04（稳定性复检 + rc3 wheel 验证）

210×2 全绿；rc3 wheel 构建成功且 METADATA 版本正确（0.1.0rc3）；基准时效性守卫实测通过（results.md 与当前代码逐字节同步）。工作区干净，无新发现——rc3 处于可发布状态（PyPI 上传等用户账号动作除外）。

## R24 — 2026-09-27 23:50–23:57（发布准备轮，用户晚间回来后指示）

**角色质疑（发布工程师）**："准备好"的标准不是代码写完，而是别人接手时每个环节都有入口。

**新增：**
- PUBLISH.md：建仓/push/tag/占位 URL 替换（附现成 sed）/CI 验证/PyPI trusted publisher 全流程手册
- .github/workflows/publish.yml：release 触发的 PyPI trusted publishing（无 token 进 secrets）
- SECURITY.md：报告渠道 + 本库供应链面说明（零依赖、无隐式外呼、语料视为不可信）
- ISSUE 模板 ×2（bug 模板内嵌偏移不变量复现提示——把核心承诺写进用户工作流）

**过程事故如实记录：** 后台任务唤醒的新 shell 会重置回主工作目录 E:\n_projects，导致 R24 首条命令把模板建到了上级目录——已移回仓库并在 WORKLOG 留痕。教训：长会话中每条命令显式 cd，不能依赖继承。

**210 passed / ruff 0。发布等待项只剩：用户提供 GitHub 仓库名。**
