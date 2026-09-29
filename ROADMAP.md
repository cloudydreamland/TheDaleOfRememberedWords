# ROADMAP — 夜间自动化迭代驱动表

> 规则：自动迭代每一轮从上到下找第一个未勾选条目，完整做完（代码+测试+文档）再勾选。
> 每条目设计为一轮（≤45 分钟）可完成。做完更新 WORKLOG.md 并提交。

## 迭代条目

- [x] **iter0（已完成）** 仓库骨架：五个核心策略、字符偏移不变量 fuzz 测试、内置迷你语料 + BM25 recall@k 评测、LLM 判分器（干跑）、CLI、CI、70 项测试全绿。
- [x] **iter1 — embedding 接入层**：新增 `worddael/embedders.py`：`OpenAICompatibleEmbedder`（urllib 调 /embeddings，读环境变量 key）、`HashingEmbedder`（从 eval.report 提升为正式类）。全部用 mock 测试（不下载任何模型）。SemanticChunker 文档更新。
- [x] **iter2 — 有区分度的基准**：把 sample_data 扩到 ≥25 篇、≥75 问（保持原创自写）；`report.py` 支持 `--k 1` 与按文档分解的失败清单输出；真实跑一轮并把结果写入 `benchmarks/results.md`（如实记录，包括平局）。
- [x] **iter3 — LLM 判分实跑引擎**：`llm_judge.py` 增加批量运行器：限速、重试、JSONL 断点续跑、成本上限熔断；用 unittest.mock 测 HTTP 路径（不发真实请求）；写 `docs/eval_guide.md` 说明拿到 API key 后如何跑大规模评测。
- [x] **iter4 — 竞品取证深化**：用 GitHub/PyPI API 重新抓取数据更新 GAP_PROOF.md（带日期与 star 数）；写 `docs/status_quo.md`：用真实代码片段展示 LangChain splitter 切中文的典型坏例子（切在何处、为什么错）；写 `docs/launch_checklist.md`（知乎/掘金/HN 发布清单）。
- [x] **iter5 — 打磨与 RC**：`compare` 子命令（对单文件并排比较各策略）；CHANGELOG.md；`docs/` 目录整理；README 补充 benchmark/results 链接；全量 pytest + ruff 清零；git tag `v0.1.0-rc1`。

## 收尾条目（08:20 定时任务）

- [x] **wrap-up（提前完成于 03:28，全部条目已完结）**：全量测试与 lint 最终确认；写 `REPORT.md`（夜间总结：完成清单、测试状态、诚实的未完成项、给用户的下一步建议）；最终提交。

## 之后（用户醒来后的人工事项）

- 注册 GitHub 仓库并 push（需要用户的账号与仓库选择）
- 中文/英文同步发布 + 提交到 awesome 列表
- 提供 LLM API key 跑首轮大规模评测报告
- 论文联动：把基准数据集发布到 HuggingFace

## 第二阶段（2026-09-26 晚启动，用户指示"继续迭代"）

- [x] **iter6 — ParentChildChunker（父子块 / 小-大检索）**：子块（小预算）负责精确匹配、父块（大预算）负责提供上下文，检索命中子块、返回父块——竞品扫描发现中文 RAG 应用模板人人手搓此模式，提升为库级标准件。要求：子块平铺与偏移不变量照旧、meta 携带 parent 链接、`chunk_families()` 返回 (parents, children)；评测侧新增父块级 recall（检索子块按父块判分）；在 full_report 增加"小-大检索对照"小节（结果如实，包括无增益）。
- [x] **iter7 — 检索侧升级 + 计数器校准**：retrieval.py 增加 `VectorRetriever`（纯 Python 余弦，可插拔 embedder，与 BM25 同接口）；recall_at_k 支持 retriever 参数；full_report 增加"分块×检索方式交互"小节（hashing toy 向量的数字必须标注 toy）；新增 `calibrate_counter` 工具：量化 CharCounter 启发式相对 jieba 的平均误差（诚实数字：预算切分够用/计费不适用）；CI 矩阵加 windows-latest。
- [x] **iter8 — 生成式 QA 扩充脚手架**：`worddael.eval.qa_gen`：LLM 从任意文档批量生成 (question, gold) 对的流水线（OpenAI 兼容端点、干跑模式用启发式问题模板、JSONL 续跑复用 JudgeRunner 机制），为"百篇级基准"铺路；测试全 mock；写使用说明。真实大规模生成仍待用户提供 key。

## 第三阶段（2026-09-26 深夜，红队驱动迭代：每轮 = 角色质疑 → 查证 → 改进 → 复盘）

红队评审 #1 见 docs/reviews/red_team_1.md。新颖性已验证：可解释切分与标点卫生保证全网无现成工具，chonkie 1.7.0 亦无父子块与解释功能。

- [x] **R2 — 可解释切分（旗舰创新）**：每个 chunk 的 meta 增加边界成因记录（end_rule/start_rule ∈ paragraph|newline|sentence|clause|budget|semantic_drop|heading），RecursiveChunker/TokenChunker/SentenceChunker 全部标注；`chunk(explain=True)` 默认关闭时零开销；测试断言规则字段与实际边界一致；报告统计各规则占比。
- [x] **R3 — 标点卫生 + overlap 句界对齐 + 评测消歧**：保证一：块尾（rstrip 后）不得为开口标点（「（“《等），块首不得为闭口标点（，。」）等），违例时回退到最近合法边界；保证二：overlap 起点对齐句界（退化到逗号界）；find_gold_chunk 多出现消歧（任一包含 gold 任一出现的块命中即算）。全部 fuzz 测试覆盖。
- [x] **R4 — 生态适配 + 一键演示**：worddael/adapters.py 提供 LangChain TextSplitter 兼容包装（langchain-text-splitters 已可真测）；CLI `worddael demo` 一键演示（内置样例、偏移量与卫生统计、策略对比一步出）。
- [x] **R5 — 真实语料 dogfooding + 性能基准**：Gutenberg 中文公版书（网络抓取，版权安全）跑全管线 → docs/dogfood.md（真实块大小分布、边界规则占比、卫生违例数、吞吐 MB/s）；perf floor 回归测试；markdown 表格原子性行为补测试。
- [x] **R6 — 架构文档 + non-goals + 红队第二轮**：docs/architecture.md（模块关系图、不变量清单、扩展点）；README 加 Non-goals（不做 Neural/Slumber 红海）；红队第二轮（安全工程师/运维视角）评审 docs/reviews/red_team_2.md，可行修复当场做。
- [ ] **R7 — 终版复盘**：全量测试 + 复盘写入 WORKLOG；REPORT.md 终版（07:30 前）。
