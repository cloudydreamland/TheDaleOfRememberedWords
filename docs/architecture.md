# 架构（Architecture）

> 面向贡献者。读完这页你应该知道：模块怎么连、哪些不变量碰不得、往哪里加东西。

## 模块关系

```
worddael/
├── types.py        Chunk（frozen dataclass）：text/start/end/seq/strategy/meta
├── sentences.py    底层结构分析：中文句界切分、markdown 分节（heading path + 代码围栏原子）
├── counters.py     token 计数：CharCounter 启发式（默认）+ jieba/tiktoken 可选 + 校准工具
├── chunkers.py     五个切分策略 + 边界移动工具（_hygiene_pass / _apply_overlap）
│                     Token / Sentence / Recursive / Markdown / Semantic
├── parent_child.py ParentChildChunker：父子块（小-大检索），组合两个 RecursiveChunker
├── embedders.py    HashingEmbedder（零依赖）+ OpenAICompatibleEmbedder（任意 /embeddings）
├── explain.py      可解释切分：后验边界成因分类器（post-hoc，保证解释≠实现漂移）
├── adapters.py     LangChain TextSplitter 兼容包装（可选依赖）
├── io_utils.py     chunk_file / to_jsonl
├── cli.py          worddael 命令（demo / chunk / --compare / --stats / --jsonl）
└── eval/           BM25 + VectorRetriever、recall@k（含父块级）、LLMJudge、
                      JudgeRunner（限速/重试/续跑/熔断）、qa_gen（生成式 QA 扩充）、
                      sample_data（25 篇原创基准）、report（双口径报告）
```

## 碰不得的不变量（每个都有 fuzz 测试守着）

1. **偏移精确**：`chunk.text == source[chunk.start:chunk.end]`，永远。
2. **平铺**：非 overlap 块的 spans 首尾相接覆盖全文（`tests/test_offsets.py` 光标断言）。
3. **标点卫生**：内部边界修复后，块尾（rstrip 后）不开口标点、块首（lstrip 后）不闭口标点（`tests/test_hygiene.py`）。例外：全文档首个字符本身是闭口标点、或末尾吸收超 +2 字符上限时保留原状（测试与文档如实标注）。
4. **预算**：非 overlap 块 ≤ max_chars + 2（+2 为卫生吸收上限）。
5. **可解释一致**：`end_rule` 必须能从源文本边界字符重新推导（后验分类器天然满足；策略覆盖值——heading/semantic_drop——由切分器自己保证）。

## 关键设计决策

- **后验解释**（explain.py）：不切分时零开销；解释从实际边界字符推导，杜绝"记录与实现不一致"。
- **卫生左移**：边界修复只向左移动（closer 吸收进左块、opener 推给右块开头），块只缩不涨，预算断言不因卫生而复杂化。
- **分隔符归属**：切分时分隔符（句号/换行）留在前一段尾部——这是偏移精确与卫生检查能工作的前提。
- **zero-dep 核心**：jieba/tiktoken/langchain/embedding 全部可选导入 + 优雅降级，测试用 `importorskip`。

## 扩展点

- **新策略**：继承 `BaseChunker`，实现 `chunk()`（用 `_build` 构造，自动获得 explain 支持）；在 `STRATEGIES` 注册；fuzz 测试加进 `test_offsets.py` 的策略参数表。
- **新检索器**：实现 `search(query, k) -> list[(index, score)]`；`recall_at_k(..., retriever_factory=...)` 直接可用。
- **新评测语料**：按 `sample_data.SAMPLE_DOCS` 的文档形状提供；`test_corpus_integrity` 会强制校验 gold 逐字存在。

## Non-goals（明确不做）

- **Neural/LLM 驱动切分**（对齐 chonkie Neural/Slumber）：红海、依赖 GPU/API key，与"零依赖、纯 CPU"定位冲突。语义切分通过可插拔 embedder 支持，模型选择权在用户。
- **通用 RAG 框架**：不做检索库、向量库、生成管线——只做切分与切分评测，通过 adapters 进入既有框架。
- **计费级 token 计数**：启发式计数器只服务预算切分；计费请用 tiktoken（见校准报告的误差数字）。
- **超内存流式切分**：当前 API 假设单文档可完整载入内存（GB 级文件请自行分片）；流式 API 列为 backlog，不是现在的目标。
