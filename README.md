# The Dale of Remembered Words — Worddael

简体中文 · [English](README.en.md)

> 展示名 **The Dale of Remembered Words** 描绘一处安放文字的山谷；Worddael 是该项目的短名。

**中文优先的 RAG 文本切分库。把中文文本切成好用的块，并保留精确到字符的原文偏移量。**

[![CI](https://github.com/cloudydreamland/TheDaleOfRememberedWords/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 为什么需要它 / Why

通用文本切分器可用于多种语言，但中文句界、标点归属和回引源文等细节往往需要应用自行处理。`worddael` 提供中文规则、可组合策略和原文偏移，让这些行为能在管线中检查：

- **中文句界原生正确**：`。！？；…` 加引号闭合格式感知；ASCII `.` 不切（保护 `3.14`、版本号、URL）
- **字符级偏移不变量**：每个 chunk 保证 `chunk.text == source[chunk.start:chunk.end]`，检索结果可直接引用回原文（fuzz 测试覆盖）
- **零必装依赖**：核心纯 Python；jieba / tiktoken / embedding 全部可选
- **自带评测**：内置 BM25 检索召回评测 + 可插拔 LLM 判分器（支持干跑，零成本先跑通）
- **CPU 即可用**：不需要 GPU，不需要下载模型

## Quickstart (English)

worddael (Worddael, "a cut-out chapter") chunks **Chinese** text for RAG, keeping exact
character offsets so every retrieved chunk can be cited back to its source.

```python
from worddael import chunk

chunks = chunk(long_text, strategy="recursive", max_chars=500, overlap_chars=50)
assert all(c.text == long_text[c.start:c.end] for c in chunks)  # invariant
```

- Chinese-aware sentence boundaries (。！？；… with quote-closer attachment;
  ASCII `.` never splits decimals/versions)
- Markdown heading paths, atomic code fences, parent-child (small-to-big)
  chunking, pluggable semantic chunking and token counters
- Zero required dependencies, CPU-only; optional extras: `worddael[jieba]`,
  `worddael[tiktoken]`
- Built-in evaluation: BM25 recall/precision@k over an original 25-doc
  benchmark, LLM answerability judging (dry-run needs no API key)

See [docs/status_quo.md](docs/status_quo.md) for reproducible failure cases
of default splitters on Chinese, and [benchmarks/results.md](benchmarks/results.md)
for current numbers.

## 安装 / Install

> 当前尚未发布到 PyPI；下方给出从 GitHub 获取并本地安装的命令。

```bash
git clone https://github.com/cloudydreamland/TheDaleOfRememberedWords.git
cd TheDaleOfRememberedWords
python -m pip install .
# PyPI 首发后：python -m pip install worddael
python -m pip install ".[jieba]"
```

## 快速开始 / Quickstart

```python
from worddael import chunk

text = open("manual.md", encoding="utf-8").read()
chunks = chunk(text, strategy="recursive", max_chars=500, overlap_chars=50)

for c in chunks:
    print(c.seq, c.start, c.end, c.meta.get("headings"), c.text[:20])
    assert c.text == text[c.start:c.end]   # 永远成立，可放心引用
```

```python
from worddael import SemanticChunker, OpenAICompatibleEmbedder, HashingEmbedder

# 语义切分：接任意 OpenAI 兼容 embedding 接口（key 从环境变量读取，批量+超时可配）

embedder = OpenAICompatibleEmbedder(
    base_url="https://open.bigmodel.cn/api/paas/v4",
    model="embedding-3",
    api_key_env="ZHIPUAI_API_KEY",
)
chunks = SemanticChunker(embed_fn=embedder, similarity_threshold=0.55,
                         max_chars=500, min_chars=80).chunk(long_text)

# 或用零依赖的 HashingEmbedder 先把管线跑通（玩具级，非质量结论）

chunks = SemanticChunker(embed_fn=HashingEmbedder()).chunk(long_text)
```

命令行：

```bash
worddael doc.md --stats
worddael manual.txt --strategy recursive --max-chars 400 --overlap-chars 50 --jsonl chunks.jsonl
```

评测（CPU、零 API 成本）：

```bash
python -m worddael.eval.report --k 3            # 内置语料上对比各策略的 recall@k
```

## 切分策略 / Strategies

| strategy | 适用场景 | 说明 |
|---|---|---|
| `recursive` | 通用中文文本 | 段落→行→句号级→逗号级递归切分，中文标点优先 |
| `markdown` | .md 文档 | 按标题结构切，heading path 写入 meta，代码块保持完整 |
| `sentence` | 需要整句边界 | 按中文句界整句打包 |
| `token` | 对齐模型 token 预算 | 可插拔计数器（启发式 / jieba / tiktoken） |
| `semantic` | 主题边界敏感 | 自带 embedding 函数，相邻句相似度下降处切分，带防碎片最小块长 |
| `parent-child` | 小-大检索（RAG 常见手搓模式） | 子块精确匹配、父块给上下文；`chunk_families()` 返回 (父, 子)，meta 双向链接，偏移不变量照旧 |

## 与现有方案的关系 / Landscape

| 方案 | 问题 |
|---|---|
| LangChain text splitters | 通用切分组件；Worddael 可作为中文切分选项，并提供自己的偏移与策略 API |
| Chonkie 等 ingestion 库 | 面向更广的文档摄取与切分场景；选择时应按格式覆盖、语言和集成需求比较 |
| RAGFlow 等 RAG 引擎 | 提供完整应用或管线；Worddael 是可嵌入现有 Python 项目的独立库 |
| jieba | 中文分词工具；与按句界、结构和长度做文档切分的目标不同 |

LangChain 默认参数切中文的**真实坏例子**（答案腰斩、标题孤立，可复现脚本）见 [docs/status_quo.md](docs/status_quo.md)；详细论证见 [GAP_PROOF.md](GAP_PROOF.md)。

## 评测 / Evaluation

- 内置基准（25 篇 / 75 问）真实结果：[benchmarks/results.md](benchmarks/results.md)
  - 当前快照：recall@1 recursive 0.96 vs fixed-window 0.907；小-大检索（子块 150 检索 / 父块 600 判分）0.987 vs 平铺 0.96
- 大规模 LLM 判分评测指南（拿到 key 后）：[docs/eval_guide.md](docs/eval_guide.md)
- 选题论证（为什么这个缺口是真的）：[GAP_PROOF.md](GAP_PROOF.md)

## 路线图 / Roadmap

见 [ROADMAP.md](ROADMAP.md)。当前 v0.1.0rc3：六个策略（含父子块）+ 可解释切分 + 标点卫生保证 + 评测框架（测试数以 CI 为准）。真实语料实测见 [docs/dogfood.md](docs/dogfood.md)：90.6 万字《紅樓夢》，24.6 MB/s，卫生违例 0。

## Non-goals（明确不做）

- **模型/LLM 驱动切分**：红海方向且依赖 GPU/API，与零依赖纯 CPU 定位冲突；语义切分走可插拔 embedder
- **通用 RAG 框架**：只做切分与切分评测，通过 [adapters](#安装--install) 进入 LangChain 等既有管线
- **计费级 token 计数**：启发式计数只服务预算切分（误差数字见基准报告校准小节）
- **流式超大文件**：单文档需可入内存；GB 级请自行分片

架构与扩展指南见 [docs/architecture.md](docs/architecture.md)。

## FAQ

- **overlap 模式下拼接结果比原文长？** 设计如此：overlap 是上下文重复（上一块尾部回看）。`overlap_chars=0` 时拼接与原文逐字相等（有测试锁定）。
- **为什么有的块以换行结尾？** 分隔符归属前一块是偏移精确的前提；判断块质量请对 `rstrip()` 后的文本判断（内置统计均如此）。
- **预算为何偶尔超出 1-2 字？** 标点卫生会把下一块开头的悬垂标点吸收进前一块（上限 2 字），换来"块首无悬垂标点"的保证。
- **测试数是固定的吗？** 以 CI 最新运行为准；文档不写死数字。

## 开发 / Development

```bash
pip install -e ".[jieba,dev]"
pytest
ruff check src tests
```

## 反馈与参与

使用问题和功能建议可以在 [Discussions](https://github.com/cloudydreamland/TheDaleOfRememberedWords/discussions) 交流；可复现缺陷请提交 [Issue](https://github.com/cloudydreamland/TheDaleOfRememberedWords/issues)。请只附合成或脱敏后的最小样例，不要上传真实个人信息、API key 或业务原文。安全问题请按 [SECURITY.md](SECURITY.md) 私下报告。

## License

MIT
