# The Dale of Remembered Words — Worddael

[简体中文](README.md) · English

**Worddael** is a lightweight Python library for chunking Chinese text for retrieval-augmented generation (RAG), while preserving exact offsets into the source.

[![CI](https://github.com/cloudydreamland/TheDaleOfRememberedWords/actions/workflows/ci.yml/badge.svg)](https://github.com/cloudydreamland/TheDaleOfRememberedWords/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![MIT license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Every returned chunk retains its original text span, so retrieved passages can be traced back to a document without reconstructing offsets.

## Quick start

> This package is not on PyPI yet. Install the current GitHub version with:


```bash
git clone https://github.com/cloudydreamland/TheDaleOfRememberedWords.git
cd TheDaleOfRememberedWords
python -m pip install .
```

```python
from worddael import chunk

source = "第一段。第二段包含一个版本号 v1.2.3。第三段。"
chunks = chunk(source, strategy="recursive", max_chars=20, overlap_chars=0)

for item in chunks:
    assert item.text == source[item.start:item.end]
```

Try the CLI or run the local retrieval evaluation:

```bash
worddael manual.md --stats
python -m worddael.eval.report --k 3
```

Optional integrations are installed only when needed:

```bash
python -m pip install ".[jieba]"    # word-aware counting
python -m pip install ".[tiktoken]" # OpenAI tokenizer counting
```

## Features

- Chinese sentence boundaries, including closing quotation marks and full-width punctuation.
- Exact source offsets for every chunk.
- Recursive, sentence, Markdown, token, semantic, and parent-child strategies.
- Markdown heading paths and code-fence handling.
- Optional embedding and token-counter integrations; no required runtime dependencies.
- Built-in retrieval evaluation and a `LangChain` adapter.

| Strategy | Best for |
| --- | --- |
| `recursive` | General Chinese prose with exact source spans. |
| `markdown` | Documents organized by headings and code fences. |
| `sentence` | Keeping sentence boundaries intact. |
| `token` | Packing text to a token budget. |
| `semantic` | Embedding guided splits when an embedder is configured. |
| `parent-child` | Retrieving small passages with larger parent context. |

## Evaluation and limitations

The included retrieval corpus is a small, versioned development benchmark, not evidence that one chunking strategy wins for every application. See [benchmark results](benchmarks/results.md) and the [evaluation guide](docs/eval_guide.md) for its scope and reproduction steps. The built-in hashing embedder is a pipeline smoke-test utility, not a semantic-quality baseline.

## Documentation

- [简体中文](README.md)
- [Architecture](docs/architecture.md)
- [Chinese splitter examples](docs/status_quo.md)
- [Evaluation guide](docs/eval_guide.md)
- [Changelog](CHANGELOG.md)
- [Roadmap](ROADMAP.md)
- [Research notes](GAP_PROOF.md)

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Please report security issues privately; see [SECURITY.md](SECURITY.md).

## Feedback and contributing

Use [Discussions](https://github.com/cloudydreamland/TheDaleOfRememberedWords/discussions) for questions and ideas, and [Issues](https://github.com/cloudydreamland/TheDaleOfRememberedWords/issues) for reproducible bugs. Share only synthetic or redacted minimal examples; never upload personal data, API keys, or private source text. Report security issues privately as described in [SECURITY.md](SECURITY.md).

## License

MIT. See [LICENSE](LICENSE).
