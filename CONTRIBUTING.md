# Contributing to worddael

Thanks for considering a contribution! Rules of the road:

1. **The offset invariant is sacred.** Every `Chunk` must satisfy
   `chunk.text == source[chunk.start:chunk.end]`, and spans must tile the
   source. Fuzz tests in `tests/test_offsets.py` enforce this — extend them
   when you add a strategy.
2. **Zero required dependencies.** Anything heavy (jieba, tiktoken,
   embedding models) goes behind optional imports / extras.
3. **No fake numbers.** Eval results in docs must come from real runs and
   state their corpus size. Smoke-test results are labeled as such.
4. Chinese punctuation semantics are the product: when in doubt, add a
   regression test with the real-world case (e.g. 「……」”., 3.14, URL).

```bash
pip install -e ".[jieba,dev]"
pytest
ruff check src tests
```

MIT-licensed. By contributing you agree your contributions are licensed
under MIT as well.
