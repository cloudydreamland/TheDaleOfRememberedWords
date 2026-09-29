# Changelog

## 0.1.0-rc2 (2026-09-26, evening)

- `worddael.eval.qa_gen`: generative QA expansion scaffold — documents to
  (question, gold) pairs with verbatim-gold validation, per-doc dedupe,
  JSONL resume and retries; deterministic dry-run mode needs no key
- `VectorRetriever` + retriever-swappable `recall_at_k`; benchmark report
  gains chunking×retrieval interaction and counter-calibration sections
- CI matrix adds windows-latest

- `ParentChildChunker` (strategy `parent-child`): small-to-big chunking —
  index children for matching, return the owning parent for context.
  Children keep the exact-offset invariant and tile the source; family
  links live in `meta["parent_seq"]` / `meta["child_seqs"]`.
  `chunk_families()` returns `(parents, children)`.
- Parent-level retrieval metric `recall_at_k_parent_level`; the benchmark
  report gains a small-to-big comparison section (k=1: 0.987 vs flat
  0.960 on the built-in corpus; k=3 tied — numbers regenerate with the
  corpus, see benchmarks/results.md)

## 0.1.0-rc1 (2026-09-26)

Night-built release candidate. Additions over the initial alpha, all
verified by 96 passing tests:

- `worddael.embedders`: `HashingEmbedder` (crc32-stable trigram hashing;
  replaces a per-process-random `hash()` in the early prototype) and
  `OpenAICompatibleEmbedder` (lazy env-var key resolution, batching,
  wrapped HTTP errors); both plug directly into `SemanticChunker`
- Benchmark corpus grown to 25 original Chinese docs / 75 questions with a
  permanent integrity test (verbatim golds); now discriminating:
  recursive recall@1 0.947 vs fixed-window 0.907 at max_chars=150
- Dual-k report (`recall@1` + `recall@3` + failure lists) in
  `benchmarks/results.md`; `ends_on_punct_frac` metric counts rstripped
  chunk tails (offset-exact chunks may carry trailing newlines)
- `JudgeRunner`: batch answerability judging with per-process rate
  limiting, exponential-backoff retries, JSONL resume (torn-line
  tolerant), and estimated-cost cap; `worddael.eval.judge_runner` CLI,
  dry-run by default
- `worddael.compare` + `worddael --compare`: side-by-side strategy metrics
- `docs/status_quo.md`: reproducible experiment showing default
  LangChain splitters isolating headings, splitting answers across
  chunks, and starting 50% of chunks mid-sentence on the same text
- `docs/eval_guide.md`, `docs/launch_checklist.md`

## 0.1.0-alpha (2026-09-26)

Initial night-built alpha.

- Strategies: `recursive`, `markdown`, `sentence`, `token`, `semantic`
- Character-offset invariant on all chunks (fuzz-tested)
- Chinese sentence boundaries: 。！？；… with quote/closer attachment;
  ASCII `.` protected (decimals, versions)
- Markdown parsing: ATX headings with heading paths, atomic code fences
- Pluggable token counters: heuristic (default), jieba, tiktoken
- Eval subpackage: built-in mini corpus (8 docs / 24 questions), pure-Python
  BM25, recall@k, LLM answerability judge (dry-run + OpenAI-compatible live
  mode), strategy comparison report with fixed-window baseline
- CLI: `worddael FILE [--strategy ... --jsonl ... --stats]`
- 70 tests passing, ruff clean, GitHub Actions CI
