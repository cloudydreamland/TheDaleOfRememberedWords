# Security Policy

## Supported versions

Only the latest tag on `main` receives fixes. Pre-releases (rc/alpha) are
best-effort.

## Reporting a vulnerability

Please do **not** open a public issue for security reports. Use GitHub's
"Report a vulnerability" (Security tab → Private vulnerability reporting),
or contact the maintainers directly (see repository owner profile).

## Scope notes for worddael specifically

- The core library makes **no network calls** and has **zero required
  dependencies** — the supply-chain surface is deliberately minimal.
- Network-capable modules (`embedders.OpenAICompatibleEmbedder`,
  `eval.llm_judge`, `eval.qa_gen`) only call endpoints you explicitly
  configure; API keys are read from parameters or environment variables and
  are never written to state files.
- Documents passed to the evaluation/LLM tooling should be treated as
  untrusted input (prompt-injection caveats are documented in
  `docs/eval_guide.md` §6).
