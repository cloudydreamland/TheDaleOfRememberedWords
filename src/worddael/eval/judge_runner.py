"""Batch answerability judging: rate limiting, retries, resume, cost cap.

The runner turns (strategy, doc, question, chunk) pairs into judged records
appended to a JSONL state file, so an interrupted run resumes exactly where
it stopped — the property that makes large-scale token-hungry evaluations
practical.

Usage::

    from worddael.eval.judge_runner import JudgeRunner, build_pairs
    from worddael.eval.llm_judge import JudgeConfig, LLMJudge
    from worddael.eval.report import build_strategies
    from worddael.eval.sample_data import SAMPLE_DOCS

    judge = LLMJudge(JudgeConfig(api_key="..."))
    runner = JudgeRunner(judge, state_path="results/judge.jsonl", rpm=30, cost_cap=5.0)
    stats = runner.run(build_pairs(build_strategies(150, 30), SAMPLE_DOCS), dry_run=False)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .llm_judge import JudgeConfig, LLMJudge
from .report import build_strategies
from .sample_data import SAMPLE_DOCS


def pair_key(pair: dict) -> str:
    return f"{pair['strategy']}|{pair['doc_id']}|{pair['q_idx']}|{pair['chunk_seq']}"


def build_pairs(strategies: dict, docs: list[dict]) -> list[dict]:
    """Cross every strategy's chunks with every question of its document."""
    pairs: list[dict] = []
    for name, chunker in strategies.items():
        for doc in docs:
            chunks = chunker.chunk(doc["text"])
            for q_idx, q in enumerate(doc["questions"]):
                for c in chunks:
                    pairs.append(
                        {
                            "strategy": name,
                            "doc_id": doc["id"],
                            "q_idx": q_idx,
                            "question": q["q"],
                            "chunk_seq": c.seq,
                            "chunk_text": c.text,
                        }
                    )
    return pairs


@dataclass
class RunStats:
    total: int = 0
    skipped: int = 0
    judged: int = 0
    errors: int = 0
    est_cost: float = 0.0
    cost_capped: bool = False
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"total={self.total} judged={self.judged} skipped={self.skipped} "
            f"errors={self.errors} est_cost={self.est_cost:.4f}"
            + (" [STOPPED: cost cap reached]" if self.cost_capped else "")
        )


class JudgeRunner:
    """Append-only judged-record runner.

    - ``rpm`` rate limit (per-process, wall-clock).
    - ``max_retries`` exponential backoff on exceptions; final failure is
      recorded as an error record, never silently dropped.
    - Resume: pairs whose key already exists in the state file are skipped.
    - ``cost_cap``: estimated spend (live mode only) that stops the run.
    """

    def __init__(
        self,
        judge: LLMJudge,
        state_path: str | Path,
        rpm: int = 60,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        cost_cap: float | None = None,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ) -> None:
        if rpm <= 0:
            raise ValueError("rpm must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.judge = judge
        self.state_path = Path(state_path)
        self.min_interval = 60.0 / rpm
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.cost_cap = cost_cap
        self._sleep = sleep
        self._monotonic = monotonic

    def load_done(self) -> dict[str, dict]:
        if not self.state_path.exists():
            return {}
        done: dict[str, dict] = {}
        with self.state_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn final line from a killed run
                if "key" in rec:
                    done[rec["key"]] = rec
        return done

    def run(self, pairs: list[dict], dry_run: bool = True) -> RunStats:
        stats = RunStats(total=len(pairs))
        done = self.load_done()
        stats.skipped = sum(1 for p in pairs if pair_key(p) in done)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        last_call: float | None = None
        with self.state_path.open("a", encoding="utf-8") as f:
            for pair in pairs:
                key = pair_key(pair)
                if key in done:
                    continue
                est = 0.0
                if not dry_run:
                    est = self.judge.estimate_cost(1, len(pair["chunk_text"]))["est_cost"]
                    if self.cost_cap is not None and stats.est_cost + est > self.cost_cap:
                        stats.cost_capped = True
                        stats.notes.append(f"stopped before {key}")
                        break

                if last_call is not None:
                    wait = self.min_interval - (self._monotonic() - last_call)
                    if wait > 0:
                        self._sleep(wait)
                result = self._attempt(pair, dry_run)
                last_call = self._monotonic()

                record = {
                    "schema": 1,
                    "key": key,
                    "strategy": pair["strategy"],
                    "doc_id": pair["doc_id"],
                    "q_idx": pair["q_idx"],
                    "question": pair["question"],
                    "chunk_seq": pair["chunk_seq"],
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    **result,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                done[key] = record
                stats.est_cost += est
                if result.get("method") == "error":
                    stats.errors += 1
                else:
                    stats.judged += 1
        return stats

    def _attempt(self, pair: dict, dry_run: bool) -> dict:
        for attempt in range(self.max_retries + 1):
            try:
                return self.judge.score_pair(
                    pair["chunk_text"], pair["question"], dry_run=dry_run
                )
            except Exception as exc:  # noqa: BLE001 — recorded, never raised past retries
                if attempt == self.max_retries:
                    return {"score": -1, "method": "error", "error": str(exc)[:200]}
                self._sleep(self.backoff_base * (2**attempt))
        return {"score": -1, "method": "error", "error": "unreachable"}  # pragma: no cover


def summarize(state_path: str | Path) -> dict:
    """Aggregate a judged state file: per-strategy mean score and errors."""
    per: dict[str, dict] = {}
    with Path(state_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            bucket = per.setdefault(rec["strategy"], {"n": 0, "score_sum": 0.0, "errors": 0})
            if rec.get("method") == "error":
                bucket["errors"] += 1
                continue
            bucket["n"] += 1
            bucket["score_sum"] += rec.get("score", -1)
    return {
        s: {
            "n": b["n"],
            "errors": b["errors"],
            "mean_score": round(b["score_sum"] / b["n"], 3) if b["n"] else None,
        }
        for s, b in sorted(per.items())
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="worddael-eval-judge")
    parser.add_argument("--state", default="results/judge.jsonl", help="JSONL state/output file")
    parser.add_argument("--max-chars", type=int, default=150)
    parser.add_argument("--overlap-chars", type=int, default=30)
    parser.add_argument("--strategy", default=None, help="judge a single strategy instead of all")
    parser.add_argument("--live", action="store_true", help="real API calls (default: dry-run)")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default="https://open.bigmodel.cn/api/paas/v4")
    parser.add_argument("--model", default="glm-4-flash")
    parser.add_argument("--rpm", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--cost-cap", type=float, default=None)
    args = parser.parse_args(argv)

    import os  # noqa: PLC0415

    strategies = build_strategies(args.max_chars, args.overlap_chars)
    if args.strategy:
        strategies = {args.strategy: strategies[args.strategy]}
    pairs = build_pairs(strategies, SAMPLE_DOCS)

    api_key = args.api_key or os.environ.get(args.api_key_env)
    if args.live and not api_key:
        parser.error(f"--live needs an API key: pass --api-key or set ${args.api_key_env}")
    cfg = JudgeConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key if args.live else None,
    )
    runner = JudgeRunner(
        LLMJudge(cfg),
        state_path=args.state,
        rpm=args.rpm,
        max_retries=args.max_retries,
        cost_cap=args.cost_cap,
    )
    stats = runner.run(pairs, dry_run=not args.live)
    print(stats.summary())
    if args.state and Path(args.state).exists():
        print(json.dumps(summarize(args.state), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
