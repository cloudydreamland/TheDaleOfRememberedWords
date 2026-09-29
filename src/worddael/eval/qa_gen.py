"""Generative QA expansion: turn any document corpus into (question, gold) pairs.

This is the scaffold behind "百篇级基准": with an API key it generates QA
pairs per document at scale; without one, a deterministic dry-run mode
exercises the whole pipeline (parse → validate → dedupe → resume → JSONL).

Quality contract enforced for every record, live or dry:
- ``gold`` must be a verbatim substring of the source document (invalid
  generations are dropped and counted, never silently kept);
- questions are unique within a document.

Usage::

    python -m worddael.eval.qa_gen --docs-dir docs/corpus --out results/qa.jsonl            # dry-run
    python -m worddael.eval.qa_gen --docs-dir docs/corpus --out results/qa.jsonl --live ...  # with key
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

from ..sentences import split_sentences

GEN_PROMPT = """你是中文 RAG 评测集构建员。阅读下面的文档，生成 {n} 个问答对，用于检索评测。

要求：
1. question 必须是用户会真实问的问题，答案必须能在文档中找到；
2. gold 必须是文档中的**逐字连续片段**（10~30 字），包含回答该问题所需的核心信息；
3. 覆盖文档的不同章节，不要集中在开头；
4. 只输出 JSON 数组：[{{"question": "...", "gold": "..."}}]

文档：
{doc}
"""


def dry_run_pairs(doc_text: str, n: int = 3) -> list[dict]:
    """Deterministic heuristic QA: pick fact-flavored sentences as golds.

    Marked ``method="dry-run"`` — pipeline validation only, question
    quality is explicitly not a claim.
    """
    flavor = re.compile(r"[0-9０-９]|标准|费用|时间|流程|要求|周期|限制|方法")
    sentences = []
    for start, end in split_sentences(doc_text):
        s = doc_text[start:end].strip()
        if len(s) >= 12 and s[-1] in "。！？；" and flavor.search(s):
            sentences.append(s)
    picked: list[str] = []
    step = max(1, len(sentences) // max(1, n))
    for i in range(0, len(sentences), step):
        picked.append(sentences[i])
        if len(picked) == n:
            break
    return [
        {"question": f"（干跑）文档中关于「{s[:10]}…」的内容是什么？", "gold": s, "method": "dry-run"}
        for s in picked
    ]


class QAGenerator:
    """LLM-backed QA pair generator over an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        model: str = "glm-4-flash",
        api_key: str | None = None,
        timeout: int = 60,
        max_pairs: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_pairs = max_pairs

    def _call(self, doc_text: str) -> list[dict]:
        prompt = GEN_PROMPT.format(n=self.max_pairs, doc=doc_text[:6000])
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON array in response: {content[:120]}")
        parsed = json.loads(m.group(0))
        if not isinstance(parsed, list):
            raise ValueError("response JSON is not an array")
        return parsed

    def generate(self, doc_id: str, doc_text: str, dry_run: bool = True) -> tuple[list[dict], dict]:
        """Returns ``(valid_pairs, stats)``; invalid generations are dropped
        and reported in stats, never kept."""
        stats = {"doc_id": doc_id, "raw": 0, "kept": 0, "dropped_bad_gold": 0, "dropped_dupe": 0}
        raw = dry_run_pairs(doc_text, self.max_pairs) if dry_run else self._call(doc_text)
        stats["raw"] = len(raw)
        seen_questions: set[str] = set()
        kept: list[dict] = []
        for item in raw:
            q = str(item.get("question", "")).strip()
            gold = str(item.get("gold", "")).strip()
            if not q or not gold or gold not in doc_text:
                stats["dropped_bad_gold"] += 1
                continue
            if q in seen_questions:
                stats["dropped_dupe"] += 1
                continue
            seen_questions.add(q)
            kept.append({"question": q, "gold": gold, "method": "dry-run" if dry_run else "llm"})
            stats["kept"] += 1
        return kept, stats


class QAGenRunner:
    """Resume-able QA generation over a list of documents (JSONL state)."""

    def __init__(self, generator: QAGenerator, state_path: str | Path, sleep=time.sleep) -> None:
        self.generator = generator
        self.state_path = Path(state_path)
        self._sleep = sleep

    def load_done(self) -> set[str]:
        if not self.state_path.exists():
            return set()
        done = set()
        for line in self.state_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["doc_id"])
            except (json.JSONDecodeError, KeyError):
                continue
        return done

    def run(self, docs: list[dict], dry_run: bool = True, max_retries: int = 2) -> dict:
        done = self.load_done()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        agg = {"docs": 0, "skipped": 0, "kept": 0, "dropped": 0, "errors": 0}
        with self.state_path.open("a", encoding="utf-8") as f:
            for doc in docs:
                if doc["id"] in done:
                    agg["skipped"] += 1
                    continue
                pairs = None
                for attempt in range(max_retries + 1):
                    try:
                        pairs, stats = self.generator.generate(doc["id"], doc["text"], dry_run=dry_run)
                        break
                    except Exception as exc:  # noqa: BLE001 — recorded, retried
                        if attempt == max_retries:
                            f.write(
                                json.dumps(
                                    {"doc_id": doc["id"], "error": str(exc)[:200], "pairs": []},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            f.flush()
                            agg["errors"] += 1
                            pairs = None
                            break
                        self._sleep(2.0 * (2**attempt))
                if pairs is None:
                    continue
                f.write(
                    json.dumps(
                        {"schema": 1, "doc_id": doc["id"], "stats": stats, "pairs": pairs},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.flush()
                agg["docs"] += 1
                agg["kept"] += stats["kept"]
                agg["dropped"] += stats["dropped_bad_gold"] + stats["dropped_dupe"]
        return agg


def load_docs_dir(docs_dir: str | Path) -> list[dict]:
    """Scan a directory of .txt/.md files into the standard doc shape."""
    docs = []
    for p in sorted(Path(docs_dir).iterdir()):
        if p.suffix.lower() in {".txt", ".md", ".markdown"} and p.is_file():
            docs.append({"id": p.stem, "title": p.stem, "text": p.read_text(encoding="utf-8")})
    return docs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="worddael-eval-qa-gen")
    parser.add_argument("--docs-dir", required=True, help="directory of .txt/.md source documents")
    parser.add_argument("--out", default="results/qa.jsonl")
    parser.add_argument("--n-per-doc", type=int, default=3)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default="https://open.bigmodel.cn/api/paas/v4")
    parser.add_argument("--model", default="glm-4-flash")
    args = parser.parse_args(argv)

    import os  # noqa: PLC0415

    docs = load_docs_dir(args.docs_dir)
    if not docs:
        parser.error(f"no .txt/.md documents found in {args.docs_dir}")
    api_key = args.api_key or os.environ.get(args.api_key_env)
    if args.live and not api_key:
        parser.error(f"--live needs an API key: pass --api-key or set ${args.api_key_env}")

    generator = QAGenerator(
        base_url=args.base_url, model=args.model, api_key=api_key if args.live else None,
        max_pairs=args.n_per_doc,
    )
    agg = QAGenRunner(generator, state_path=args.out).run(docs, dry_run=not args.live)
    print(json.dumps(agg, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
