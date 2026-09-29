"""Real-corpus dogfooding + throughput benchmark.

Downloads 《紅樓夢》 (Project Gutenberg pg24264, public domain) if missing,
strips the Gutenberg wrapper, runs worddael's pipeline over the full text and
reports: throughput, chunk shape, boundary-rule distribution (explain), and
punctuation-hygiene violations. Writes docs/dogfood.md.

Run::

    .venv/Scripts/python.exe tools/dogfood.py
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

from worddael.chunkers import MarkdownChunker, RecursiveChunker
from worddael.sentences import split_sentences

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "dogfood_corpus"
RAW = CORPUS / "hongloumeng_raw.txt"
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/24264/pg24264.txt"
MAX_CHARS = 400
OVERLAP = 50
OPENERS = set("「『“《（〔［｛『")


def ensure_corpus() -> str:
    CORPUS.mkdir(exist_ok=True)
    if not RAW.exists():
        print(f"downloading {GUTENBERG_URL} ...")
        urllib.request.urlretrieve(GUTENBERG_URL, RAW)
    text = RAW.read_text(encoding="utf-8", errors="replace")
    start = text.find("*** START")
    end = text.find("*** END")
    if start >= 0 and end >= 0:
        text = text[text.find("\n", start) + 1 : end]
    return text.strip()


def main() -> None:
    text = ensure_corpus()
    n_chars = len(text)
    n_sentences = len(split_sentences(text))
    print(f"corpus: 紅樓夢 {n_chars:,} chars, {n_sentences:,} sentences")

    chunker = RecursiveChunker(max_chars=MAX_CHARS, overlap_chars=OVERLAP, explain=True)

    started = time.perf_counter()
    chunks = chunker.chunk(text)
    elapsed = time.perf_counter() - started
    throughput = n_chars / elapsed / 1e6

    rules = Counter(c.meta["end_rule"] for c in chunks)
    lens = [c.n_chars for c in chunks]
    over_budget = sum(1 for c in chunks if c.n_chars > MAX_CHARS + OVERLAP)
    hygiene_end_openers = sum(1 for c in chunks if c.text.rstrip() and c.text.rstrip()[-1] in OPENERS)
    hygiene_start_closers = sum(
        1 for i, c in enumerate(chunks) if i > 0 and c.meta["overlap_chars"] == 0 and c.text and c.text[0] in "，。！？；：、」』”）"
    )

    # markdown pipeline over the same content wrapped in fake chapter headings
    md_lines = text.split("\n")
    md = "\n".join(
        (f"## 第{i + 1}回\n\n" if i % 60 == 0 else "") + line for i, line in enumerate(md_lines[:20000])
    )
    md_chunker = MarkdownChunker(max_chars=MAX_CHARS, overlap_chars=OVERLAP, explain=True)
    md_started = time.perf_counter()
    md_chunks = md_chunker.chunk(md)
    md_elapsed = time.perf_counter() - md_started
    md_rules = Counter(c.meta["end_rule"] for c in md_chunks)

    report = {
        "corpus": {"chars": n_chars, "sentences": n_sentences, "source": "Project Gutenberg pg24264 紅樓夢 (public domain)"},
        "recursive": {
            "max_chars": MAX_CHARS,
            "overlap": OVERLAP,
            "chunks": len(chunks),
            "avg_chars": round(sum(lens) / max(1, len(lens)), 1),
            "min_chars": min(lens),
            "max_chars": max(lens),
            "throughput_MBps": round(throughput, 3),
            "elapsed_s": round(elapsed, 3),
            "over_budget": over_budget,
            "hygiene_end_opener_violations": hygiene_end_openers,
            "hygiene_start_closer_violations": hygiene_start_closers,
            "end_rule_distribution": dict(rules.most_common()),
        },
        "markdown": {
            "wrapped_chars": len(md),
            "chunks": len(md_chunks),
            "throughput_MBps": round(len(md) / md_elapsed / 1e6, 3),
            "end_rule_distribution": dict(md_rules.most_common()),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out = ROOT / "dogfood_stats.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"stats -> {out}")


if __name__ == "__main__":
    main()
