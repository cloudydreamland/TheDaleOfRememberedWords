"""Side-by-side strategy comparison for a single text.

Powers ``worddael FILE --compare``. All structural metrics are computed the
same way as the eval report so numbers stay comparable.
"""

from __future__ import annotations

import time

from .chunkers import RecursiveChunker, SemanticChunker, SentenceChunker, TokenChunker
from .embedders import HashingEmbedder
from .eval.report import FixedWindowChunker

_PUNCT = "。！？；…!?;"


def _row(name: str, chunks: list, text_len: int, elapsed_ms: float) -> dict:
    lens = [c.n_chars for c in chunks]
    end_ok = sum(1 for c in chunks if c.text.rstrip() and c.text.rstrip()[-1] in _PUNCT)
    return {
        "strategy": name,
        "chunks": len(chunks),
        "avg_chars": round(sum(lens) / max(1, len(lens)), 1),
        "min_chars": min(lens) if lens else 0,
        "max_chars": max(lens) if lens else 0,
        "ends_punct": f"{end_ok}/{len(lens)}",
        # coverage signal: ~1.0 for non-overlapping tiling, >1.0 with overlap
        "coverage": round(sum(c.end - c.start for c in chunks) / max(1, text_len), 3),
        "time_ms": elapsed_ms,
    }


def compare_text(text: str, max_chars: int = 300, overlap_chars: int = 50) -> list[dict]:
    """Run every strategy over ``text`` and return per-strategy metric rows."""
    strategies: list[tuple[str, object]] = [
        ("fixed-window", FixedWindowChunker(max_chars=max_chars, overlap_chars=overlap_chars)),
        ("sentence", SentenceChunker(max_chars=max_chars, overlap_chars=overlap_chars)),
        ("token", TokenChunker(max_tokens=max_chars, overlap_tokens=overlap_chars)),
        ("recursive", RecursiveChunker(max_chars=max_chars, overlap_chars=overlap_chars)),
        (
            "semantic-hash-toy",
            SemanticChunker(
                embed_fn=HashingEmbedder(),
                max_chars=max_chars,
                min_chars=max(20, max_chars // 3),
                overlap_chars=overlap_chars,
            ),
        ),
    ]
    rows = []
    for name, chunker in strategies:
        started = time.perf_counter()
        chunks = chunker.chunk(text)
        elapsed = (time.perf_counter() - started) * 1000
        row = _row(name, chunks, len(text), round(elapsed, 1))
        rows.append(row)
    return rows


def format_table(rows: list[dict]) -> str:
    cols = ["strategy", "chunks", "avg_chars", "min_chars", "max_chars", "ends_punct", "time_ms"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)
