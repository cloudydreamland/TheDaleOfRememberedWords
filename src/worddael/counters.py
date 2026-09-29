"""Token counters used for budget-based chunking.

The core library has zero required dependencies: the default ``CharCounter``
is a pure-Python heuristic tuned for mixed Chinese text (CJK chars count as
~1 token each, Latin runs are grouped into ~4-char pieces, approximating
BPE behaviour of popular tokenizers). Pluggable exact counters (jieba /
tiktoken) are available as optional extras.
"""

from __future__ import annotations

import math
from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


def chars_for_tokens(counter: TokenCounter, text: str, n_tokens: int) -> int:
    """Best-effort reverse mapping: how many chars of ``text`` correspond to
    ``n_tokens`` under ``counter``. Used by TokenChunker for oversized units
    and overlap windows. Works for any counter."""
    if not text:
        return 0
    measured = counter.count(text)
    if measured == 0:
        return len(text)
    return max(1, round(n_tokens * len(text) / measured))


def calibration_report(
    counter: TokenCounter, reference: TokenCounter, texts: list[str]
) -> dict:
    """Quantify how far a cheap counter's token estimates drift from a
    reference counter over a corpus.

    Returns per-text absolute relative error stats. Honest interpretation:
    a heuristic counter with small mean error is fine for *budget
    chunking*; none of them are billing-grade. Use exact counters
    (jieba/tiktoken) as ``reference`` and the heuristic as ``counter``.
    """
    if not texts:
        return {"n": 0, "mean_abs_rel_err": 0.0, "max_abs_rel_err": 0.0, "per_text": []}
    per_text = []
    for t in texts:
        est = counter.count(t)
        ref = reference.count(t)
        err = abs(est - ref) / ref if ref else 0.0
        per_text.append({"chars": len(t), "estimate": est, "reference": ref, "abs_rel_err": round(err, 4)})
    errs = [p["abs_rel_err"] for p in per_text]
    return {
        "n": len(texts),
        "mean_abs_rel_err": round(sum(errs) / len(errs), 4),
        "max_abs_rel_err": round(max(errs), 4),
        "per_text": per_text,
    }


def _is_cjk(cp: int) -> bool:
    return (
        0x2E80 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
        or 0xFF00 <= cp <= 0xFF60  # fullwidth forms
        or 0x3000 <= cp <= 0x303F  # CJK punctuation
    )


class CharCounter:
    """Dependency-free heuristic counter for mixed Chinese/Latin text."""

    def count(self, text: str) -> int:
        if not text:
            return 0
        tokens = 0
        latin_run = 0
        for ch in text:
            cp = ord(ch)
            if _is_cjk(cp):
                if latin_run:
                    tokens += math.ceil(latin_run / 4) + 1
                    latin_run = 0
                tokens += 1
            elif ch.isspace():
                if latin_run:
                    tokens += math.ceil(latin_run / 4) + 1
                    latin_run = 0
                tokens += 1
            else:
                latin_run += 1
        if latin_run:
            tokens += math.ceil(latin_run / 4) + 1
        return tokens


class JiebaCounter:
    """Word-level counter backed by jieba. Optional dependency."""

    def __init__(self) -> None:
        import jieba  # noqa: PLC0415

        self._cut = jieba.cut

    def count(self, text: str) -> int:
        return sum(1 for tok in self._cut(text) if not tok.isspace())


class TiktokenCounter:
    """Exact BPE counter backed by tiktoken. Optional dependency."""

    def __init__(self, encoding: str = "cl100k_base") -> None:
        import tiktoken  # noqa: PLC0415

        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


def get_counter(name: str = "char", **kwargs) -> TokenCounter:
    name = name.lower()
    if name == "char":
        return CharCounter()
    if name == "jieba":
        return JiebaCounter()
    if name == "tiktoken":
        return TiktokenCounter(**kwargs)
    raise ValueError(f"unknown counter: {name!r} (expected char|jieba|tiktoken)")
