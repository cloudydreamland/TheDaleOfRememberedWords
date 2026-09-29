"""Core data types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Chunk:
    """A chunk of text with exact character offsets into its source string.

    Invariant: ``chunk.text == source[chunk.start:chunk.end]`` always holds.
    ``start``/``end`` are Python-style slices, i.e. ``end`` is exclusive.
    This invariant is what makes reliable citation possible in RAG answers.
    """

    text: str
    start: int
    end: int
    seq: int = 0
    strategy: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "seq": self.seq,
            "strategy": self.strategy,
            "meta": self.meta,
        }
