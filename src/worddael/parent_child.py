"""Parent-child (small-to-big) chunking.

The pattern every Chinese RAG app template hand-rolls: index small child
chunks for precise matching, but return the large parent chunk each hit
belongs to, so the answer always arrives with its surrounding context.

Library-level guarantees here:
- children tile the text exactly and satisfy the same offset invariant as
  every other strategy (``child.text == source[child.start:child.end]``);
- each child carries ``meta["parent_seq"]``, and each parent
  ``meta["child_seqs"]``, so the family relation is data, not re-derivation;
- parents tile the text as well.
"""

from __future__ import annotations

from .chunkers import BaseChunker, RecursiveChunker
from .types import Chunk


class ParentChildChunker(BaseChunker):
    strategy = "parent-child"

    def __init__(
        self,
        parent_max_chars: int = 600,
        child_max_chars: int = 150,
        child_overlap_chars: int = 30,
        parent_overlap_chars: int = 0,
        explain: bool = False,
    ) -> None:
        if parent_max_chars < 1 or child_max_chars < 1:
            raise ValueError("parent_max_chars and child_max_chars must be >= 1")
        if child_max_chars >= parent_max_chars:
            raise ValueError("child_max_chars must be < parent_max_chars (degenerate families)")
        if not 0 <= child_overlap_chars < child_max_chars:
            raise ValueError("child_overlap_chars must be in [0, child_max_chars)")
        if not 0 <= parent_overlap_chars < parent_max_chars:
            raise ValueError("parent_overlap_chars must be in [0, parent_max_chars)")
        self.parent_max_chars = parent_max_chars
        self.child_max_chars = child_max_chars
        self.explain = explain
        self._parent_chunker = RecursiveChunker(
            max_chars=parent_max_chars, overlap_chars=parent_overlap_chars, explain=explain
        )
        self._child_chunker = RecursiveChunker(
            max_chars=child_max_chars, overlap_chars=child_overlap_chars, explain=explain
        )

    def chunk_families(self, text: str) -> tuple[list[Chunk], list[Chunk]]:
        """Return ``(parents, children)``.

        Children are absolute-offset slices of the source, tiling it; each
        carries ``meta["parent_seq"]``. Parents carry ``meta["child_seqs"]``.
        """
        parents_raw = self._parent_chunker.chunk(text)
        parents: list[Chunk] = []
        children: list[Chunk] = []
        for pi, parent in enumerate(parents_raw):
            parents.append(
                Chunk(
                    text=parent.text,
                    start=parent.start,
                    end=parent.end,
                    seq=pi,
                    strategy=self.strategy,
                    meta={"overlap_chars": parent.meta.get("overlap_chars", 0), "child_seqs": []},
                )
            )
            for inner in self._child_chunker.chunk(parent.text):
                child_meta = {
                    "parent_seq": pi,
                    "overlap_chars": inner.meta.get("overlap_chars", 0),
                }
                for rule_key in ("end_rule", "start_rule"):
                    if rule_key in inner.meta:
                        child_meta[rule_key] = inner.meta[rule_key]
                children.append(
                    Chunk(
                        text=text[parent.start + inner.start : parent.start + inner.end],
                        start=parent.start + inner.start,
                        end=parent.start + inner.end,
                        seq=len(children),
                        strategy=self.strategy,
                        meta=child_meta,
                    )
                )
                parents[pi].meta["child_seqs"].append(children[-1].seq)
        return parents, children

    def chunk(self, text: str) -> list[Chunk]:
        """Children only (tiling + offset invariant). For the parents as
        well, use :meth:`chunk_families`."""
        if not text:
            return []
        return self.chunk_families(text)[1]
