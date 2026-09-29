"""File IO helpers: chunk a document, serialize chunks to JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from .chunkers import BaseChunker
from .types import Chunk

_MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx"}


def chunker_for_path(path: str | Path, default_strategy: str = "recursive", **kwargs) -> BaseChunker:
    """Pick a sensible default strategy from the file extension."""
    from .chunkers import STRATEGIES  # noqa: PLC0415

    path = Path(path)
    strategy = "markdown" if path.suffix.lower() in _MARKDOWN_SUFFIXES else default_strategy
    kwargs.setdefault("max_chars", 500)
    kwargs.setdefault("overlap_chars", min(50, max(0, int(kwargs["max_chars"]) // 10)))
    return STRATEGIES[strategy](**kwargs)


def chunk_file(path: str | Path, strategy: str | None = None, **kwargs) -> list[Chunk]:
    """Chunk a text/markdown file. Markdown files default to the markdown
    strategy unless ``strategy`` is given explicitly."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if strategy is None:
        return chunker_for_path(path, **kwargs).chunk(text)
    from .chunkers import STRATEGIES  # noqa: PLC0415

    return STRATEGIES[strategy](**kwargs).chunk(text)


def to_jsonl(chunks: list[Chunk], path: str | Path) -> Path:
    """Write chunks to a JSONL file (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    return path
