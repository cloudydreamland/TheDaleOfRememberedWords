"""Chinese-aware low-level text structure analysis.

Two jobs:
1. Sentence boundary detection tuned for Chinese punctuation.
2. Markdown section parsing with heading paths and atomic code fences.

All functions return spans ``(start, end)`` that tile the source string
exactly (``text[start:end]`` slices, ``end`` exclusive), so callers can
build offset-exact chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Sentence-ending punctuation (Chinese first, ASCII fallbacks).
# Includes the fullwidth full stop "．" (common in traditional-Chinese
# reprints, e.g. the Gutenberg 紅樓夢 text) and small/variant forms.
SENT_END = set("。！？；!?;…．﹖﹗‥⋯")
# Closers that belong to the preceding sentence: 」』”）)》】〉 etc.
CLOSERS = set("」』”）)》】〉>'\"")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_OPEN_RE = re.compile(r"^\s*(```|~~~)")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into sentence spans.

    A boundary is placed after sentence-ending punctuation plus any run of
    closing quotes/brackets, or after a newline. ASCII ``.`` is deliberately
    *not* a boundary (protects decimals, versions, URLs); Chinese ``。`` is.
    """
    spans: list[tuple[int, int]] = []
    n = len(text)
    start = 0
    i = 0
    while i < n:
        ch = text[i]
        if ch in SENT_END:
            j = i + 1
            while j < n and text[j] in CLOSERS:
                j += 1
            spans.append((start, j))
            start = j
            i = j
        elif ch == "\n":
            if i > start:
                spans.append((start, i))
            spans.append((i, i + 1))
            start = i + 1
            i += 1
        else:
            i += 1
    if start < n:
        spans.append((start, n))
    return spans


@dataclass
class Section:
    """A markdown section: the heading line plus everything under it."""

    headings: tuple[str, ...]
    start: int
    end: int


def markdown_sections(text: str) -> list[Section]:
    """Parse ATX headings (``#``..``######``) into sections.

    - Returns sections that tile ``[0, len(text))``.
    - Content before the first heading gets the empty heading path.
    - Fenced code blocks (``` or ~~~) are never scanned for headings and
      stay attached to the section they opened in.
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    heading_path: tuple[str, ...] = ()
    seg_start = 0
    offset = 0
    in_fence = False
    fence_marker = ""

    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        if in_fence:
            if fence_marker in stripped:
                in_fence = False
        else:
            m = _FENCE_OPEN_RE.match(stripped)
            if m:
                in_fence = True
                fence_marker = m.group(1)
            else:
                hm = _HEADING_RE.match(stripped)
                if hm:
                    if offset > seg_start:
                        sections.append(Section(heading_path, seg_start, offset))
                    level = len(hm.group(1))
                    title = hm.group(2)
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, title))
                    heading_path = tuple(t for _, t in stack)
                    seg_start = offset
        offset += len(line)

    if offset > seg_start:
        sections.append(Section(heading_path, seg_start, offset))
    if not sections:
        sections.append(Section((), 0, len(text)))
    return sections
