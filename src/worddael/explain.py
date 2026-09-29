"""Explainable chunking: annotate every chunk boundary with its cause.

The classifier is *post-hoc by design*: it derives the rule from the actual
source characters around each boundary, so the explanation can never drift
from what was really cut (no bookkeeping to fall out of sync). Chunkers
that know better (markdown headings, semantic drops) may override per-span.

Zero cost unless ``explain=True`` is passed to a chunker.
"""

from __future__ import annotations

SENT_END = set("。！？；!?;…")
CLAUSE = set("，、,:：")

# rules recorded in chunk meta
RULE_TEXT_START = "text_start"
RULE_TEXT_END = "text_end"
RULE_HEADING = "heading"
RULE_SEMANTIC_DROP = "semantic_drop"


def classify_end(text: str, end: int) -> str:
    """Why did a chunk end at ``end`` (exclusive)?

    Inspects the characters immediately before the boundary (separators are
    attached to the preceding part by the splitters, so the boundary cause
    is inside the chunk tail):
    - trailing newline run → ``paragraph`` (blank line, or sentence+newline)
      or ``newline`` (single hard line cut on non-sentence content)
    - sentence-ending punctuation → ``sentence``
    - clause punctuation → ``clause``
    - nothing recognizable (cut landed on content) → ``budget``
    - ``end`` at end of source → ``text_end``
    """
    if end >= len(text):
        return RULE_TEXT_END
    j = end - 1
    if j < 0:
        return "start"
    # walk back over ALL whitespace (including fullwidth space, NBSP); count
    # the newlines separately — they decide paragraph vs line cuts
    k = j
    newlines = 0
    while k >= 0 and text[k].isspace():
        if text[k] in "\r\n":
            newlines += 1
        k -= 1
    if newlines:
        if newlines >= 2 or (k >= 0 and text[k] in SENT_END):
            return "paragraph"
        return "newline"
    ch = text[k] if k >= 0 else text[j]
    if ch in SENT_END:
        return "sentence"
    if ch in CLAUSE:
        return "clause"
    return "budget"


def annotate(
    text: str,
    spans: list[tuple[int, int, int]],
    metas: list[dict],
    end_rule_overrides: dict[int, str] | None = None,
) -> None:
    """Fill ``end_rule``/``start_rule`` into ``metas`` in place.

    ``spans`` are ``(start, end, overlap_chars)`` triples aligned 1:1 with
    ``metas``. ``end_rule_overrides`` lets strategy-aware chunkers replace
    the post-hoc verdict for specific chunk indexes (e.g. markdown
    ``heading`` or ``semantic_drop``).
    """
    overrides = end_rule_overrides or {}
    rules: list[str] = []
    for i, (s, e, _ov) in enumerate(spans):
        rule = overrides.get(i) or classify_end(text, e)
        rules.append(rule)
        metas[i]["end_rule"] = rule
    prev = RULE_TEXT_START
    for i, meta in enumerate(metas):
        meta["start_rule"] = prev
        prev = rules[i]
