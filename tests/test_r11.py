"""R11: fullwidth punctuation and special-whitespace hardening.

Discovered from the 紅樓夢 real corpus (traditional reprints use '．' as
sentence end) and CJK typography (ideographic space U+3000, NBSP).
"""

from __future__ import annotations

from worddael.chunkers import RecursiveChunker
from worddael.explain import classify_end
from worddael.sentences import split_sentences


def test_fullwidth_full_stop_is_sentence_boundary():
    text = "他走了过来．她没有说话．雪落了下来"
    spans = split_sentences(text)
    assert text[spans[0][0] : spans[0][1]] == "他走了过来．"
    assert text[spans[1][0] : spans[1][1]] == "她没有说话．"


def test_recursive_splits_on_fullwidth_stop():
    text = "前一句用全角句点．后一句继续．" + "补足长度用的内容。" * 10
    chunks = RecursiveChunker(max_chars=20, overlap_chars=0).chunk(text)
    ends_ok = sum(1 for c in chunks if c.text.rstrip()[-1] in "。．！？")
    assert ends_ok >= len(chunks) - 1


def test_ideographic_space_and_nbsp_masked_hygiene():
    """Opener hidden behind U+3000 / NBSP before the boundary must still be
    fixed — whitespace checks use str.isspace(), not an ASCII literal."""
    text = "引用开始「什么内容都有一些\u3000\u3000这是补白的文字。继续往下写。\n\n更多内容让块超长。结尾"
    chunks = RecursiveChunker(max_chars=18, overlap_chars=0).chunk(text)
    for c in chunks[:-1]:
        assert c.text.rstrip()[-1] not in "「『“《（", f"ends on opener: {c.text[-6:]!r}"


def test_classify_end_with_fullwidth_space_run():
    text = "句子结束。\u3000\u3000下一段开始。内容若干"
    # boundary right after the sentence period (fullwidth spaces follow) —
    # the trailing ideographic spaces must not mask the sentence end
    assert classify_end(text, 6) == "sentence"
    assert classify_end(text, 13) == "sentence"


def test_explain_rules_with_fullwidth_punct():
    text = "第一句！第二句？第三句．第四句"
    chunks = RecursiveChunker(max_chars=8, overlap_chars=0, explain=True).chunk(text)
    rules = [c.meta["end_rule"] for c in chunks if c.text.rstrip()[-1] in "。．！？"]
    assert "sentence" in rules
