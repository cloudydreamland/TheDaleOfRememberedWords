"""Tests for token counters."""

from __future__ import annotations

import pytest

from worddael.counters import CharCounter, get_counter


def test_char_counter_cjk_vs_latin():
    c = CharCounter()
    zh = c.count("你好世界")  # 4 CJK chars
    en = c.count("helloworld")  # 10 latin chars ~ 3 groups
    assert zh == 4
    assert en < 6
    assert c.count("") == 0


def test_char_counter_monotonic():
    c = CharCounter()
    assert c.count("短") <= c.count("短一点") <= c.count("短一点的文本内容更多")


def test_get_counter_unknown():
    with pytest.raises(ValueError):
        get_counter("nope")


def test_jieba_counter_optional():
    pytest.importorskip("jieba")
    from worddael.counters import JiebaCounter

    jc = JiebaCounter()
    n = jc.count("中文文本切分测试")
    assert 2 <= n <= 8
