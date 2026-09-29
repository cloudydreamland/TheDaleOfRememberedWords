"""Status-quo experiment: how do careless defaults cut Chinese text?

Runs LangChain's RecursiveCharacterTextSplitter (the de-facto default in RAG
tutorials) against worddael's RecursiveChunker on the same Chinese text and the
same char budget, then reports structural quality metrics and answer-split
incidents. Every number in docs/status_quo.md comes from this script.

Run::

    .venv/Scripts/python.exe tools/status_quo_experiment.py
"""

from __future__ import annotations

import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

from worddael import RecursiveChunker

TEXT = (
    "第三章 差旅报销。员工出差乘坐高铁的，二等座可全额报销，一等座需部门总监审批。"
    "住宿费报销标准为每晚三百元，一线城市可上浮至四百五十元。"
    "市内交通费凭发票据实报销，每天上限八十元。\n\n"
    "第四章 报销时限。所有发票必须在费用发生后三十天内提交，"
    "逾期财务系统将自动关闭该笔报销入口。报销款在审批后5个工作日内到账。"
)
GOLDS = ["每晚三百元", "三十天内提交", "每天上限八十元", "审批后5个工作日内到账"]
SENT_END = "。！？；…!?"


def lc_chunks(separators: list[str], size: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(separators=separators, chunk_size=size, chunk_overlap=0)
    return [{"text": c, "start": TEXT.find(c)} for c in splitter.split_text(TEXT)]


def qg_chunks(size: int) -> list[dict]:
    return [
        {"text": c.text, "start": c.start}
        for c in RecursiveChunker(max_chars=size, overlap_chars=0).chunk(TEXT)
    ]


def metrics(chunks: list[dict]) -> dict:
    ends = 0
    starts_mid = 0
    for c in chunks:
        body = c["text"].rstrip()
        if body and body[-1] in SENT_END:
            ends += 1
        s = c["start"]
        if s > 0 and TEXT[s - 1] not in SENT_END and TEXT[s - 1] != "\n":
            starts_mid += 1
    n = max(1, len(chunks))
    return {
        "n_chunks": len(chunks),
        "ends_on_sentence_frac": round(ends / n, 3),
        "starts_mid_sentence_frac": round(starts_mid / n, 3),
    }


def answer_integrity(chunks: list[dict]) -> dict:
    out = {}
    for gold in GOLDS:
        intact = any(gold in c["text"] for c in chunks)
        out[gold] = "intact" if intact else "SPLIT across chunks"
    return out


def main() -> None:
    report = {
        "text_chars": len(TEXT),
        "langchain_default_80": {
            "chunks": lc_chunks(["\n\n", "\n", " ", ""], 80),
            "metrics": metrics(lc_chunks(["\n\n", "\n", " ", ""], 80)),
            "answers": answer_integrity(lc_chunks(["\n\n", "\n", " ", ""], 80)),
        },
        "langchain_custom_zh_80": {
            "chunks": lc_chunks(["\n\n", "。", "，", ""], 80),
            "metrics": metrics(lc_chunks(["\n\n", "。", "，", ""], 80)),
            "answers": answer_integrity(lc_chunks(["\n\n", "。", "，", ""], 80)),
        },
        "worddael_recursive_80": {
            "chunks": qg_chunks(80),
            "metrics": metrics(qg_chunks(80)),
            "answers": answer_integrity(qg_chunks(80)),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
