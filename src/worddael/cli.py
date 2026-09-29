"""Command-line interface.

Examples::

    worddael doc.md --stats
    worddael manual.txt --strategy recursive --max-chars 400 --overlap 50 --jsonl out.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, chunk_file
from .types import Chunk


def _configure_utf8_output() -> None:
    """Keep Chinese CLI output usable on legacy Windows console encodings."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("_", "-")
        if reconfigure is not None and encoding not in {"utf-8", "utf8"}:
            reconfigure(encoding="utf-8")


def _stats(chunks: list[Chunk]) -> str:
    if not chunks:
        return "no chunks"
    lens = [c.n_chars for c in chunks]
    sentence_end = "。！？；…!?;"
    end_ok = sum(1 for c in chunks if c.text and c.text[-1] in sentence_end)
    lines = [
        f"chunks: {len(chunks)}",
        f"chars: min={min(lens)} median={sorted(lens)[len(lens) // 2]} max={max(lens)} total={sum(lens)}",
        f"ends_on_sentence_punct: {end_ok}/{len(chunks)}",
    ]
    return "\n".join(lines)


def _demo() -> int:
    """`worddael demo` — zero-setup tour: sample text, offsets, rules, compare."""
    from .chunkers import RecursiveChunker  # noqa: PLC0415
    from .compare import compare_text, format_table  # noqa: PLC0415
    from .eval.sample_data import SAMPLE_DOCS  # noqa: PLC0415

    doc = SAMPLE_DOCS[0]
    print(f"worddael demo — 样例文档《{doc['title']}》（{len(doc['text'])} 字，原创内置语料）")
    print()
    print("== recursive / max_chars=100 / overlap=20（explain 开启）==")
    chunker = RecursiveChunker(max_chars=100, overlap_chars=20, explain=True)
    for c in chunker.chunk(doc["text"]):
        body = c.text.replace("\n", "")
        flag = "句尾✓" if body.rstrip()[-1] in "。！？；…!?;" else "句尾✗"
        print(
            f"[{c.seq:>2}] {c.start:>3}-{c.end:<3} {flag} end_rule={c.meta['end_rule']:<9} {body[:24]}…"
        )
    print()
    print("== 策略对比（同预算）==")
    print(format_table(compare_text(doc["text"], max_chars=150, overlap_chars=30)))
    print()
    print("试试: worddael 你的文件.txt --stats  或  worddael 文档.md --compare")
    return 0


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_output()
    parser = argparse.ArgumentParser(
        prog="worddael",
        description="Worddael — Chinese-first text chunking for RAG",
    )
    parser.add_argument("file", help="text or markdown file to chunk (or 'demo' for a tour)")
    parser.add_argument(
        "--strategy",
        choices=["recursive", "sentence", "token", "markdown", "semantic"],
        default=None,
        help="chunking strategy (default: markdown for .md, else recursive)",
    )
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--overlap-chars", type=int, default=50)
    parser.add_argument("--jsonl", metavar="PATH", help="write chunks as JSONL")
    parser.add_argument("--stats", action="store_true", help="print summary statistics")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="compare all strategies side by side instead of chunking with one",
    )
    parser.add_argument("--version", action="version", version=f"worddael {__version__}")
    args = parser.parse_args(argv)

    if args.file == "demo":
        return _demo()

    if args.compare:
        from pathlib import Path  # noqa: PLC0415

        from .compare import compare_text, format_table  # noqa: PLC0415

        try:
            text = Path(args.file).read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"error: file not found: {args.file} (checked working directory; try `worddael demo` first)", file=sys.stderr)
            return 1
        except OSError as exc:
            print(f"error: cannot read {args.file}: {exc}", file=sys.stderr)
            return 1
        print(format_table(compare_text(text, max_chars=args.max_chars, overlap_chars=args.overlap_chars)))
        return 0

    kwargs = {"max_chars": args.max_chars}
    if args.strategy != "token":
        kwargs["overlap_chars"] = min(args.overlap_chars, max(0, args.max_chars // 10))

    try:
        chunks = chunk_file(args.file, strategy=args.strategy, **kwargs)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.jsonl:
        from .io_utils import to_jsonl

        out = to_jsonl(chunks, args.jsonl)
        print(f"wrote {len(chunks)} chunks -> {out}")
    else:
        for c in chunks:
            print(json.dumps(c.to_dict(), ensure_ascii=False))

    if args.stats:
        print("---", file=sys.stderr)
        print(_stats(chunks), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
