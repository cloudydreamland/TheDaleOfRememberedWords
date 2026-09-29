"""Evaluation subpackage: retrieval recall, LLM judging, reports.

Note: ``report`` is intentionally not imported here — it is the ``-m``
entry point (``python -m worddael.eval.report``) and stays import-light.
"""

from __future__ import annotations

from .llm_judge import JudgeConfig, LLMJudge
from .retrieval import BM25, recall_at_k, tokenize_zh
from .sample_data import SAMPLE_DOCS, all_questions

__all__ = [
    "BM25",
    "JudgeConfig",
    "LLMJudge",
    "SAMPLE_DOCS",
    "all_questions",
    "recall_at_k",
    "tokenize_zh",
]
