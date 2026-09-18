#!/usr/bin/env python3
"""
Benchmark parser for Lulu evaluation.

- math500 / aime25 / olympiadbench: use Lulu's bundled validated math parser.
- mmlu_pro / gpqa_diamond: parse a final multiple-choice letter.
- livecodebench: scoring is deferred to the official LiveCodeBench evaluator; this parser
  deliberately returns a non-matching sentinel during generation.
"""
from __future__ import annotations

import re
from lulu import math_parser

_CHOICE_PREFIX = "__CHOICE__"
_LCB_PRED = "__LCB_DEFERRED_PRED__"
_LCB_GT = "__LCB_DEFERRED_GT__"

def _math_parser():
    return math_parser

def _choice_letter(text: str) -> str:
    # Prioritize explicit final-answer language and search from the end.
    tail = str(text)[-1600:]
    # A one-letter LaTeX formatting wrapper is still an explicit option label.
    # Normalize only this narrow form, never arbitrary words or expressions.
    tail = re.sub(r"\\(?:text|mathrm|mathbf|textbf|mathit)\s*\{\s*([A-J])\s*\}",
                  r"\1", tail, flags=re.IGNORECASE)
    patterns = [
        r"(?is)(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\-]?\s*\(?\s*([A-J])(?![A-Za-z])\s*\)?",
        r"(?is)(?:final\s+answer|answer)\s*[:\-]\s*\(?\s*([A-J])(?![A-Za-z])\s*\)?",
        r"(?is)\\boxed\s*\{\s*([A-J])\s*\}",
        r"(?im)^\s*\(\s*([A-J])\s*\)\s*$",
        r"(?im)^\s*([A-J])\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, tail)
        if matches:
            return str(matches[-1]).upper()
    # Fallback only after the existing recognized-label rules fail. A formatted
    # explicit label can include a description, e.g. **B. -0.7** or
    # \boxed{\text{The answer is } G}. Never map a numeric/formula answer to an
    # option, or treat the C of C6H12O / C_6 as an option label.
    plain = tail.replace("**", "").replace("__", "")
    plain = re.sub(r"\\(?:text|mathrm|mathbf|textbf|mathit)\s*\{([^{}]*)\}", r"\1", plain)
    fallback = [
        r"(?is)(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\-]?\s*\(?\s*([A-J])(?![A-Za-z0-9_])\s*\)?",
        r"(?is)(?:final\s+answer|answer)\s*[:\-]\s*\(?\s*([A-J])(?![A-Za-z0-9_])\s*\)?",
        r"(?is)\\boxed\s*\{\s*([A-J])\s*(?:[.)]\s|\})",
    ]
    matches = [(match.start(), match.group(1).upper())
               for pattern in fallback for match in re.finditer(pattern, plain)]
    return max(matches)[1] if matches else ""


def extract_answer(text: str, data_source: str):
    source = str(data_source).lower()
    if source in {"mmlu_pro", "gpqa_diamond"}:
        letter = _choice_letter(text)
        return f"{_CHOICE_PREFIX}{letter}" if letter else f"{_CHOICE_PREFIX}?"
    if source == "livecodebench":
        return _LCB_PRED
    if source == "olympiadbench":
        # This prepared benchmark stores one final answer per problem, including
        # tuples inside a single box. Concatenating every box also collected
        # draft answers and repeated conclusions from thinking traces. Use the
        # same last-box policy as the other math benchmarks, without consulting gold.
        return _math_parser().extract_answer(text, "math500")
    return _math_parser().extract_answer(text, data_source)


def math_equal(pred, ground_truth) -> bool:
    p = str(pred)
    g = str(ground_truth)
    if g.startswith(_CHOICE_PREFIX):
        return p == g
    if g == _LCB_GT:
        return False
    return bool(_math_parser().math_equal(pred, ground_truth))
