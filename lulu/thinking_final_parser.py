"""Score the final assistant answer of a thinking response, never its draft.

A missing closing thought marker is an unfinished answer, even if a candidate
number appears in the reasoning. Keep the existing math/choice equivalence rules.
"""
from lulu import benchmark_parser as legacy

MISSING_FINAL = '__LULU_MISSING_FINAL_ANSWER__'


def extract_answer(text, data_source):
    if '</think>' not in str(text):
        return MISSING_FINAL
    final=str(text).rsplit('</think>',1)[1].strip()
    return legacy.extract_answer(final,data_source) if final else MISSING_FINAL


def math_equal(prediction, ground_truth):
    return False if prediction==MISSING_FINAL else legacy.math_equal(prediction,ground_truth)
