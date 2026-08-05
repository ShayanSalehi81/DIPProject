"""Condition-blinded prompt construction for Micro-Bench multiple-choice tasks."""

from __future__ import annotations

import string

from microq_robustness.models import SampleSpec

SYSTEM_PROMPT = (
    "You are evaluating a microscopy multiple-choice benchmark. Inspect the image carefully. "
    "Choose exactly one option using only visual evidence. Return JSON only in the form "
    '{"answer":"A"}. Do not explain your choice.'
)


def build_question_prompt(sample: SampleSpec) -> str:
    if len(sample.options) > len(string.ascii_uppercase):
        raise ValueError("The question has too many answer options.")
    choices = "\n".join(
        f"{string.ascii_uppercase[index]}. {option}"
        for index, option in enumerate(sample.options)
    )
    return f"Question: {sample.question}\n\nOptions:\n{choices}\n\nReturn only the JSON answer."

