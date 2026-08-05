"""Conservative parsing of multiple-choice responses from an instruction VLM."""

from __future__ import annotations

import json
import re
import string
import unicodedata

from microq_robustness.models import ScoredAnswer


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\w]+", " ", value).strip()


class MultipleChoiceScorer:
    """Parse explicit JSON/letter/index answers without guessing ambiguous prose."""

    def score(self, response: str, options: tuple[str, ...]) -> ScoredAnswer:
        text = response.strip()
        if not text:
            return ScoredAnswer(None, "empty", "unparseable")
        valid_letters = string.ascii_uppercase[: len(options)]

        parsed = self._from_json(text, valid_letters)
        if parsed is not None:
            return ScoredAnswer(parsed, "json", "parsed")

        explicit = re.findall(
            rf"(?i)(?:answer|option|choice)\s*(?:is|:|=|-)?\s*([A-{valid_letters[-1]}])\b",
            text,
        )
        unique_explicit = {letter.upper() for letter in explicit}
        if len(unique_explicit) == 1:
            return ScoredAnswer(
                valid_letters.index(next(iter(unique_explicit))), "explicit_letter", "parsed"
            )

        full_letter = re.fullmatch(
            rf"\s*(?:\{{?\s*)?([A-{valid_letters[-1]}])(?:\s*[.)}}]?\s*)",
            text,
            flags=re.IGNORECASE,
        )
        if full_letter:
            return ScoredAnswer(
                valid_letters.index(full_letter.group(1).upper()), "standalone_letter", "parsed"
            )

        explicit_number = re.findall(
            r"(?i)(?:answer|option|choice)\s*(?:is|:|=|-)?\s*(\d+)\b", text
        )
        number_values = {int(value) for value in explicit_number}
        if len(number_values) == 1:
            value = next(iter(number_values))
            if 1 <= value <= len(options):
                return ScoredAnswer(value - 1, "explicit_one_based_index", "parsed")

        normalized_response = _normalized(text)
        exact_matches = [
            index
            for index, option in enumerate(options)
            if _normalized(option) == normalized_response
        ]
        if len(exact_matches) == 1:
            return ScoredAnswer(exact_matches[0], "exact_option_text", "parsed")

        return ScoredAnswer(None, "ambiguous", "unparseable")

    @staticmethod
    def _from_json(text: str, valid_letters: str) -> int | None:
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{[^{}]*\}", stripped, flags=re.DOTALL)
            if not match:
                return None
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        if not isinstance(payload, dict):
            return None
        value = payload.get("answer")
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in valid_letters:
                return valid_letters.index(normalized)
            if normalized.isdigit() and 1 <= int(normalized) <= len(valid_letters):
                return int(normalized) - 1
        if isinstance(value, int) and 1 <= value <= len(valid_letters):
            return value - 1
        return None
