from __future__ import annotations

import re
from typing import Any


class AnswerFormatError(ValueError):
    pass


def normalize_answer(raw_answer: Any, answer_format: str, options: dict[str, Any]) -> str:
    letters = sorted(str(key).upper() for key in options)
    valid = set(letters)
    if isinstance(raw_answer, list):
        answer = "".join(str(item).upper().strip() for item in raw_answer)
    else:
        answer = str(raw_answer or "").upper().strip()
    answer = re.sub(r"[^A-Z]", "", answer)

    if not answer:
        raise AnswerFormatError("answer is empty")
    unknown = set(answer).difference(valid)
    if unknown:
        raise AnswerFormatError(f"answer contains invalid option letters: {sorted(unknown)}")

    if answer_format in {"mcq", "tf"}:
        if len(answer) != 1:
            raise AnswerFormatError(f"{answer_format} requires exactly one option letter")
        return answer

    if answer_format == "multi":
        normalized = "".join(letter for letter in letters if letter in set(answer))
        if not normalized:
            raise AnswerFormatError("multi answer is empty after normalization")
        return normalized

    raise AnswerFormatError(f"unsupported answer format: {answer_format}")


def option_text(question: dict[str, Any]) -> str:
    options = question.get("options") or {}
    return "\n".join(f"{key}. {value}" for key, value in sorted(options.items()))

