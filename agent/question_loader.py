from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.io_utils import read_json


def load_questions(question_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(question_dir)
    if root.is_file():
        data = read_json(root)
        if not isinstance(data, list):
            raise ValueError(f"question file must contain a JSON list: {root}")
        return [_validate_question(item, root) for item in data]

    files = sorted(root.glob("*_questions.json"))
    if not files:
        raise FileNotFoundError(f"no *_questions.json files found in {root}")
    questions: list[dict[str, Any]] = []
    for path in files:
        data = read_json(path)
        if not isinstance(data, list):
            raise ValueError(f"question file must contain a JSON list: {path}")
        questions.extend(_validate_question(item, path) for item in data)
    seen: set[str] = set()
    for item in questions:
        qid = item["qid"]
        if qid in seen:
            raise ValueError(f"duplicate qid: {qid}")
        seen.add(qid)
    return questions


def _validate_question(item: Any, source: Path) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"question item must be object in {source}")
    required = {"qid", "domain", "question", "options", "answer_format"}
    missing = required.difference(item)
    if missing:
        raise ValueError(f"question {item.get('qid')} missing fields {sorted(missing)} in {source}")
    if item["answer_format"] not in {"mcq", "multi", "tf"}:
        raise ValueError(f"unsupported answer_format for {item['qid']}: {item['answer_format']}")
    options = item["options"]
    if not isinstance(options, dict) or not options:
        raise ValueError(f"question {item['qid']} options must be object")
    return item

