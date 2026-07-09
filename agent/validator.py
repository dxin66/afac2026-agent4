from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from agent.answer_utils import normalize_answer
from agent.io_utils import read_json


def validate_answer_csv(
    *,
    questions: list[dict[str, Any]],
    answer_csv: str | Path,
    evidence_json: str | Path | None = None,
) -> None:
    path = Path(answer_csv)
    if not path.exists():
        raise FileNotFoundError(f"answer.csv not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("answer.csv has no rows")
    if rows[0].get("qid") != "summary":
        raise ValueError("first answer.csv data row must be qid=summary")

    question_by_qid = {item["qid"]: item for item in questions}
    answer_rows = rows[1:]
    row_by_qid = {row["qid"]: row for row in answer_rows}
    if set(row_by_qid) != set(question_by_qid):
        missing = sorted(set(question_by_qid).difference(row_by_qid))
        extra = sorted(set(row_by_qid).difference(question_by_qid))
        raise ValueError(f"answer qid mismatch; missing={missing[:10]} extra={extra[:10]}")

    prompt_sum = 0
    completion_sum = 0
    total_sum = 0
    for qid, row in row_by_qid.items():
        question = question_by_qid[qid]
        normalized = normalize_answer(row.get("answer"), question["answer_format"], question["options"])
        if normalized != row.get("answer"):
            raise ValueError(f"answer for {qid} is not normalized: {row.get('answer')} != {normalized}")
        prompt = int(row.get("prompt_tokens") or 0)
        completion = int(row.get("completion_tokens") or 0)
        total = int(row.get("total_tokens") or 0)
        if total != prompt + completion:
            raise ValueError(f"token mismatch for {qid}: total != prompt + completion")
        prompt_sum += prompt
        completion_sum += completion
        total_sum += total

    summary = rows[0]
    if int(summary.get("prompt_tokens") or 0) != prompt_sum:
        raise ValueError("summary.prompt_tokens does not equal per-question sum")
    if int(summary.get("completion_tokens") or 0) != completion_sum:
        raise ValueError("summary.completion_tokens does not equal per-question sum")
    if int(summary.get("total_tokens") or 0) != total_sum:
        raise ValueError("summary.total_tokens does not equal per-question sum")

    if evidence_json is not None:
        evidence = read_json(evidence_json)
        if set(evidence) != set(question_by_qid):
            raise ValueError("evidence.json qids do not match questions")
        for qid, row in evidence.items():
            payload = row.get("evidence") if isinstance(row, dict) else None
            if not isinstance(payload, dict):
                raise ValueError(f"evidence for {qid} must be an object")
            items = payload.get("items")
            if not isinstance(items, list) or not items:
                raise ValueError(f"evidence for {qid} has no items")
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    raise ValueError(f"evidence item {index} for {qid} must be an object")
                if not str(item.get("chunk_id") or "").strip():
                    raise ValueError(f"evidence item {index} for {qid} missing chunk_id")
                if not str(item.get("text") or "").strip():
                    raise ValueError(f"evidence item {index} for {qid} missing text")
            _validate_v6_matrix_ids(qid, payload)


def _validate_v6_matrix_ids(qid: str, evidence: dict[str, Any]) -> None:
    matrix = evidence.get("classified_evidence_matrix") or evidence.get("verified_evidence_matrix")
    if not isinstance(matrix, dict):
        return
    for option, by_doc in matrix.items():
        if not isinstance(by_doc, dict):
            raise ValueError(f"V6 matrix option {option} for {qid} must be an object")
        for doc_id, bucket in by_doc.items():
            if not isinstance(bucket, dict):
                raise ValueError(f"V6 matrix bucket {option}/{doc_id} for {qid} must be an object")
            candidate_ids = {
                str(item.get("evidence_id") or "")
                for item in bucket.get("candidates") or []
                if isinstance(item, dict)
            }
            for key in ["direct_support", "direct_refute", "derived_support", "derived_refute", "support", "refute", "neutral"]:
                rows = bucket.get(key) or []
                if not isinstance(rows, list):
                    raise ValueError(f"V6 matrix {key} for {qid}/{option}/{doc_id} must be a list")
                for item in rows:
                    if not isinstance(item, dict):
                        raise ValueError(f"V6 matrix {key} row for {qid}/{option}/{doc_id} must be an object")
                    evidence_id = str(item.get("evidence_id") or "")
                    if evidence_id and evidence_id not in candidate_ids:
                        raise ValueError(f"V6 matrix {key} row for {qid}/{option}/{doc_id} has unknown evidence_id: {evidence_id}")
