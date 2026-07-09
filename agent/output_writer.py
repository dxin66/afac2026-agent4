from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from agent.io_utils import write_json


CSV_FIELDS = ["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"]


def write_outputs(
    *,
    results: list[dict[str, Any]],
    answer_csv: str | Path,
    evidence_json: str | Path,
) -> None:
    result_by_qid = {row["qid"]: row for row in results}
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in results)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in results)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in results)

    answer_path = Path(answer_csv)
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    with answer_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "qid": "summary",
                "answer": "",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        )
        for qid in result_by_qid:
            row = result_by_qid[qid]
            writer.writerow(
                {
                    "qid": qid,
                    "answer": row["answer"],
                    "prompt_tokens": int(row.get("prompt_tokens") or 0),
                    "completion_tokens": int(row.get("completion_tokens") or 0),
                    "total_tokens": int(row.get("total_tokens") or 0),
                }
            )

    evidence = {
        row["qid"]: {
            "answer": row["answer"],
            "usage": {
                "prompt_tokens": int(row.get("prompt_tokens") or 0),
                "completion_tokens": int(row.get("completion_tokens") or 0),
                "total_tokens": int(row.get("total_tokens") or 0),
            },
            "evidence": row.get("evidence"),
            "model_output": row.get("model_output"),
            "raw_model_content": row.get("raw_model_content"),
        }
        for row in results
    }
    write_json(evidence_json, evidence)

