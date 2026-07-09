from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.io_utils import read_jsonl
from agent.output_writer import write_outputs
from agent.question_loader import load_questions


def export_answer(
    *,
    question_dir: str | Path,
    logs_dir: str | Path,
    answer_csv: str | Path,
    evidence_json: str | Path,
) -> dict[str, int]:
    results = read_jsonl(Path(logs_dir) / "question_results.jsonl")
    if not results:
        raise ValueError("no question results to export")
    official_qids = [question["qid"] for question in load_questions(question_dir)]
    result_by_qid = {str(row.get("qid") or ""): row for row in results}
    missing = [qid for qid in official_qids if qid not in result_by_qid]
    if missing:
        raise ValueError(f"missing official question results: {missing[:10]}")
    filtered = [result_by_qid[qid] for qid in official_qids]
    write_outputs(results=filtered, answer_csv=answer_csv, evidence_json=evidence_json)
    return {"rows": len(filtered)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export answer.csv and evidence.json.")
    parser.add_argument("--question-dir", default=str(ROOT / "public_dataset_upload" / "questions" / "group_a"))
    parser.add_argument("--logs-dir", default=str(ROOT / "logs"))
    parser.add_argument("--answer-csv", default=str(ROOT / "answer.csv"))
    parser.add_argument("--evidence-json", default=str(ROOT / "evidence.json"))
    args = parser.parse_args()
    summary = export_answer(
        question_dir=args.question_dir,
        logs_dir=args.logs_dir,
        answer_csv=args.answer_csv,
        evidence_json=args.evidence_json,
    )
    print(f"exported {summary['rows']} answers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
