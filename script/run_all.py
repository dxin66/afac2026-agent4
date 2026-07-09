from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script.answer_questions import answer_questions
from script.build_index import build_index
from script.evaluate_evidence_coverage import evaluate_evidence_coverage
from script.export_answer import export_answer
from script.prepare_data import prepare_data
from script.validate_outputs import validate_outputs


def run_all(
    *,
    data_dir: str | Path,
    question_dir: str | Path,
    processed_dir: str | Path,
    logs_dir: str | Path,
    answer_csv: str | Path,
    evidence_json: str | Path,
    limit: int | None = None,
    resume: bool = True,
) -> None:
    prepare_data(data_dir=data_dir, output_dir=processed_dir)
    build_index(processed_dir=processed_dir)
    evaluate_evidence_coverage(question_dir=question_dir, processed_dir=processed_dir, logs_dir=logs_dir)
    answer_questions(
        question_dir=question_dir,
        processed_dir=processed_dir,
        logs_dir=logs_dir,
        limit=limit,
        resume=resume,
        answer_csv=answer_csv,
        evidence_json=evidence_json,
    )
    export_answer(question_dir=question_dir, logs_dir=logs_dir, answer_csv=answer_csv, evidence_json=evidence_json)
    validate_outputs(question_dir=question_dir, answer_csv=answer_csv, evidence_json=evidence_json)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full financial long-text QA pipeline.")
    parser.add_argument("--data-dir", default=str(ROOT / "public_dataset_upload"))
    parser.add_argument("--question-dir", default=str(ROOT / "public_dataset_upload" / "questions" / "group_a"))
    parser.add_argument("--processed-dir", default=str(ROOT / "processed_data"))
    parser.add_argument("--logs-dir", default=str(ROOT / "logs"))
    parser.add_argument("--answer-csv", default=str(ROOT / "answer.csv"))
    parser.add_argument("--evidence-json", default=str(ROOT / "evidence.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fresh", action="store_true", help="ignore existing question_results.jsonl and rerun all questions")
    args = parser.parse_args()
    run_all(
        data_dir=args.data_dir,
        question_dir=args.question_dir,
        processed_dir=args.processed_dir,
        logs_dir=args.logs_dir,
        answer_csv=args.answer_csv,
        evidence_json=args.evidence_json,
        limit=args.limit,
        resume=not args.fresh,
    )
    print("pipeline completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
