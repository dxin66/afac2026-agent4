from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.question_loader import load_questions
from agent.validator import validate_answer_csv


def validate_outputs(
    *,
    question_dir: str | Path,
    answer_csv: str | Path,
    evidence_json: str | Path,
) -> None:
    questions = load_questions(question_dir)
    validate_answer_csv(questions=questions, answer_csv=answer_csv, evidence_json=evidence_json)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate answer.csv and evidence.json.")
    parser.add_argument("--question-dir", default=str(ROOT / "public_dataset_upload" / "questions" / "group_a"))
    parser.add_argument("--answer-csv", default=str(ROOT / "answer.csv"))
    parser.add_argument("--evidence-json", default=str(ROOT / "evidence.json"))
    args = parser.parse_args()
    validate_outputs(question_dir=args.question_dir, answer_csv=args.answer_csv, evidence_json=args.evidence_json)
    print("outputs are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
