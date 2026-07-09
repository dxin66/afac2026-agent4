from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.question_loader import load_questions


def compare_answers(
    base_csv: str | Path,
    candidate_csv: str | Path,
    *,
    question_dir: str | Path | None = None,
) -> dict[str, Any]:
    base = _load_answers(base_csv)
    candidate = _load_answers(candidate_csv)
    if question_dir is not None:
        official_qids = [question["qid"] for question in load_questions(question_dir)]
        official = set(official_qids)
        base = {qid: answer for qid, answer in base.items() if qid in official}
        candidate = {qid: answer for qid, answer in candidate.items() if qid in official}
    missing = sorted(set(base).difference(candidate))
    extra = sorted(set(candidate).difference(base))
    changed = [
        {"qid": qid, "base": base[qid], "candidate": candidate[qid]}
        for qid in sorted(set(base).intersection(candidate))
        if base[qid] != candidate[qid]
    ]
    return {
        "base_rows": len(base),
        "candidate_rows": len(candidate),
        "missing": missing,
        "extra": extra,
        "changed": changed,
    }


def _load_answers(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        return {
            str(row.get("qid") or ""): str(row.get("answer") or "")
            for row in rows
            if row.get("qid") and row.get("qid") != "summary"
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare qid->answer changes between two answer.csv files.")
    parser.add_argument("base_csv")
    parser.add_argument("candidate_csv")
    parser.add_argument("--question-dir", default=str(ROOT / "public_dataset_upload" / "questions" / "group_a"))
    args = parser.parse_args()
    result = compare_answers(args.base_csv, args.candidate_csv, question_dir=args.question_dir)
    print(f"base_rows={result['base_rows']} candidate_rows={result['candidate_rows']}")
    print(f"missing={len(result['missing'])} extra={len(result['extra'])} changed={len(result['changed'])}")
    for row in result["changed"]:
        print(f"{row['qid']}: {row['base']} -> {row['candidate']}")
    if result["missing"]:
        print("missing_qids=" + ",".join(result["missing"]))
    if result["extra"]:
        print("extra_qids=" + ",".join(result["extra"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
