from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.answer_engine import AnswerEngine
from agent.evidence import EvidenceBuilder
from agent.io_utils import append_jsonl, ensure_dir, read_jsonl, write_json
from agent.models import EvidenceBlock
from agent.output_writer import write_outputs
from agent.question_loader import load_questions
from agent.qwen_client import QwenClient
from agent.retrieval import SparseRetriever


def answer_questions(
    *,
    question_dir: str | Path,
    processed_dir: str | Path,
    logs_dir: str | Path,
    limit: int | None = None,
    qids: list[str] | None = None,
    replace_existing: bool = False,
    resume: bool = True,
    answer_csv: str | Path | None = None,
    evidence_json: str | Path | None = None,
) -> dict[str, int]:
    if (answer_csv is None) != (evidence_json is None):
        raise ValueError("answer_csv and evidence_json must be provided together")

    logs = ensure_dir(logs_dir)
    results_path = logs / "question_results.jsonl"
    errors_path = logs / "question_errors.jsonl"
    if results_path.exists() and not resume:
        results_path.unlink()
    if errors_path.exists():
        errors_path.unlink()

    questions = load_questions(question_dir)
    if qids:
        requested = set(qids)
        questions = [question for question in questions if question["qid"] in requested]
        found = {question["qid"] for question in questions}
        missing = sorted(requested.difference(found))
        if missing:
            raise ValueError(f"requested qids not found: {missing}")
    if limit is not None:
        questions = questions[:limit]
    question_ids = {question["qid"] for question in questions}
    existing_records = _load_existing_records(results_path, question_ids=question_ids) if resume else {}
    if replace_existing and qids:
        replace_set = set(qids)
        # Rewrite against every record on disk, not just the ones scoped to this
        # run's (possibly qid-filtered) question set -- otherwise rows for
        # questions outside `qids` silently vanish from the results file.
        all_existing_records = _load_existing_records(results_path, question_ids=None)
        all_existing_records = {qid: row for qid, row in all_existing_records.items() if qid not in replace_set}
        _rewrite_records(results_path, all_existing_records)
        existing_records = {qid: row for qid, row in existing_records.items() if qid not in replace_set}
    pending_questions = [question for question in questions if question["qid"] not in existing_records]
    completed_records = dict(existing_records)

    evidence_blocks = [EvidenceBlock.from_dict(row) for row in read_jsonl(Path(processed_dir) / "evidence_blocks.jsonl")]
    retriever = SparseRetriever.load(Path(processed_dir), evidence_blocks)
    evidence_builder = EvidenceBuilder(retriever)
    engine = AnswerEngine(QwenClient())

    answered = len(existing_records)
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in existing_records.values())
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in existing_records.values())
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in existing_records.values())
    errors: list[dict[str, Any]] = []
    progress = _AnswerProgress(total=len(questions), completed=answered)
    _write_summary(
        logs,
        answered=answered,
        skipped_existing=len(existing_records),
        pending=len(pending_questions),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        errors=errors,
    )
    _write_incremental_outputs(
        questions=questions,
        records=completed_records,
        answer_csv=answer_csv,
        evidence_json=evidence_json,
    )
    progress.render(current_qid="resume" if existing_records else "")

    for question in pending_questions:
        try:
            evidence = evidence_builder.build(question)
            result = engine.answer(question, evidence)
            record = result.to_log_record()
            append_jsonl(results_path, record)
            completed_records[record["qid"]] = record
            answered += 1
            prompt_tokens += result.usage.prompt_tokens
            completion_tokens += result.usage.completion_tokens
            total_tokens += result.usage.total_tokens
            _write_summary(
                logs,
                answered=answered,
                skipped_existing=len(existing_records),
                pending=len(questions) - answered,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                errors=errors,
            )
            _write_incremental_outputs(
                questions=questions,
                records=completed_records,
                answer_csv=answer_csv,
                evidence_json=evidence_json,
            )
            progress.update(completed=answered, current_qid=question["qid"], total_tokens=total_tokens)
        except Exception as exc:
            error = {"qid": question.get("qid"), "error": str(exc)}
            errors.append(error)
            append_jsonl(errors_path, error)
            _write_summary(
                logs,
                answered=answered,
                skipped_existing=len(existing_records),
                pending=len(questions) - answered,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                errors=errors,
            )
            progress.finish(error_qid=str(question.get("qid") or ""))
            raise RuntimeError(f"failed to answer {question.get('qid')}: {exc}") from exc

    progress.finish()
    return {
        "answered": answered,
        "skipped_existing": len(existing_records),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


class _AnswerProgress:
    def __init__(self, *, total: int, completed: int) -> None:
        self.total = max(total, 0)
        self.completed = min(max(completed, 0), self.total)
        self.started_at = time.time()
        self.enabled = sys.stderr.isatty()
        self.width = max(16, min(36, shutil.get_terminal_size((100, 20)).columns - 64))
        self.last_line_len = 0

    def render(self, *, current_qid: str = "", total_tokens: int = 0) -> None:
        if not self.enabled:
            return
        line = self._line(current_qid=current_qid, total_tokens=total_tokens)
        padding = " " * max(0, self.last_line_len - len(line))
        sys.stderr.write("\r" + line + padding)
        sys.stderr.flush()
        self.last_line_len = len(line)

    def update(self, *, completed: int, current_qid: str, total_tokens: int) -> None:
        self.completed = min(max(completed, 0), self.total)
        self.render(current_qid=current_qid, total_tokens=total_tokens)

    def finish(self, *, error_qid: str = "") -> None:
        if not self.enabled:
            return
        if error_qid:
            line = self._line(current_qid=error_qid, total_tokens=0, status="error")
        else:
            self.completed = self.total
            line = self._line(current_qid="done", total_tokens=0, status="done")
        padding = " " * max(0, self.last_line_len - len(line))
        sys.stderr.write("\r" + line + padding + "\n")
        sys.stderr.flush()
        self.last_line_len = 0

    def _line(self, *, current_qid: str, total_tokens: int, status: str = "running") -> str:
        if self.total <= 0:
            percent = 100.0
            filled = self.width
        else:
            percent = self.completed / self.total * 100
            filled = int(self.width * self.completed / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = int(time.time() - self.started_at)
        tokens = f" tokens={total_tokens}" if total_tokens else ""
        qid = f" qid={current_qid}" if current_qid else ""
        return f"[{bar}] {self.completed}/{self.total} {percent:5.1f}% elapsed={elapsed}s{tokens}{qid} {status}"


def _load_existing_records(results_path: Path, *, question_ids: set[str] | None) -> dict[str, dict[str, Any]]:
    if not results_path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(results_path):
        qid = str(row.get("qid") or "")
        if qid and (question_ids is None or qid in question_ids):
            records[qid] = row
    return records


def _rewrite_records(results_path: Path, records: dict[str, dict[str, Any]]) -> None:
    if not records:
        if results_path.exists():
            results_path.unlink()
        return
    from agent.io_utils import write_jsonl

    write_jsonl(results_path, records.values())


def _write_incremental_outputs(
    *,
    questions: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    answer_csv: str | Path | None,
    evidence_json: str | Path | None,
) -> None:
    if answer_csv is None or evidence_json is None:
        return
    ordered_records = [records[question["qid"]] for question in questions if question["qid"] in records]
    write_outputs(results=ordered_records, answer_csv=answer_csv, evidence_json=evidence_json)


def _write_summary(
    logs: Path,
    *,
    answered: int,
    skipped_existing: int,
    pending: int,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    errors: list[dict[str, Any]],
) -> None:
    write_json(
        logs / "run_summary.json",
        {
            "answered": answered,
            "skipped_existing": skipped_existing,
            "pending": pending,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "errors": errors,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve evidence and answer questions with Qwen.")
    parser.add_argument("--question-dir", default=str(ROOT / "public_dataset_upload" / "questions" / "group_a"))
    parser.add_argument("--processed-dir", default=str(ROOT / "processed_data"))
    parser.add_argument("--logs-dir", default=str(ROOT / "logs"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--qid", action="append", help="answer only this qid; may be provided multiple times")
    parser.add_argument("--replace-existing", action="store_true", help="rerun requested --qid rows and replace existing records")
    parser.add_argument("--fresh", action="store_true", help="ignore existing question_results.jsonl and rerun all questions")
    parser.add_argument("--answer-csv", help="refresh this CSV after each answered question; provide with --evidence-json")
    parser.add_argument("--evidence-json", help="refresh this JSON after each answered question; provide with --answer-csv")
    args = parser.parse_args()
    summary = answer_questions(
        question_dir=args.question_dir,
        processed_dir=args.processed_dir,
        logs_dir=args.logs_dir,
        limit=args.limit,
        qids=args.qid,
        replace_existing=args.replace_existing,
        resume=not args.fresh,
        answer_csv=args.answer_csv,
        evidence_json=args.evidence_json,
    )
    print(
        f"answered {summary['answered']} questions; "
        f"skipped_existing={summary['skipped_existing']}; "
        f"total_tokens={summary['total_tokens']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
