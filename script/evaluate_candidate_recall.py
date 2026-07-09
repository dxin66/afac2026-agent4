from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.evidence import EvidenceBuilder
from agent.io_utils import ensure_dir, read_jsonl, write_json
from agent.models import EvidenceBlock
from agent.question_loader import load_questions
from agent.retrieval import SparseRetriever


def evaluate_candidate_recall(
    *,
    question_dir: str | Path,
    processed_dir: str | Path,
    logs_dir: str | Path,
    top_docs: int = 20,
    candidate_chunk_hits: int = 600,
) -> dict[str, Any]:
    questions = [q for q in load_questions(question_dir) if q.get("doc_ids")]
    evidence_blocks = [EvidenceBlock.from_dict(row) for row in read_jsonl(Path(processed_dir) / "evidence_blocks.jsonl")]
    retriever = SparseRetriever.load(Path(processed_dir), evidence_blocks)
    builder = EvidenceBuilder(
        retriever,
        top_docs_without_doc_ids=top_docs,
        candidate_chunk_hits=candidate_chunk_hits,
    )

    rows: list[dict[str, Any]] = []
    per_domain: dict[str, dict[str, int]] = {}
    for question in questions:
        simulated = dict(question)
        truth = [str(doc_id) for doc_id in question.get("doc_ids") or []]
        simulated.pop("doc_ids", None)
        predicted = builder.infer_doc_ids(simulated)
        hit_docs = [doc_id for doc_id in truth if doc_id in predicted]
        exact_all = len(hit_docs) == len(truth)
        any_hit = bool(hit_docs)
        domain = question["domain"]
        stats = per_domain.setdefault(domain, {"total": 0, "all_hit": 0, "any_hit": 0, "truth_docs": 0, "hit_docs": 0})
        stats["total"] += 1
        stats["all_hit"] += int(exact_all)
        stats["any_hit"] += int(any_hit)
        stats["truth_docs"] += len(truth)
        stats["hit_docs"] += len(hit_docs)
        rows.append(
            {
                "qid": question["qid"],
                "domain": domain,
                "truth_doc_ids": "|".join(truth),
                "predicted_doc_ids": "|".join(predicted),
                "hit_doc_ids": "|".join(hit_docs),
                "all_docs_recalled": exact_all,
                "any_doc_recalled": any_hit,
                "doc_recall": len(hit_docs) / len(truth) if truth else 0.0,
            }
        )

    total = len(rows)
    all_hit = sum(1 for row in rows if row["all_docs_recalled"])
    any_hit = sum(1 for row in rows if row["any_doc_recalled"])
    truth_docs = sum(len(str(row["truth_doc_ids"]).split("|")) for row in rows if row["truth_doc_ids"])
    hit_docs = sum(len(str(row["hit_doc_ids"]).split("|")) for row in rows if row["hit_doc_ids"])
    summary = {
        "metric": "simulate_no_doc_ids_candidate_recall_on_public_a",
        "top_docs": top_docs,
        "candidate_chunk_hits": candidate_chunk_hits,
        "total_questions": total,
        "all_docs_recalled": all_hit,
        "all_docs_recall_rate": all_hit / total if total else 0.0,
        "any_doc_recalled": any_hit,
        "any_doc_recall_rate": any_hit / total if total else 0.0,
        "doc_level_recall": hit_docs / truth_docs if truth_docs else 0.0,
        "per_domain": {
            domain: {
                **stats,
                "all_docs_recall_rate": stats["all_hit"] / stats["total"] if stats["total"] else 0.0,
                "any_doc_recall_rate": stats["any_hit"] / stats["total"] if stats["total"] else 0.0,
                "doc_level_recall": stats["hit_docs"] / stats["truth_docs"] if stats["truth_docs"] else 0.0,
            }
            for domain, stats in sorted(per_domain.items())
        },
        "misses": [row for row in rows if not row["all_docs_recalled"]],
    }

    logs = ensure_dir(logs_dir)
    write_json(logs / "candidate_recall_summary.json", summary)
    csv_path = logs / "candidate_recall_details.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate B-style candidate document recall on public A labels.")
    parser.add_argument("--question-dir", default=str(ROOT / "public_dataset_upload" / "questions" / "group_a"))
    parser.add_argument("--processed-dir", default=str(ROOT / "processed_data"))
    parser.add_argument("--logs-dir", default=str(ROOT / "logs"))
    parser.add_argument("--top-docs", type=int, default=20)
    parser.add_argument("--candidate-chunk-hits", type=int, default=600)
    args = parser.parse_args()
    summary = evaluate_candidate_recall(
        question_dir=args.question_dir,
        processed_dir=args.processed_dir,
        logs_dir=args.logs_dir,
        top_docs=args.top_docs,
        candidate_chunk_hits=args.candidate_chunk_hits,
    )
    print(
        "candidate_recall "
        f"all={summary['all_docs_recalled']}/{summary['total_questions']} "
        f"rate={summary['all_docs_recall_rate']:.4f} "
        f"doc_level={summary['doc_level_recall']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
