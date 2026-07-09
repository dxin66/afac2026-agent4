from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.evidence import EvidenceBuilder
from agent.io_utils import read_jsonl, write_json
from agent.models import EvidenceBlock
from agent.question_loader import load_questions
from agent.retrieval import SparseRetriever


REGRESSION_CASES = {
    "ins_a_001": [
        "身故保险金",
        "现金价值",
        "保单账户价值",
    ],
    "ins_a_012": [
        "宽限期",
        "效力中止",
        "保险责任",
    ],
    "fin_a_001": [
        "营业收入",
        "研发投入",
        "经营活动产生的现金流量净额",
        "归属于上市公司股东的净利润",
    ],
    "fc_a_014": [
        "违约金",
        "本金和利息",
        "资产减值补偿",
        "兑付日",
    ],
    "res_a_005": [
        "服务消费",
        "银保渠道",
        "保费贡献",
        "居民可支配收入增长",
    ],
    "res_a_012": [
        "芯片定制服务",
        "内置检测规则",
        "解析规则",
    ],
    "res_a_016": [
        "服务零售",
        "商品零售",
        "居民可支配收入",
        "手续费及佣金净收入",
    ],
}


REGRESSION_QUESTIONS: dict[str, dict[str, Any]] = {}


def evaluate_evidence_coverage(
    *,
    question_dir: str | Path,
    processed_dir: str | Path,
    logs_dir: str | Path,
    qids: list[str] | None = None,
) -> dict[str, Any]:
    questions = load_questions(question_dir)
    question_by_qid = {question["qid"]: question for question in questions}
    question_by_qid.update(REGRESSION_QUESTIONS)
    target_qids = qids or list(REGRESSION_CASES)
    missing_qids = [qid for qid in target_qids if qid not in question_by_qid]
    if missing_qids:
        raise ValueError(f"coverage qids not found: {missing_qids}")

    evidence_blocks = [EvidenceBlock.from_dict(row) for row in read_jsonl(Path(processed_dir) / "evidence_blocks.jsonl")]
    retriever = SparseRetriever.load(Path(processed_dir), evidence_blocks)
    builder = EvidenceBuilder(retriever)

    rows: list[dict[str, Any]] = []
    for qid in target_qids:
        question = question_by_qid[qid]
        evidence = builder.build(question)
        evidence_text = "\n".join(str(item.get("text") or "") for item in evidence["items"])
        required_terms = REGRESSION_CASES.get(qid, [])
        missing_terms = [term for term in required_terms if term not in evidence_text]
        expected_docs = [str(doc_id) for doc_id in question.get("doc_ids") or []]
        evidence_docs = sorted({str(item.get("doc_id") or "") for item in evidence["items"]})
        missing_docs = [doc_id for doc_id in expected_docs if doc_id not in evidence_docs]
        rows.append(
            {
                "qid": qid,
                "required_terms": required_terms,
                "missing_terms": missing_terms,
                "expected_docs": expected_docs,
                "evidence_docs": evidence_docs,
                "missing_docs": missing_docs,
                "items": len(evidence["items"]),
                "passed": not missing_terms and not missing_docs,
            }
        )

    summary = {
        "metric": "evidence_required_term_coverage",
        "total": len(rows),
        "passed": sum(1 for row in rows if row["passed"]),
        "failed": [row for row in rows if not row["passed"]],
        "details": rows,
    }
    logs = Path(logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    write_json(logs / "evidence_coverage_summary.json", summary)
    if summary["failed"]:
        raise ValueError(f"evidence coverage failed for {[row['qid'] for row in summary['failed']]}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate evidence coverage for known hard questions.")
    parser.add_argument("--question-dir", default=str(ROOT / "public_dataset_upload" / "questions" / "group_a"))
    parser.add_argument("--processed-dir", default=str(ROOT / "processed_data"))
    parser.add_argument("--logs-dir", default=str(ROOT / "logs"))
    parser.add_argument("--qid", action="append", help="evaluate only this qid; may be provided multiple times")
    args = parser.parse_args()
    summary = evaluate_evidence_coverage(
        question_dir=args.question_dir,
        processed_dir=args.processed_dir,
        logs_dir=args.logs_dir,
        qids=args.qid,
    )
    print(f"evidence_coverage passed={summary['passed']}/{summary['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
