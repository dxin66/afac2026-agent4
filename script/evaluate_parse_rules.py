from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.document_processor import _parse_quality_report
from agent.io_utils import ensure_dir, read_jsonl, write_json
from agent.models import PageIR, ParsedDocument


RULE_BUCKETS = (
    "empty_pages",
    "mojibake_pages",
    "pages_without_section",
    "table_shape_warnings",
    "table_column_inconsistency",
    "generic_header_tables",
    "long_cell_tables",
    "critical_financial_tables_missing",
    "amount_truncation_tables",
)


def evaluate_parse_rules(processed_dir: str | Path, logs_dir: str | Path) -> dict[str, Any]:
    root = Path(processed_dir)
    documents = [ParsedDocument.from_dict(row) for row in read_jsonl(root / "documents.jsonl")]
    pages = [PageIR.from_dict(row) for row in read_jsonl(root / "page_ir.jsonl")]
    quality = _parse_quality_report(documents, pages)
    summary = {
        "documents": len(documents),
        "pages": len(pages),
        "rule_exception_counts": {
            bucket: int((quality.get(bucket) or {}).get("count") or 0)
            for bucket in RULE_BUCKETS
        },
        "rule_exception_samples": {
            bucket: list((quality.get(bucket) or {}).get("samples") or [])
            for bucket in RULE_BUCKETS
        },
        "text_metrics": quality.get("text_metrics") or {},
        "table_metrics": quality.get("table_metrics") or {},
        "document_metrics": quality.get("document_metrics") or {},
    }
    out_dir = ensure_dir(logs_dir)
    write_json(out_dir / "parse_rule_validation_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate deterministic parse-quality rule exceptions.")
    parser.add_argument("--processed-dir", default=str(ROOT / "processed_data"))
    parser.add_argument("--logs-dir", default=str(ROOT / "logs"))
    args = parser.parse_args()
    summary = evaluate_parse_rules(args.processed_dir, args.logs_dir)
    print(
        "parse rules evaluated: "
        f"{summary['pages']} pages, "
        f"{sum(summary['rule_exception_counts'].values())} rule exceptions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
