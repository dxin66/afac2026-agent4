from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.io_utils import ensure_dir, read_json, read_jsonl, write_json


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


def evaluate_llm_parse_validation(
    *,
    processed_dir: str | Path,
    logs_dir: str | Path,
    rule_summary_path: str | Path | None = None,
    llm_judgments_path: str | Path | None = None,
    sample_limit: int = 100,
    input_price_per_1m: float = 0.0,
    output_price_per_1m: float = 0.0,
    expected_output_tokens: int = 128,
) -> dict[str, Any]:
    logs_root = Path(logs_dir)
    rule_summary_file = Path(rule_summary_path) if rule_summary_path else logs_root / "parse_rule_validation_summary.json"
    rule_summary = read_json(rule_summary_file)
    markdown_by_page = _load_markdown(Path(processed_dir))
    exception_items = _rule_exception_items(rule_summary)[:sample_limit]
    prompt_lengths = [_prompt_token_estimate(markdown_by_page.get(_page_key(item), ""), item) for item in exception_items]
    judgment_summary = _judgment_summary(llm_judgments_path)
    estimated_prompt_tokens = sum(prompt_lengths)
    estimated_completion_tokens = len(exception_items) * expected_output_tokens
    summary = {
        "llm_calls_made": 0,
        "default_chain_enabled": False,
        "sampled_rule_exception_pages": len(exception_items),
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "average_prompt_tokens": round(estimated_prompt_tokens / max(len(exception_items), 1), 2),
        "estimated_completion_tokens": estimated_completion_tokens,
        "estimated_cost": round(
            estimated_prompt_tokens * input_price_per_1m / 1_000_000
            + estimated_completion_tokens * output_price_per_1m / 1_000_000,
            6,
        ),
        "rule_exception_counts": rule_summary.get("rule_exception_counts") or {},
        "judgment_comparison": judgment_summary,
        "samples": exception_items[:20],
    }
    out_dir = ensure_dir(logs_dir)
    write_json(out_dir / "llm_parse_validation_feasibility.json", summary)
    return summary


def _load_markdown(processed_dir: Path) -> dict[str, str]:
    path = processed_dir / "document_markdown.jsonl"
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {_page_key(row): str(row.get("markdown") or "") for row in rows}


def _rule_exception_items(rule_summary: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    samples_by_rule = rule_summary.get("rule_exception_samples") or {}
    for rule in RULE_BUCKETS:
        for sample in samples_by_rule.get(rule) or []:
            item = dict(sample)
            item["rule"] = rule
            items.append(item)
    seen: set[tuple[str, int, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = (str(item.get("doc_id") or ""), int(item.get("page_no") or 0), str(item.get("rule") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _prompt_token_estimate(markdown: str, item: dict[str, Any]) -> int:
    instruction = (
        "判断该页是否存在规则指出的解析异常，输出 JSON，包含 issue_present、reason、evidence。"
    )
    payload = f"{instruction}\n规则: {item.get('rule')}\n页内容:\n{markdown[:6000]}"
    return max(1, len(payload) // 4)


def _judgment_summary(llm_judgments_path: str | Path | None) -> dict[str, Any]:
    if not llm_judgments_path:
        return {
            "provided": False,
            "consistent": 0,
            "conflicting": 0,
            "unknown": 0,
            "consistency_ratio": None,
            "conflict_ratio": None,
        }
    rows = read_jsonl(llm_judgments_path)
    consistent = 0
    conflicting = 0
    unknown = 0
    for row in rows:
        rule_issue = bool(row.get("rule_issue_present", True))
        llm_issue = row.get("llm_issue_present")
        if llm_issue is None:
            unknown += 1
        elif bool(llm_issue) == rule_issue:
            consistent += 1
        else:
            conflicting += 1
    comparable = consistent + conflicting
    return {
        "provided": True,
        "rows": len(rows),
        "consistent": consistent,
        "conflicting": conflicting,
        "unknown": unknown,
        "consistency_ratio": round(consistent / comparable, 6) if comparable else None,
        "conflict_ratio": round(conflicting / comparable, 6) if comparable else None,
    }


def _page_key(row: dict[str, Any]) -> str:
    return f"{row.get('doc_id')}:{int(row.get('page_no') or 0)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate LLM parse-validation feasibility on deterministic rule exceptions.")
    parser.add_argument("--processed-dir", default=str(ROOT / "processed_data"))
    parser.add_argument("--logs-dir", default=str(ROOT / "logs"))
    parser.add_argument("--rule-summary-path")
    parser.add_argument("--llm-judgments-path")
    parser.add_argument("--sample-limit", type=int, default=100)
    parser.add_argument("--input-price-per-1m", type=float, default=0.0)
    parser.add_argument("--output-price-per-1m", type=float, default=0.0)
    parser.add_argument("--expected-output-tokens", type=int, default=128)
    args = parser.parse_args()
    summary = evaluate_llm_parse_validation(
        processed_dir=args.processed_dir,
        logs_dir=args.logs_dir,
        rule_summary_path=args.rule_summary_path,
        llm_judgments_path=args.llm_judgments_path,
        sample_limit=args.sample_limit,
        input_price_per_1m=args.input_price_per_1m,
        output_price_per_1m=args.output_price_per_1m,
        expected_output_tokens=args.expected_output_tokens,
    )
    print(
        "LLM parse-validation feasibility estimated: "
        f"{summary['sampled_rule_exception_pages']} sampled rule exceptions, "
        f"{summary['estimated_prompt_tokens']} prompt tokens"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
