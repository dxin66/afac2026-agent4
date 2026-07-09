from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.document_processor import (
    CRITICAL_FINANCIAL_TABLE_MARKERS,
    NUMBER_RE,
    YEAR_RE,
    _extract_pymupdf_tables,
    _open_pdfplumber,
    _pdfplumber_doc_page_tables,
    _table_text,
)
from agent.io_utils import ensure_dir, write_json


def evaluate_tabula_tables(data_dir: str | Path, logs_dir: str | Path, sample_limit: int = 20) -> dict[str, Any]:
    samples = _select_financial_report_pages(Path(data_dir), sample_limit)
    tabula_available, tabula_error = _tabula_status()
    rows: list[dict[str, Any]] = []
    for sample in samples:
        rows.append(_evaluate_sample(sample, tabula_available=tabula_available))
    summary = {
        "adapter_only": True,
        "tabula_available": tabula_available,
        "tabula_error": tabula_error,
        "sample_pages": len(samples),
        "samples": rows,
        "aggregate": _aggregate(rows),
    }
    out_dir = ensure_dir(logs_dir)
    write_json(out_dir / "tabula_table_eval_summary.json", summary)
    return summary


def _select_financial_report_pages(data_dir: Path, sample_limit: int) -> list[dict[str, Any]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for tabula sample selection.") from exc
    report_dir = data_dir / "raw" / "financial_reports"
    samples: list[dict[str, Any]] = []
    for path in sorted([*report_dir.glob("*.pdf"), *report_dir.glob("*.PDF")]):
        with fitz.open(path) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                text = page.get_text("text") or ""
                markers = [marker for marker in CRITICAL_FINANCIAL_TABLE_MARKERS if marker in text]
                if not markers:
                    continue
                samples.append({"path": str(path), "doc_id": path.stem, "page_no": page_index, "markers": markers[:6]})
                if len(samples) >= sample_limit:
                    return samples
    return samples


def _tabula_status() -> tuple[bool, str]:
    try:
        import tabula  # noqa: F401
    except Exception as exc:
        return False, str(exc)
    return True, ""


def _evaluate_sample(sample: dict[str, Any], *, tabula_available: bool) -> dict[str, Any]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for tabula evaluation.") from exc
    path = Path(sample["path"])
    page_no = int(sample["page_no"])
    with fitz.open(path) as pdf:
        page = pdf[page_no - 1]
        started = time.perf_counter()
        pymupdf_tables = _extract_pymupdf_tables(page=page, page_no=page_no, doc_id=sample["doc_id"], section_state=[])
        pymupdf_elapsed = time.perf_counter() - started
    plumber_pdf = _open_pdfplumber(path)
    try:
        started = time.perf_counter()
        pdfplumber_tables = _pdfplumber_doc_page_tables(pdf=plumber_pdf, page_no=page_no, doc_id=sample["doc_id"], section_state=[]) if plumber_pdf else []
        pdfplumber_elapsed = time.perf_counter() - started
    finally:
        if plumber_pdf is not None:
            plumber_pdf.close()
    tabula_tables: list[Any] = []
    tabula_elapsed = 0.0
    tabula_error = ""
    if tabula_available:
        try:
            import tabula

            started = time.perf_counter()
            tabula_tables = tabula.read_pdf(str(path), pages=page_no, multiple_tables=True, lattice=True)
            tabula_elapsed = time.perf_counter() - started
        except Exception as exc:
            tabula_error = str(exc)
    return {
        **sample,
        "pymupdf": _parser_metrics([_table_text(table) for table in pymupdf_tables], pymupdf_elapsed),
        "pdfplumber": _parser_metrics([_table_text(table) for table in pdfplumber_tables], pdfplumber_elapsed),
        "tabula": _parser_metrics([str(table) for table in tabula_tables], tabula_elapsed),
        "tabula_error": tabula_error,
    }


def _parser_metrics(table_texts: list[str], elapsed_seconds: float) -> dict[str, Any]:
    combined = "\n".join(table_texts)
    return {
        "tables": len(table_texts),
        "elapsed_seconds": round(elapsed_seconds, 4),
        "header_hit": any(marker in combined for marker in ("项目", "指标", "主要会计数据", "资产负债表", "利润表", "现金流量表")),
        "year_columns": sorted(set(YEAR_RE.findall(combined))),
        "amount_cells": len(NUMBER_RE.findall(combined)),
        "cjk_chars": sum(1 for char in combined if "\u4e00" <= char <= "\u9fff"),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for parser_name in ("pymupdf", "pdfplumber", "tabula"):
        parser_rows = [row[parser_name] for row in rows]
        out[parser_name] = {
            "tables": sum(int(row["tables"]) for row in parser_rows),
            "header_hit_pages": sum(1 for row in parser_rows if row["header_hit"]),
            "amount_cells": sum(int(row["amount_cells"]) for row in parser_rows),
            "cjk_chars": sum(int(row["cjk_chars"]) for row in parser_rows),
            "elapsed_seconds": round(sum(float(row["elapsed_seconds"]) for row in parser_rows), 4),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Adapter-only tabula table extraction comparison for selected financial-report pages.")
    parser.add_argument("--data-dir", default=str(ROOT / "public_dataset_upload"))
    parser.add_argument("--logs-dir", default=str(ROOT / "logs"))
    parser.add_argument("--sample-limit", type=int, default=20)
    args = parser.parse_args()
    summary = evaluate_tabula_tables(args.data_dir, args.logs_dir, sample_limit=args.sample_limit)
    print(
        "tabula table evaluation completed: "
        f"{summary['sample_pages']} sample pages, "
        f"tabula_available={summary['tabula_available']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
