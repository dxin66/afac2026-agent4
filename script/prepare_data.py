from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.document_processor import build_parse_artifacts
from agent.io_utils import ensure_dir, write_json, write_jsonl


def prepare_data(data_dir: str | Path, output_dir: str | Path) -> dict[str, int]:
    output = ensure_dir(output_dir)
    artifacts = build_parse_artifacts(data_dir)
    documents = artifacts["documents"]
    pages = artifacts["pages"]
    evidence_blocks = artifacts["evidence_blocks"]
    document_markdown = artifacts["document_markdown"]
    write_jsonl(output / "documents.jsonl", (doc.to_dict() for doc in documents))
    write_jsonl(output / "page_ir.jsonl", (page.to_dict() for page in pages))
    write_jsonl(output / "evidence_blocks.jsonl", (block.to_dict() for block in evidence_blocks))
    write_jsonl(output / "document_markdown.jsonl", document_markdown)
    write_json(
        output / "doc_map.json",
        {
            doc.doc_id: {
                "domain": doc.domain,
                "title": doc.title,
                "source_path": doc.source_path,
                "pages": doc.page_count,
                "file_type": doc.source_type,
            }
            for doc in documents
        },
    )
    write_json(output / "parse_summary.json", artifacts["summary"])
    write_json(output / "prepare_summary.json", artifacts["summary"])
    return {"documents": len(documents), "pages": len(pages), "evidence_blocks": len(evidence_blocks)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract and normalize raw financial documents.")
    parser.add_argument("--data-dir", default=str(ROOT / "public_dataset_upload"))
    parser.add_argument("--output-dir", default=str(ROOT / "processed_data"))
    args = parser.parse_args()
    summary = prepare_data(args.data_dir, args.output_dir)
    print(f"prepared {summary['documents']} documents, {summary['pages']} pages, {summary['evidence_blocks']} evidence blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
