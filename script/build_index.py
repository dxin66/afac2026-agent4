from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.io_utils import read_jsonl, write_json
from agent.models import EvidenceBlock
from agent.retrieval import SparseRetriever


def build_index(processed_dir: str | Path) -> dict[str, int]:
    root = Path(processed_dir)
    evidence_blocks = [EvidenceBlock.from_dict(row) for row in read_jsonl(root / "evidence_blocks.jsonl")]
    retriever = SparseRetriever.build(evidence_blocks)
    retriever.save(root)
    summary = {
        "evidence_blocks": len(evidence_blocks),
        "article_entries": sum(len(rows) for rows in retriever.article_index.values()),
        "table_entries": len(retriever.table_index),
        "number_entries": len(retriever.number_index),
        "version": "ir_v1",
    }
    write_json(root / "index_summary.json", summary)
    return summary


def load_evidence_blocks(processed_dir: str | Path) -> list[EvidenceBlock]:
    return [EvidenceBlock.from_dict(row) for row in read_jsonl(Path(processed_dir) / "evidence_blocks.jsonl")]


def load_chunks(processed_dir: str | Path) -> list[EvidenceBlock]:
    return load_evidence_blocks(processed_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build structured evidence retrieval indexes.")
    parser.add_argument("--processed-dir", default=str(ROOT / "processed_data"))
    args = parser.parse_args()
    summary = build_index(args.processed_dir)
    print(
        "indexed "
        f"{summary['evidence_blocks']} evidence blocks, "
        f"{summary['table_entries']} table entries, "
        f"{summary['number_entries']} number entries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
