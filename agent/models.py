from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParsedDocument:
    doc_id: str
    domain: str
    title: str
    source_path: str
    source_type: str
    page_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParsedDocument":
        return cls(
            doc_id=str(data["doc_id"]),
            domain=str(data["domain"]),
            title=str(data.get("title") or data["doc_id"]),
            source_path=str(data.get("source_path") or ""),
            source_type=str(data.get("source_type") or data.get("file_type") or ""),
            page_count=int(data.get("page_count") or data.get("pages") or 0),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class TextSpan:
    text: str
    bbox: list[float] = field(default_factory=list)
    font: str = ""
    size: float = 0.0
    flags: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextSpan":
        return cls(
            text=str(data.get("text") or ""),
            bbox=[float(item) for item in data.get("bbox") or []],
            font=str(data.get("font") or ""),
            size=float(data.get("size") or 0.0),
            flags=int(data.get("flags") or 0),
        )


@dataclass(frozen=True)
class TextBlock:
    block_id: str
    page_no: int
    bbox: list[float]
    text: str
    spans: list[TextSpan] = field(default_factory=list)
    block_role: str = "text"
    section_path: list[str] = field(default_factory=list)
    article_no: str = ""
    font_stats: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextBlock":
        return cls(
            block_id=str(data["block_id"]),
            page_no=int(data.get("page_no") or 1),
            bbox=[float(item) for item in data.get("bbox") or []],
            text=str(data.get("text") or ""),
            spans=[TextSpan.from_dict(item) for item in data.get("spans") or []],
            block_role=str(data.get("block_role") or "text"),
            section_path=[str(item) for item in data.get("section_path") or []],
            article_no=str(data.get("article_no") or ""),
            font_stats=dict(data.get("font_stats") or {}),
        )


@dataclass(frozen=True)
class TableCell:
    row: int
    col: int
    text: str
    bbox: list[float] = field(default_factory=list)
    is_header: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TableCell":
        return cls(
            row=int(data.get("row") or 0),
            col=int(data.get("col") or 0),
            text=str(data.get("text") or ""),
            bbox=[float(item) for item in data.get("bbox") or []],
            is_header=bool(data.get("is_header")),
        )


@dataclass(frozen=True)
class TableRecord:
    table_id: str
    page_no: int
    bbox: list[float]
    cells: list[TableCell] = field(default_factory=list)
    header_rows: list[list[str]] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    nearby_caption: str = ""
    section_path: list[str] = field(default_factory=list)
    source: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TableRecord":
        return cls(
            table_id=str(data["table_id"]),
            page_no=int(data.get("page_no") or 1),
            bbox=[float(item) for item in data.get("bbox") or []],
            cells=[TableCell.from_dict(item) for item in data.get("cells") or []],
            header_rows=[[str(cell) for cell in row] for row in data.get("header_rows") or []],
            records=[dict(item) for item in data.get("records") or []],
            nearby_caption=str(data.get("nearby_caption") or ""),
            section_path=[str(item) for item in data.get("section_path") or []],
            source=str(data.get("source") or ""),
        )


@dataclass(frozen=True)
class NumberRecord:
    number_id: str
    raw: str
    value: float
    unit: str = ""
    scale: float = 1.0
    year: str = ""
    metric_hint: str = ""
    source_block_id: str = ""
    source_table_id: str = ""
    bbox: list[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NumberRecord":
        return cls(
            number_id=str(data["number_id"]),
            raw=str(data.get("raw") or ""),
            value=float(data.get("value") or 0.0),
            unit=str(data.get("unit") or ""),
            scale=float(data.get("scale") or 1.0),
            year=str(data.get("year") or ""),
            metric_hint=str(data.get("metric_hint") or ""),
            source_block_id=str(data.get("source_block_id") or ""),
            source_table_id=str(data.get("source_table_id") or ""),
            bbox=[float(item) for item in data.get("bbox") or []],
        )


@dataclass(frozen=True)
class PageIR:
    doc_id: str
    domain: str
    page_no: int
    width: float
    height: float
    text_blocks: list[TextBlock] = field(default_factory=list)
    tables: list[TableRecord] = field(default_factory=list)
    numbers: list[NumberRecord] = field(default_factory=list)
    sections: list[list[str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PageIR":
        return cls(
            doc_id=str(data["doc_id"]),
            domain=str(data["domain"]),
            page_no=int(data.get("page_no") or 1),
            width=float(data.get("width") or 0.0),
            height=float(data.get("height") or 0.0),
            text_blocks=[TextBlock.from_dict(item) for item in data.get("text_blocks") or []],
            tables=[TableRecord.from_dict(item) for item in data.get("tables") or []],
            numbers=[NumberRecord.from_dict(item) for item in data.get("numbers") or []],
            sections=[[str(part) for part in section] for section in data.get("sections") or []],
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class EvidenceBlock:
    evidence_id: str
    doc_id: str
    domain: str
    title: str
    source_path: str
    page_start: int
    page_end: int
    text: str
    block_type: str = "text"
    section_path: list[str] = field(default_factory=list)
    article_no: str = ""
    table_id: str = ""
    number_ids: list[str] = field(default_factory=list)
    row_key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return self.evidence_id

    @property
    def section_title(self) -> str:
        return " > ".join(self.section_path)

    @property
    def clause_no(self) -> str:
        return self.article_no

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["chunk_id"] = self.evidence_id
        data["section_title"] = self.section_title
        data["clause_no"] = self.article_no
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceBlock":
        section_path = [str(item) for item in data.get("section_path") or []]
        if not section_path and data.get("section_title"):
            section_path = [part.strip() for part in str(data.get("section_title") or "").split(">") if part.strip()]
        return cls(
            evidence_id=str(data.get("evidence_id") or data.get("chunk_id") or ""),
            doc_id=str(data["doc_id"]),
            domain=str(data["domain"]),
            title=str(data.get("title") or data["doc_id"]),
            source_path=str(data.get("source_path") or ""),
            page_start=int(data.get("page_start") or 1),
            page_end=int(data.get("page_end") or data.get("page_start") or 1),
            text=str(data.get("text") or ""),
            block_type=str(data.get("block_type") or "text"),
            section_path=section_path,
            article_no=str(data.get("article_no") or data.get("clause_no") or ""),
            table_id=str(data.get("table_id") or ""),
            number_ids=[str(item) for item in data.get("number_ids") or []],
            row_key=str(data.get("row_key") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


ChunkRecord = EvidenceBlock


@dataclass(frozen=True)
class RetrievalHit:
    chunk: EvidenceBlock
    score: float

    def to_evidence_item(self) -> dict[str, Any]:
        return {
            "evidence_id": self.chunk.evidence_id,
            "chunk_id": self.chunk.evidence_id,
            "doc_id": self.chunk.doc_id,
            "domain": self.chunk.domain,
            "title": self.chunk.title,
            "page_start": self.chunk.page_start,
            "page_end": self.chunk.page_end,
            "score": self.score,
            "block_type": self.chunk.block_type,
            "section_path": self.chunk.section_path,
            "section_title": self.chunk.section_title,
            "article_no": self.chunk.article_no,
            "clause_no": self.chunk.article_no,
            "table_id": self.chunk.table_id,
            "number_ids": self.chunk.number_ids,
            "row_key": self.chunk.row_key,
            "text": self.chunk.text,
            "metadata": self.chunk.metadata,
        }


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Usage":
        data = data or {}
        prompt = int(data.get("prompt_tokens") or 0)
        completion = int(data.get("completion_tokens") or 0)
        total = int(data.get("total_tokens") or prompt + completion)
        return cls(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)

    def add(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ModelResponse:
    parsed: dict[str, Any]
    content: str
    usage: Usage
    raw: dict[str, Any]


@dataclass(frozen=True)
class AnswerResult:
    qid: str
    answer: str
    usage: Usage
    evidence: dict[str, Any]
    model_output: dict[str, Any]
    raw_model_content: str

    def to_log_record(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "answer": self.answer,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total_tokens,
            "evidence": self.evidence,
            "model_output": self.model_output,
            "raw_model_content": self.raw_model_content,
        }
