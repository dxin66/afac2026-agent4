from __future__ import annotations

from agent.models import DocumentBlock, ChunkRecord, DocumentRecord
from agent.text_utils import normalize_text


SINGLETON_BLOCK_TYPES = {"table_row"}
STRUCTURED_DOMAINS = {"financial_contracts", "financial_reports", "insurance", "regulatory", "research"}


def build_chunks(
    documents: list[DocumentRecord],
    target_chars: int = 1000,
    overlap_chars: int = 160,
) -> list[ChunkRecord]:
    if target_chars <= overlap_chars:
        raise ValueError("target_chars must be greater than overlap_chars")
    chunks: list[ChunkRecord] = []
    for doc in documents:
        chunks.extend(_chunk_document(doc, target_chars=target_chars, overlap_chars=overlap_chars))
    return chunks


def _chunk_document(doc: DocumentRecord, target_chars: int, overlap_chars: int) -> list[ChunkRecord]:
    blocks = _document_blocks(doc)
    if blocks and doc.domain in STRUCTURED_DOMAINS:
        return _chunk_blocks(doc, blocks=blocks, target_chars=target_chars)
    return _chunk_pages(doc, target_chars=target_chars, overlap_chars=overlap_chars)


def _chunk_blocks(doc: DocumentRecord, blocks: list[DocumentBlock], target_chars: int) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    buffer: list[DocumentBlock] = []
    local_index = 0

    def append_chunk(text: str, *, page_start: int, page_end: int, block_type: str, section_title: str, clause_no: str, row_key: str) -> None:
        nonlocal local_index
        text = normalize_text(text)
        if not text:
            return
        for segment in _split_long_text(text, target_chars):
            local_index += 1
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{doc.domain}:{doc.doc_id}:{local_index:05d}",
                    doc_id=doc.doc_id,
                    domain=doc.domain,
                    title=doc.title,
                    source_path=doc.source_path,
                    page_start=page_start,
                    page_end=page_end,
                    text=segment,
                    block_type=block_type,
                    section_title=section_title,
                    clause_no=clause_no,
                    row_key=row_key,
                    metadata={"chunking": "structure"},
                )
            )

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        text = "\n".join(_format_block(block) for block in buffer)
        first = buffer[0]
        last = buffer[-1]
        section_title = _last_nonempty(block.section_title for block in buffer)
        clause_no = _last_nonempty(block.clause_no for block in buffer)
        row_key = _last_nonempty(block.row_key for block in buffer)
        block_types = {block.block_type for block in buffer}
        block_type = next(iter(block_types)) if len(block_types) == 1 else "mixed"
        append_chunk(
            text,
            page_start=first.page,
            page_end=last.page,
            block_type=block_type,
            section_title=section_title,
            clause_no=clause_no,
            row_key=row_key,
        )
        buffer = []

    for block in blocks:
        if not normalize_text(block.text):
            continue
        if block.block_type in SINGLETON_BLOCK_TYPES:
            flush()
            append_chunk(
                _format_block(block),
                page_start=block.page,
                page_end=block.page,
                block_type=block.block_type,
                section_title=block.section_title,
                clause_no=block.clause_no,
                row_key=block.row_key,
            )
            continue

        starts_new_unit = block.block_type == "title" or bool(block.clause_no)
        if starts_new_unit and buffer:
            flush()
        buffer.append(block)
        if sum(len(_format_block(item)) for item in buffer) >= target_chars:
            flush()
    flush()
    return chunks


def _chunk_pages(doc: DocumentRecord, target_chars: int, overlap_chars: int) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    buffer: list[str] = []
    page_start: int | None = None
    page_end: int | None = None
    local_index = 0

    def flush() -> None:
        nonlocal buffer, page_start, page_end, local_index
        text = normalize_text("\n".join(buffer))
        if not text:
            buffer = []
            page_start = None
            page_end = None
            return
        local_index += 1
        chunks.append(
            ChunkRecord(
                chunk_id=f"{doc.domain}:{doc.doc_id}:{local_index:05d}",
                doc_id=doc.doc_id,
                domain=doc.domain,
                title=doc.title,
                source_path=doc.source_path,
                page_start=page_start or 1,
                page_end=page_end or page_start or 1,
                text=text,
                metadata={"chunking": "char_window"},
            )
        )
        overlap = text[-overlap_chars:] if overlap_chars > 0 else ""
        buffer = [overlap] if overlap else []
        page_start = page_end if overlap else None

    for page in doc.pages:
        text = normalize_text(page.text)
        if not text:
            continue
        if len(text) > target_chars:
            if buffer:
                flush()
            start = 0
            step = target_chars - overlap_chars
            while start < len(text):
                segment = text[start : start + target_chars]
                if segment.strip():
                    local_index += 1
                    chunks.append(
                        ChunkRecord(
                            chunk_id=f"{doc.domain}:{doc.doc_id}:{local_index:05d}",
                            doc_id=doc.doc_id,
                            domain=doc.domain,
                            title=doc.title,
                            source_path=doc.source_path,
                            page_start=page.page,
                            page_end=page.page,
                            text=segment.strip(),
                            metadata={"chunking": "char_window"},
                        )
                    )
                start += step
            buffer = []
            page_start = None
            page_end = None
            continue
        if page_start is None:
            page_start = page.page
        page_end = page.page
        buffer.append(text)
        if sum(len(item) for item in buffer) >= target_chars:
            flush()
    if buffer:
        text = normalize_text("\n".join(buffer))
        if text and (not chunks or chunks[-1].text != text):
            local_index += 1
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{doc.domain}:{doc.doc_id}:{local_index:05d}",
                    doc_id=doc.doc_id,
                    domain=doc.domain,
                    title=doc.title,
                    source_path=doc.source_path,
                    page_start=page_start or 1,
                    page_end=page_end or page_start or 1,
                    text=text,
                    metadata={"chunking": "char_window"},
                )
            )
    return chunks


def _document_blocks(doc: DocumentRecord) -> list[DocumentBlock]:
    blocks: list[DocumentBlock] = []
    for page in doc.pages:
        if page.blocks:
            blocks.extend(page.blocks)
        elif page.text:
            blocks.append(
                DocumentBlock(
                    page=page.page,
                    block_index=len(blocks) + 1,
                    block_type="text",
                    text=page.text,
                )
            )
    return blocks


def _format_block(block: DocumentBlock) -> str:
    parts: list[str] = []
    if block.section_title and block.block_type != "title":
        parts.append(f"【章节】{block.section_title}")
    if block.clause_no:
        parts.append(f"【条款】{block.clause_no}")
    if block.row_key:
        parts.append(f"【行名】{block.row_key}")
    parts.append(block.text)
    return normalize_text("\n".join(parts))


def _split_long_text(text: str, target_chars: int) -> list[str]:
    text = normalize_text(text)
    if len(text) <= target_chars:
        return [text]
    units = [unit.strip() for unit in text.splitlines() if unit.strip()]
    if len(units) <= 1:
        return [text[start : start + target_chars].strip() for start in range(0, len(text), target_chars) if text[start : start + target_chars].strip()]
    segments: list[str] = []
    buffer: list[str] = []
    used = 0
    for unit in units:
        if buffer and used + len(unit) > target_chars:
            segments.append(normalize_text("\n".join(buffer)))
            buffer = []
            used = 0
        buffer.append(unit)
        used += len(unit)
    if buffer:
        segments.append(normalize_text("\n".join(buffer)))
    return segments


def _last_nonempty(values: object) -> str:
    result = ""
    for value in values:
        value = str(value or "").strip()
        if value:
            result = value
    return result
