from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
from statistics import median
from typing import Any, Iterable

from agent.models import (
    EvidenceBlock,
    NumberRecord,
    PageIR,
    ParsedDocument,
    TableCell,
    TableRecord,
    TextBlock,
    TextSpan,
)
from agent.text_utils import first_nonempty_line, normalize_text


PDF_DOMAINS = {"financial_contracts", "financial_reports", "insurance", "research"}
STRICT_ARTICLE_RE = re.compile(
    r"^\s*("
    r"第[一二三四五六七八九十百千万零〇\d]+[章节条款]"
    r"|\d+(?:\.\d+){1,5}(?=[、.．\s])"
    r"|[一二三四五六七八九十]+[、.．]"
    r"|[（(][一二三四五六七八九十\d]+[)）]"
    r")"
)
YEAR_OR_DATE_PREFIX_RE = re.compile(r"^\s*(?:19|20)\d{2}\s*(?:年|[-/.])")
AMOUNT_PREFIX_RE = re.compile(r"^\s*-?\d[\d,]*(?:\.\d+)?\s*(?:万亿元|亿元|百万元|万元|千元|元|%|％|个百分点|百分点|亿|万)")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])(-?\d[\d,]*(?:\.\d+)?)\s*(万亿元|亿元|百万元|万元|千元|元|%|％|个百分点|百分点|亿|万)?")
YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})年?")
TITLE_MARKERS = (
    "目录",
    "释义",
    "特别提示",
    "重大事项",
    "保险责任",
    "责任免除",
    "等待期",
    "免赔额",
    "赔偿处理",
    "合同效力",
    "违约责任",
    "付款",
    "交付",
    "报告期",
    "经营情况",
    "财务状况",
    "监管要求",
    "法律责任",
    "投资建议",
    "风险提示",
)
METRIC_HINTS = (
    "营业收入",
    "研发投入",
    "研发投入占营业收入",
    "经营活动产生的现金流量净额",
    "归属于上市公司股东的净利润",
    "归母净利润",
    "净利润",
    "现金分红",
    "毛利率",
    "免赔额",
    "等待期",
    "违约金",
    "本金",
    "利息",
)


def build_parse_artifacts(data_dir: str | Path) -> dict[str, Any]:
    raw_dir = Path(data_dir) / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw data directory not found: {raw_dir}")

    documents: list[ParsedDocument] = []
    pages: list[PageIR] = []
    evidence_blocks: list[EvidenceBlock] = []
    errors: list[dict[str, str]] = []
    empty_pages = 0

    for domain, path in _iter_sources(raw_dir):
        try:
            doc, doc_pages = _parse_source(path=path, domain=domain)
        except Exception as exc:
            errors.append({"path": str(path), "domain": domain, "error": str(exc)})
            continue
        documents.append(doc)
        pages.extend(doc_pages)
        empty_pages += sum(1 for page in doc_pages if not page.text_blocks and not page.tables)
        evidence_blocks.extend(_build_evidence_blocks(doc, doc_pages))

    documents.sort(key=lambda item: (item.domain, item.doc_id))
    pages.sort(key=lambda item: (item.domain, item.doc_id, item.page_no))
    evidence_blocks.sort(key=lambda item: (item.domain, item.doc_id, item.page_start, item.evidence_id))
    return {
        "documents": documents,
        "pages": pages,
        "evidence_blocks": evidence_blocks,
        "summary": _parse_summary(documents, pages, evidence_blocks, empty_pages, errors),
    }


def _iter_sources(raw_dir: Path) -> Iterable[tuple[str, Path]]:
    for domain in sorted(PDF_DOMAINS):
        for path in sorted([*(raw_dir / domain).glob("*.pdf"), *(raw_dir / domain).glob("*.PDF")]):
            yield domain, path
    regulatory = raw_dir / "regulatory"
    for path in sorted((regulatory / "html").glob("*.html")):
        yield "regulatory", path
    for path in sorted((regulatory / "attachments").glob("*.pdf")):
        yield "regulatory", path
    for path in sorted((regulatory / "txt").glob("*.txt")):
        yield "regulatory", path


def _parse_source(*, path: Path, domain: str) -> tuple[ParsedDocument, list[PageIR]]:
    doc_id = _canonical_doc_id(path, domain)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = _parse_pdf(path=path, domain=domain, doc_id=doc_id)
        source_type = "pdf"
    elif suffix == ".html":
        pages = _parse_html(path=path, domain=domain, doc_id=doc_id)
        source_type = "html"
    elif suffix == ".txt":
        pages = _parse_txt(path=path, domain=domain, doc_id=doc_id)
        source_type = "txt"
    else:
        raise ValueError(f"unsupported source type: {path}")
    if not pages:
        raise ValueError(f"source produced no pages: {path}")
    title = _source_title(path, suffix) or first_nonempty_line("\n".join(block.text for block in pages[0].text_blocks), default=doc_id)
    return (
        ParsedDocument(
            doc_id=doc_id,
            domain=domain,
            title=title,
            source_path=str(path),
            source_type=source_type,
            page_count=len(pages),
            metadata={"parser": "deterministic_ir_v1"},
        ),
        pages,
    )


def _canonical_doc_id(path: Path, domain: str) -> str:
    stem = path.stem.strip()
    if domain == "financial_contracts":
        match = re.match(r"^(text\d{1,2})\b", stem, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return stem


def _source_title(path: Path, suffix: str) -> str:
    if suffix != ".html":
        return ""
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return ""
    try:
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    except Exception:
        return ""
    meta_title = soup.find("meta", attrs={"name": "ArticleTitle"})
    if meta_title and meta_title.get("content"):
        return normalize_text(str(meta_title["content"]))[:120]
    if soup.title and soup.title.string:
        return normalize_text(soup.title.string.replace("_中国证券监督管理委员会", ""))[:120]
    return ""


def _parse_pdf(*, path: Path, domain: str, doc_id: str) -> list[PageIR]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF extraction. Install requirements.txt first.") from exc

    markdown_heading_hints = _pymupdf4llm_heading_hints(path)
    pages: list[PageIR] = []
    section_state: list[str] = []
    plumber_pdf = _open_pdfplumber(path) if domain == "financial_reports" else None
    with fitz.open(path) as pdf:
        try:
            for page_index, page in enumerate(pdf, start=1):
                text_blocks, section_state = _extract_pdf_text_blocks(
                    page=page,
                    page_no=page_index,
                    domain=domain,
                    doc_id=doc_id,
                    section_state=section_state,
                    markdown_heading_hints=markdown_heading_hints,
                )
                page_text = "\n".join(block.text for block in text_blocks)
                if _page_has_table_signal(domain, page_text):
                    table_candidates = _extract_pymupdf_tables(page=page, page_no=page_index, doc_id=doc_id, section_state=section_state)
                    if plumber_pdf is not None:
                        table_candidates.extend(_pdfplumber_doc_page_tables(pdf=plumber_pdf, page_no=page_index, doc_id=doc_id, section_state=section_state))
                    tables = _merge_tables(table_candidates)
                else:
                    tables = []
                numbers = _extract_page_numbers(doc_id=doc_id, page_no=page_index, text_blocks=text_blocks, tables=tables)
                pages.append(
                    PageIR(
                        doc_id=doc_id,
                        domain=domain,
                        page_no=page_index,
                        width=float(page.rect.width),
                        height=float(page.rect.height),
                        text_blocks=text_blocks,
                        tables=tables,
                        numbers=numbers,
                        sections=_unique_sections([block.section_path for block in text_blocks] + [table.section_path for table in tables]),
                        metadata={"source_type": "pdf"},
                    )
                )
        finally:
            if plumber_pdf is not None:
                plumber_pdf.close()
    return pages


def _parse_html(*, path: Path, domain: str, doc_id: str) -> list[PageIR]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("beautifulsoup4 is required for HTML extraction. Install requirements.txt first.") from exc

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    body = soup.find(class_="detail-news") or soup.find("body") or soup
    section_state: list[str] = []
    blocks: list[TextBlock] = []
    tables: list[TableRecord] = []
    for tag in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr", "table"]):
        if tag.name == "table":
            rows = [
                [normalize_text(cell.get_text(" ")) for cell in row.find_all(["th", "td"])]
                for row in tag.find_all("tr")
            ]
            table = _table_from_rows(
                rows=rows,
                table_id=f"{doc_id}:p1:htmltbl{len(tables)+1}",
                page_no=1,
                bbox=[],
                source="html",
                section_path=section_state,
                nearby_caption="",
            )
            if table is not None:
                tables.append(table)
            continue
        if tag.name == "tr":
            continue
        text = normalize_text(tag.get_text("\n"))
        if not text:
            continue
        article_no = extract_article_no(text)
        role = "title" if tag.name in {"h1", "h2", "h3", "h4"} else ("article" if article_no else "text")
        if role == "title" or _is_heading_text(text):
            section_state = _update_section_path(section_state, text, article_no=article_no, font_rank=2)
            role = "title"
        blocks.append(
            TextBlock(
                block_id=f"{doc_id}:p1:b{len(blocks)+1}",
                page_no=1,
                bbox=[],
                text=text,
                block_role=role,
                section_path=list(section_state),
                article_no=article_no,
                font_stats={},
            )
        )
    if not blocks and not tables:
        blocks = _text_to_blocks(normalize_text(body.get_text("\n")), doc_id=doc_id, page_no=1, section_state=section_state)
    numbers = _extract_page_numbers(doc_id=doc_id, page_no=1, text_blocks=blocks, tables=tables)
    return [
        PageIR(
            doc_id=doc_id,
            domain=domain,
            page_no=1,
            width=0.0,
            height=0.0,
            text_blocks=blocks,
            tables=tables,
            numbers=numbers,
            sections=_unique_sections([block.section_path for block in blocks] + [table.section_path for table in tables]),
            metadata={"source_type": "html"},
        )
    ]


def _parse_txt(*, path: Path, domain: str, doc_id: str) -> list[PageIR]:
    text = normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
    blocks = _text_to_blocks(text, doc_id=doc_id, page_no=1, section_state=[])
    numbers = _extract_page_numbers(doc_id=doc_id, page_no=1, text_blocks=blocks, tables=[])
    return [
        PageIR(
            doc_id=doc_id,
            domain=domain,
            page_no=1,
            width=0.0,
            height=0.0,
            text_blocks=blocks,
            tables=[],
            numbers=numbers,
            sections=_unique_sections([block.section_path for block in blocks]),
            metadata={"source_type": "txt"},
        )
    ]


def _extract_pdf_text_blocks(
    *,
    page: Any,
    page_no: int,
    domain: str,
    doc_id: str,
    section_state: list[str],
    markdown_heading_hints: set[str],
) -> tuple[list[TextBlock], list[str]]:
    raw = page.get_text("dict")
    entries: list[dict[str, Any]] = []
    font_sizes: list[float] = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines: list[str] = []
        spans: list[TextSpan] = []
        sizes: list[float] = []
        flags: list[int] = []
        for line in block.get("lines", []):
            parts: list[str] = []
            for span in line.get("spans", []):
                value = str(span.get("text") or "").strip()
                if not value:
                    continue
                size = float(span.get("size") or 0.0)
                flag = int(span.get("flags") or 0)
                parts.append(value)
                sizes.append(size)
                flags.append(flag)
                if size > 0:
                    font_sizes.append(size)
                spans.append(
                    TextSpan(
                        text=value,
                        bbox=[float(item) for item in span.get("bbox") or []],
                        font=str(span.get("font") or ""),
                        size=size,
                        flags=flag,
                    )
                )
            line_text = normalize_text("".join(parts))
            if line_text:
                lines.append(line_text)
        text = normalize_text("\n".join(lines))
        if text:
            entries.append(
                {
                    "text": text,
                    "spans": spans,
                    "bbox": [float(item) for item in block.get("bbox") or []],
                    "max_size": max(sizes) if sizes else 0.0,
                    "avg_size": sum(sizes) / len(sizes) if sizes else 0.0,
                    "bold_like": any(flag & 16 for flag in flags),
                }
            )
    entries.sort(key=lambda item: ((item["bbox"][1] if item["bbox"] else 0.0), (item["bbox"][0] if item["bbox"] else 0.0)))
    typical_size = median(font_sizes) if font_sizes else 0.0
    blocks: list[TextBlock] = []
    for entry in entries:
        text = str(entry["text"])
        article_no = extract_article_no(text)
        role = _block_role(
            text=text,
            domain=domain,
            max_size=float(entry["max_size"]),
            typical_size=typical_size,
            bold_like=bool(entry["bold_like"]),
            markdown_heading_hints=markdown_heading_hints,
            article_no=article_no,
        )
        if role == "title":
            section_state = _update_section_path(section_state, text, article_no=article_no, font_rank=_font_rank(float(entry["max_size"]), typical_size))
        blocks.append(
            TextBlock(
                block_id=f"{doc_id}:p{page_no}:b{len(blocks)+1}",
                page_no=page_no,
                bbox=list(entry["bbox"]),
                text=text,
                spans=list(entry["spans"]),
                block_role=role,
                section_path=list(section_state),
                article_no=article_no,
                font_stats={
                    "max_font_size": round(float(entry["max_size"]), 2),
                    "avg_font_size": round(float(entry["avg_size"]), 2),
                    "typical_font_size": round(float(typical_size), 2),
                    "bold_like": bool(entry["bold_like"]),
                },
            )
        )
    return blocks, section_state


def _pymupdf4llm_heading_hints(path: Path) -> set[str]:
    try:
        import pymupdf4llm
    except Exception:
        return set()
    try:
        markdown = pymupdf4llm.to_markdown(str(path), pages=[0])
    except Exception:
        return set()
    hints: set[str] = set()
    for line in str(markdown).splitlines():
        line = line.strip()
        if line.startswith("#"):
            value = normalize_text(line.lstrip("#").strip())
            if value:
                hints.add(value[:80])
    return hints


def _open_pdfplumber(path: Path) -> Any | None:
    try:
        import pdfplumber
    except Exception:
        return None
    try:
        return pdfplumber.open(str(path))
    except Exception:
        return None


def _pdfplumber_doc_page_tables(*, pdf: Any, page_no: int, doc_id: str, section_state: list[str]) -> list[TableRecord]:
    out: list[TableRecord] = []
    try:
        if page_no < 1 or page_no > len(pdf.pages):
            return []
        page = pdf.pages[page_no - 1]
        for table_index, table in enumerate(page.find_tables() or [], start=1):
            record = _table_from_rows(
                rows=table.extract() or [],
                table_id=f"{doc_id}:p{page_no}:plumber{table_index}",
                page_no=page_no,
                bbox=[float(item) for item in (table.bbox or [])],
                source="pdfplumber",
                section_path=section_state,
                nearby_caption="",
            )
            if record is not None:
                out.append(record)
    except Exception:
        return []
    return out


def _page_has_table_signal(domain: str, text: str) -> bool:
    compact = normalize_text(text)
    if not compact:
        return False
    years = set(YEAR_RE.findall(compact))
    has_amount = bool(re.search(r"\d[\d,]*(?:\.\d+)?\s*(?:亿元|万元|%|％|元|亿|万)", compact))
    if domain == "financial_reports" and any(term in compact for term in METRIC_HINTS):
        return True
    if domain == "financial_reports" and len(years) >= 2 and has_amount:
        return True
    if domain == "regulatory" and len(years) >= 2 and has_amount and any(marker in compact for marker in ["项目", "指标", "单位：", "单位:", "合计"]):
        return True
    return False


def _extract_pymupdf_tables(*, page: Any, page_no: int, doc_id: str, section_state: list[str]) -> list[TableRecord]:
    finder = getattr(page, "find_tables", None)
    if finder is None:
        return []
    out: list[TableRecord] = []
    try:
        tables = finder()
    except Exception:
        return []
    for table_index, table in enumerate(getattr(tables, "tables", []) or [], start=1):
        record = _table_from_rows(
            rows=table.extract() or [],
            table_id=f"{doc_id}:p{page_no}:fitz{table_index}",
            page_no=page_no,
            bbox=[float(item) for item in getattr(table, "bbox", []) or []],
            source="pymupdf",
            section_path=section_state,
            nearby_caption="",
        )
        if record is not None:
            out.append(record)
    return out


def _table_from_rows(
    *,
    rows: Iterable[Iterable[Any]],
    table_id: str,
    page_no: int,
    bbox: list[float],
    source: str,
    section_path: list[str],
    nearby_caption: str,
) -> TableRecord | None:
    clean_rows = [[normalize_text(str(cell or "")).replace("\n", " ") for cell in row] for row in rows]
    clean_rows = [row for row in clean_rows if any(cell for cell in row)]
    if not clean_rows:
        return None
    width = max(len(row) for row in clean_rows)
    clean_rows = [row + [""] * (width - len(row)) for row in clean_rows]
    header_count = _header_row_count(clean_rows)
    header_rows = clean_rows[:header_count]
    headers = _flatten_headers(header_rows, width)
    cells: list[TableCell] = []
    for row_index, row in enumerate(clean_rows):
        for col_index, cell in enumerate(row):
            if cell:
                cells.append(TableCell(row=row_index, col=col_index, text=cell, is_header=row_index < header_count))
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(clean_rows[header_count:], start=header_count):
        row_key = next((cell for cell in row if cell), "")
        values = {headers[col] or f"col_{col+1}": row[col] for col in range(width) if row[col] or headers[col]}
        if values:
            records.append({"row_index": row_index, "row_key": row_key[:120], "values": values})
    return TableRecord(
        table_id=table_id,
        page_no=page_no,
        bbox=bbox,
        cells=cells,
        header_rows=header_rows,
        records=records,
        nearby_caption=nearby_caption,
        section_path=list(section_path),
        source=source,
    )


def _merge_tables(tables: list[TableRecord]) -> list[TableRecord]:
    kept: list[TableRecord] = []
    for table in tables:
        duplicate_index = next((index for index, existing in enumerate(kept) if _bbox_iou(table.bbox, existing.bbox) >= 0.75), None)
        if duplicate_index is None:
            kept.append(table)
            continue
        existing = kept[duplicate_index]
        if len(table.cells) > len(existing.cells):
            kept[duplicate_index] = table
    return kept


def _extract_page_numbers(*, doc_id: str, page_no: int, text_blocks: list[TextBlock], tables: list[TableRecord]) -> list[NumberRecord]:
    out: list[NumberRecord] = []
    for block in text_blocks:
        out.extend(_numbers_from_text(doc_id=doc_id, page_no=page_no, text=block.text, source_block_id=block.block_id, source_table_id="", bbox=block.bbox, offset=len(out)))
    for table in tables:
        for record_index, record in enumerate(table.records):
            row_key = str(record.get("row_key") or "")
            for key, value in (record.get("values") or {}).items():
                text = f"{row_key} {key} {value}"
                out.extend(
                    _numbers_from_text(
                        doc_id=doc_id,
                        page_no=page_no,
                        text=text,
                        source_block_id="",
                        source_table_id=table.table_id,
                        bbox=table.bbox,
                        offset=len(out),
                    )
                )
    return out


def _numbers_from_text(
    *,
    doc_id: str,
    page_no: int,
    text: str,
    source_block_id: str,
    source_table_id: str,
    bbox: list[float],
    offset: int,
) -> list[NumberRecord]:
    out: list[NumberRecord] = []
    years = YEAR_RE.findall(text)
    metric_hint = _metric_hint(text)
    for match_index, match in enumerate(NUMBER_RE.finditer(text), start=1):
        raw_num = match.group(1)
        unit = match.group(2) or ""
        raw = f"{raw_num}{unit}"
        if unit == "" and raw_num in years:
            scale = 1.0
            value = float(raw_num)
            year = raw_num
        else:
            value, scale = _normalize_number(raw_num, unit)
            year = _nearest_year(text, match.start(), years)
        out.append(
            NumberRecord(
                number_id=f"{doc_id}:p{page_no}:n{offset + match_index}",
                raw=raw,
                value=value,
                unit=unit,
                scale=scale,
                year=year,
                metric_hint=metric_hint,
                source_block_id=source_block_id,
                source_table_id=source_table_id,
                bbox=list(bbox),
            )
        )
    return out


def _build_evidence_blocks(doc: ParsedDocument, pages: list[PageIR]) -> list[EvidenceBlock]:
    out: list[EvidenceBlock] = []
    local_index = 0
    for page in pages:
        buffer: list[TextBlock] = []

        def flush() -> None:
            nonlocal local_index, buffer
            if not buffer:
                return
            text = normalize_text("\n".join(_format_text_block(block) for block in buffer))
            if text:
                local_index += 1
                last = buffer[-1]
                first = buffer[0]
                block_ids = {block.block_id for block in buffer}
                block_numbers = [number for number in page.numbers if number.source_block_id in block_ids]
                out.append(
                    EvidenceBlock(
                        evidence_id=f"{doc.domain}:{doc.doc_id}:ev{local_index:05d}",
                        doc_id=doc.doc_id,
                        domain=doc.domain,
                        title=doc.title,
                        source_path=doc.source_path,
                        page_start=first.page_no,
                        page_end=last.page_no,
                        text=text,
                        block_type=_evidence_block_type(buffer),
                        section_path=list(last.section_path),
                        article_no=last.article_no,
                        number_ids=[number.number_id for number in block_numbers],
                        metadata={
                            "source_block_ids": [block.block_id for block in buffer],
                            "numbers": [number.__dict__ for number in block_numbers],
                        },
                    )
                )
            buffer = []

        for block in page.text_blocks:
            if not block.text:
                continue
            starts_unit = block.block_role == "title" or bool(block.article_no)
            if starts_unit:
                flush()
            buffer.append(block)
            if sum(len(item.text) for item in buffer) >= 1000:
                flush()
        flush()

        for table in page.tables:
            for record in table.records:
                text = _format_table_record(table, record)
                if not text:
                    continue
                local_index += 1
                table_numbers = [number for number in page.numbers if number.source_table_id == table.table_id and number.raw in text]
                out.append(
                    EvidenceBlock(
                        evidence_id=f"{doc.domain}:{doc.doc_id}:ev{local_index:05d}",
                        doc_id=doc.doc_id,
                        domain=doc.domain,
                        title=doc.title,
                        source_path=doc.source_path,
                        page_start=table.page_no,
                        page_end=table.page_no,
                        text=text,
                        block_type="table_record",
                        section_path=list(table.section_path),
                        table_id=table.table_id,
                        row_key=str(record.get("row_key") or ""),
                        number_ids=[number.number_id for number in table_numbers],
                        metadata={"table_record": record, "table_source": table.source, "numbers": [number.__dict__ for number in table_numbers]},
                    )
                )
    return out


def extract_article_no(text: str) -> str:
    compact = normalize_text(text).replace("\n", "")
    if not compact or YEAR_OR_DATE_PREFIX_RE.match(compact) or AMOUNT_PREFIX_RE.match(compact):
        return ""
    match = STRICT_ARTICLE_RE.match(compact)
    if not match:
        return ""
    article = match.group(1).strip()
    if re.fullmatch(r"\d{4}", article):
        return ""
    return article


def _block_role(
    *,
    text: str,
    domain: str,
    max_size: float,
    typical_size: float,
    bold_like: bool,
    markdown_heading_hints: set[str],
    article_no: str,
) -> str:
    compact = normalize_text(text).replace("\n", "")
    if article_no and ("条" not in article_no and "款" not in article_no) and _article_is_heading(article_no, compact):
        return "title"
    if article_no:
        return "article"
    if _is_heading_text(compact) or compact[:80] in markdown_heading_hints:
        return "title"
    if domain in {"financial_reports", "research"} and len(compact) <= 36 and re.search(r"(报告|行业|公司|摘要|分析|财务|经营)", compact):
        return "title"
    if typical_size and max_size >= typical_size + 1.8 and len(compact) <= 80:
        return "title"
    if bold_like and len(compact) <= 60 and any(marker in compact for marker in TITLE_MARKERS):
        return "title"
    return "text"


def _is_heading_text(text: str) -> bool:
    compact = normalize_text(text).replace("\n", "")
    if len(compact) <= 48 and any(marker in compact for marker in TITLE_MARKERS):
        return True
    if re.fullmatch(r"第[一二三四五六七八九十百千万零〇\d]+[章节].{0,40}", compact):
        return True
    if re.fullmatch(r"[一二三四五六七八九十]+[、.．].{1,42}", compact):
        return True
    return False


def _article_is_heading(article_no: str, text: str) -> bool:
    if "章" in article_no or "节" in article_no:
        return True
    return len(text) <= 80 and bool(re.match(r"^\s*(?:\d+(?:\.\d+){1,5}|[一二三四五六七八九十]+[、.．])", text))


def _update_section_path(section_state: list[str], text: str, *, article_no: str, font_rank: int) -> list[str]:
    title = normalize_text(text).splitlines()[0][:100]
    if not title:
        return section_state
    if "章" in article_no:
        level = 1
    elif "节" in article_no:
        level = 2
    elif re.match(r"^\s*[一二三四五六七八九十]+[、.．]", title):
        level = 2
    elif re.match(r"^\s*\d+(?:\.\d+){1,5}", title):
        level = min(title.split()[0].count(".") + 1, 4)
    else:
        level = max(1, min(font_rank, 4))
    state = list(section_state[: max(level - 1, 0)])
    state.append(title)
    return state


def _font_rank(max_size: float, typical_size: float) -> int:
    if typical_size <= 0:
        return 3
    delta = max_size - typical_size
    if delta >= 6:
        return 1
    if delta >= 3:
        return 2
    return 3


def _text_to_blocks(text: str, *, doc_id: str, page_no: int, section_state: list[str]) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    for line in [item for item in normalize_text(text).splitlines() if item.strip()]:
        article_no = extract_article_no(line)
        if article_no and ("条" in article_no or "款" in article_no):
            role = "article"
        else:
            role = "title" if _is_heading_text(line) else ("article" if article_no else "text")
        if role == "title":
            section_state = _update_section_path(section_state, line, article_no=article_no, font_rank=3)
        blocks.append(
            TextBlock(
                block_id=f"{doc_id}:p{page_no}:b{len(blocks)+1}",
                page_no=page_no,
                bbox=[],
                text=line,
                block_role=role,
                section_path=list(section_state),
                article_no=article_no,
            )
        )
    return blocks


def _format_text_block(block: TextBlock) -> str:
    parts: list[str] = []
    if block.section_path and block.block_role != "title":
        parts.append(f"【章节】{' > '.join(block.section_path)}")
    if block.article_no:
        parts.append(f"【条款】{block.article_no}")
    parts.append(block.text)
    return normalize_text("\n".join(parts))


def _format_table_record(table: TableRecord, record: dict[str, Any]) -> str:
    values = record.get("values") or {}
    parts: list[str] = []
    if table.section_path:
        parts.append(f"【章节】{' > '.join(table.section_path)}")
    if record.get("row_key"):
        parts.append(f"【行名】{record['row_key']}")
    parts.append("表格记录: " + " | ".join(f"{key}: {value}" for key, value in values.items() if str(value).strip()))
    return normalize_text("\n".join(parts))


def _evidence_block_type(blocks: list[TextBlock]) -> str:
    roles = {block.block_role for block in blocks}
    if len(roles) == 1:
        role = next(iter(roles))
        return "article" if role == "article" else role
    return "mixed"


def _number_ids_for_blocks(numbers: list[NumberRecord], block_ids: set[str]) -> list[str]:
    return [number.number_id for number in numbers if number.source_block_id in block_ids]


def _header_row_count(rows: list[list[str]]) -> int:
    if len(rows) <= 1:
        return 1
    first = rows[0]
    if any(YEAR_RE.search(cell) for cell in first[1:]) or any("项目" in cell or "指标" in cell for cell in first):
        return 1
    second = rows[1]
    if sum(bool(cell) for cell in first) <= max(2, len(first) // 2) and any(YEAR_RE.search(cell) for cell in second):
        return 2
    return 1


def _flatten_headers(header_rows: list[list[str]], width: int) -> list[str]:
    headers: list[str] = []
    for col in range(width):
        parts = [row[col] for row in header_rows if col < len(row) and row[col]]
        headers.append(" ".join(dict.fromkeys(parts)) if parts else f"col_{col+1}")
    return headers


def _bbox_iou(left: list[float], right: list[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    l_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    r_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    denom = l_area + r_area - inter
    return inter / denom if denom > 0 else 0.0


def _normalize_number(raw: str, unit: str) -> tuple[float, float]:
    value = float(raw.replace(",", ""))
    scale = 1.0
    if unit == "万亿元":
        scale = 1_000_000_000_000.0
    elif unit in {"亿元", "亿"}:
        scale = 100_000_000.0
    elif unit == "百万元":
        scale = 1_000_000.0
    elif unit == "万元" or unit == "万":
        scale = 10_000.0
    elif unit == "千元":
        scale = 1_000.0
    elif unit in {"%", "％"}:
        scale = 0.01
    elif unit in {"个百分点", "百分点"}:
        scale = 0.01
    return value * scale, scale


def _nearest_year(text: str, offset: int, years: list[str]) -> str:
    if not years:
        return ""
    positions = [(match.group(1), abs(match.start() - offset)) for match in YEAR_RE.finditer(text)]
    return min(positions, key=lambda item: item[1])[0] if positions else years[0]


def _metric_hint(text: str) -> str:
    for metric in METRIC_HINTS:
        if metric in text:
            return metric
    return ""


def _unique_sections(sections: list[list[str]]) -> list[list[str]]:
    out: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for section in sections:
        key = tuple(item for item in section if item)
        if key and key not in seen:
            seen.add(key)
            out.append(list(key))
    return out


def _parse_summary(
    documents: list[ParsedDocument],
    pages: list[PageIR],
    evidence_blocks: list[EvidenceBlock],
    empty_pages: int,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    for doc in documents:
        by_domain[doc.domain]["documents"] += 1
        by_domain[doc.domain][f"source_{doc.source_type}"] += 1
    for page in pages:
        by_domain[page.domain]["pages"] += 1
        by_domain[page.domain]["text_blocks"] += len(page.text_blocks)
        by_domain[page.domain]["tables"] += len(page.tables)
        by_domain[page.domain]["numbers"] += len(page.numbers)
        by_domain[page.domain]["articles"] += sum(1 for block in page.text_blocks if block.article_no)
    for block in evidence_blocks:
        by_domain[block.domain]["evidence_blocks"] += 1
    return {
        "documents": len(documents),
        "pages": len(pages),
        "evidence_blocks": len(evidence_blocks),
        "empty_pages": empty_pages,
        "errors": errors,
        "domains": {domain: dict(counter) for domain, counter in sorted(by_domain.items())},
    }
