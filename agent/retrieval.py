from __future__ import annotations

from collections import defaultdict
import json
import os
import pickle
from pathlib import Path
import re
from typing import Any, Iterable

from agent.models import EvidenceBlock, RetrievalHit


RRF_K = 60
METRIC_TERMS = (
    "营业收入",
    "研发投入",
    "研发投入占营业收入",
    "研发投入强度",
    "经营活动产生的现金流量净额",
    "经营活动现金流净额",
    "现金流量净额",
    "归属于上市公司股东的净利润",
    "归母净利润",
    "净利润",
    "同比增长",
    "同比增减",
    "现金分红",
    "毛利率",
    "免赔额",
    "等待期",
    "责任免除",
    "保险责任",
    "违约责任",
    "付款期限",
    "交付",
)


class SparseRetriever:
    def __init__(
        self,
        evidence_blocks: list[EvidenceBlock],
        *,
        fulltext_bm25: object,
        fulltext_tokens: list[list[str]],
        field_vectorizer: object,
        field_matrix: object,
        article_index: dict[str, list[int]],
        table_index: list[dict[str, Any]],
        number_index: list[dict[str, Any]],
    ) -> None:
        self.chunks = evidence_blocks
        self.evidence_blocks = evidence_blocks
        self.fulltext_bm25 = fulltext_bm25
        self.fulltext_tokens = fulltext_tokens
        self.field_vectorizer = field_vectorizer
        self.field_matrix = field_matrix
        self.article_index = article_index
        self.table_index = table_index
        self.number_index = number_index

    @classmethod
    def build(cls, evidence_blocks: list[EvidenceBlock]) -> "SparseRetriever":
        if not evidence_blocks:
            raise ValueError("cannot build index without evidence blocks")
        from rank_bm25 import BM25Okapi
        from sklearn.feature_extraction.text import TfidfVectorizer

        fulltext_corpus = [_fulltext(block) for block in evidence_blocks]
        fulltext_tokens = [_tokenize(text) for text in fulltext_corpus]
        fulltext_bm25 = BM25Okapi(fulltext_tokens)
        field_corpus = [_field_text(block) for block in evidence_blocks]
        field_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        field_matrix = field_vectorizer.fit_transform(field_corpus)
        return cls(
            evidence_blocks,
            fulltext_bm25=fulltext_bm25,
            fulltext_tokens=fulltext_tokens,
            field_vectorizer=field_vectorizer,
            field_matrix=field_matrix,
            article_index=_build_article_index(evidence_blocks),
            table_index=_build_table_index(evidence_blocks),
            number_index=_build_number_index(evidence_blocks),
        )

    def save(self, processed_dir: str | Path) -> None:
        root = Path(processed_dir)
        root.mkdir(parents=True, exist_ok=True)
        _pickle_atomic(
            root / "fulltext_bm25.pkl",
            {"bm25": self.fulltext_bm25, "tokens": self.fulltext_tokens, "version": "ir_v1"},
        )
        _pickle_atomic(
            root / "field_index.pkl",
            {"vectorizer": self.field_vectorizer, "matrix": self.field_matrix, "version": "ir_v1"},
        )
        _write_json(root / "article_index.json", self.article_index)
        _write_jsonl(root / "table_index.jsonl", self.table_index)
        _write_jsonl(root / "number_index.jsonl", self.number_index)

    @classmethod
    def load(cls, processed_dir: str | Path, evidence_blocks: list[EvidenceBlock]) -> "SparseRetriever":
        root = Path(processed_dir)
        with (root / "fulltext_bm25.pkl").open("rb") as f:
            fulltext_payload = pickle.load(f)
        with (root / "field_index.pkl").open("rb") as f:
            field_payload = pickle.load(f)
        return cls(
            evidence_blocks,
            fulltext_bm25=fulltext_payload["bm25"],
            fulltext_tokens=fulltext_payload.get("tokens") or [],
            field_vectorizer=field_payload["vectorizer"],
            field_matrix=field_payload["matrix"],
            article_index={str(key): [int(item) for item in value] for key, value in _read_json(root / "article_index.json").items()},
            table_index=_read_jsonl(root / "table_index.jsonl"),
            number_index=_read_jsonl(root / "number_index.jsonl"),
        )

    def search(
        self,
        query: str,
        *,
        domain: str | None = None,
        doc_ids: Iterable[str] | None = None,
        top_k: int = 8,
    ) -> list[RetrievalHit]:
        if not query.strip():
            raise ValueError("empty query")
        candidate_indices = self._candidate_indices(domain=domain, doc_ids=set(doc_ids or []))
        if not candidate_indices:
            return []
        rank_window = max(top_k * 8, 40)
        rankings = [
            self._bm25_ranking(query, candidate_indices, rank_window),
            self._field_ranking(query, candidate_indices, rank_window),
            self._article_ranking(query, candidate_indices, rank_window),
            self._table_ranking(query, candidate_indices, rank_window),
            self._number_ranking(query, candidate_indices, rank_window),
            self._exact_ranking(query, candidate_indices, rank_window),
        ]
        fused: dict[int, float] = defaultdict(float)
        raw_best: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            for rank, (index, score) in enumerate(ranking, start=1):
                if score <= 0:
                    continue
                fused[index] += 1.0 / (RRF_K + rank)
                raw_best[index] = max(raw_best[index], float(score))
        ranked = sorted(fused.items(), key=lambda item: (item[1], raw_best[item[0]]), reverse=True)
        return [RetrievalHit(chunk=self.chunks[index], score=float(score)) for index, score in ranked[:top_k]]

    def _bm25_ranking(self, query: str, candidate_indices: list[int], top_k: int) -> list[tuple[int, float]]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        ranked = [(index, _bm25_subset_score(self.fulltext_bm25, query_tokens, index)) for index in candidate_indices]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def _field_ranking(self, query: str, candidate_indices: list[int], top_k: int) -> list[tuple[int, float]]:
        qvec = self.field_vectorizer.transform([query])
        scores = self.field_matrix[candidate_indices].dot(qvec.T).toarray().ravel()
        ranked = sorted(((candidate_indices[i], float(score)) for i, score in enumerate(scores)), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def _article_ranking(self, query: str, candidate_indices: list[int], top_k: int) -> list[tuple[int, float]]:
        refs = set(_article_refs(query))
        if not refs:
            return []
        candidate_set = set(candidate_indices)
        scores: dict[int, float] = defaultdict(float)
        for ref in refs:
            for index in self.article_index.get(ref, []):
                if index in candidate_set:
                    scores[index] += 30.0
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    def _table_ranking(self, query: str, candidate_indices: list[int], top_k: int) -> list[tuple[int, float]]:
        terms = _exact_terms(query)
        if not terms:
            return []
        candidate_set = set(candidate_indices)
        scores: dict[int, float] = defaultdict(float)
        for row in self.table_index:
            index = int(row.get("index") or -1)
            if index not in candidate_set:
                continue
            text = str(row.get("search_text") or "")
            score = sum((len(term) + 5.0) for term in terms if term in text)
            if score:
                scores[index] += score
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    def _number_ranking(self, query: str, candidate_indices: list[int], top_k: int) -> list[tuple[int, float]]:
        years = set(re.findall(r"(?:19|20)\d{2}", query))
        metrics = [term for term in METRIC_TERMS if term in query]
        has_number = bool(re.search(r"\d", query))
        if not years and not metrics and not has_number:
            return []
        candidate_set = set(candidate_indices)
        scores: dict[int, float] = defaultdict(float)
        for row in self.number_index:
            index = int(row.get("index") or -1)
            if index not in candidate_set:
                continue
            score = 0.0
            if row.get("year") and str(row["year"]) in years:
                score += 12.0
            if row.get("metric_hint") and str(row["metric_hint"]) in metrics:
                score += 18.0
            if has_number:
                score += 2.0
            if score:
                scores[index] += score
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    def _exact_ranking(self, query: str, candidate_indices: list[int], top_k: int) -> list[tuple[int, float]]:
        terms = _exact_terms(query)
        if not terms:
            return []
        ranked: list[tuple[int, float]] = []
        for index in candidate_indices:
            block = self.chunks[index]
            text = _fulltext(block)
            score = sum(text.count(term) * (3.0 + min(len(term), 18)) for term in terms)
            if block.article_no and block.article_no in query:
                score += 25.0
            if block.row_key and block.row_key in query:
                score += 20.0
            if score > 0:
                ranked.append((index, score / max(len(text) ** 0.25, 1.0)))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def _candidate_indices(self, *, domain: str | None, doc_ids: set[str]) -> list[int]:
        out: list[int] = []
        for index, block in enumerate(self.chunks):
            if domain and block.domain != domain:
                continue
            if doc_ids and block.doc_id not in doc_ids:
                continue
            out.append(index)
        return out


def _fulltext(block: EvidenceBlock) -> str:
    return " ".join(item for item in [block.domain, block.doc_id, block.title, block.section_title, block.article_no, block.row_key, block.text] if item)


def _field_text(block: EvidenceBlock) -> str:
    return " ".join(
        item
        for item in [
            f"domain:{block.domain}",
            f"doc:{block.doc_id}",
            f"type:{block.block_type}",
            f"section:{block.section_title}",
            f"article:{block.article_no}",
            f"row:{block.row_key}",
            block.text,
        ]
        if item
    )


def _tokenize(text: str) -> list[str]:
    try:
        import jieba
    except ImportError as exc:
        raise RuntimeError("jieba is required for hybrid retrieval. Install requirements.txt first.") from exc
    tokens: list[str] = []
    for part in re.findall(r"[A-Za-z0-9_.-]+|[\u4e00-\u9fff]+", text):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            tokens.extend(item for item in jieba.lcut(part, cut_all=False) if len(item.strip()) >= 2)
        elif len(part.strip()) >= 2:
            tokens.append(part.lower())
    return tokens


def _exact_terms(query: str) -> list[str]:
    terms: list[str] = [term for term in METRIC_TERMS if term in query]
    terms.extend(_article_refs(query))
    terms.extend(re.findall(r"[A-Za-z0-9_./-]{3,}", query))
    terms.extend(re.findall(r"[\u4e00-\u9fff]{4,18}", query))
    ignored = {"正确的是", "正确的有", "下列哪些", "以下哪些", "判断题", "根据相关", "关于下列"}
    unique: list[str] = []
    for term in terms:
        term = term.strip()
        if len(term) < 2 or term in ignored:
            continue
        if term not in unique:
            unique.append(term)
    return unique[:80]


def _article_refs(text: str) -> list[str]:
    refs = re.findall(r"第[一二三四五六七八九十百千万零〇\d]+[章节条款]", text)
    refs.extend(re.findall(r"\d+(?:\.\d+){1,5}", text))
    unique: list[str] = []
    for ref in refs:
        if ref not in unique:
            unique.append(ref)
    return unique


def _build_article_index(blocks: list[EvidenceBlock]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for i, block in enumerate(blocks):
        if block.article_no:
            index[block.article_no].append(i)
    return dict(index)


def _build_table_index(blocks: list[EvidenceBlock]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, block in enumerate(blocks):
        if block.table_id or block.block_type == "table_record":
            rows.append(
                {
                    "index": i,
                    "evidence_id": block.evidence_id,
                    "doc_id": block.doc_id,
                    "table_id": block.table_id,
                    "row_key": block.row_key,
                    "search_text": _fulltext(block),
                }
            )
    return rows


def _build_number_index(blocks: list[EvidenceBlock]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, block in enumerate(blocks):
        metadata = block.metadata or {}
        numbers = metadata.get("numbers") or []
        if not numbers and block.number_ids:
            numbers = [{"number_id": number_id} for number_id in block.number_ids]
        for number in numbers:
            rows.append(
                {
                    "index": i,
                    "evidence_id": block.evidence_id,
                    "doc_id": block.doc_id,
                    "number_id": number.get("number_id"),
                    "year": number.get("year"),
                    "metric_hint": number.get("metric_hint"),
                    "raw": number.get("raw"),
                }
            )
    return rows


def _bm25_subset_score(bm25: object, query_tokens: list[str], index: int) -> float:
    idf = getattr(bm25, "idf", {})
    doc_freqs = getattr(bm25, "doc_freqs", [])
    doc_len = getattr(bm25, "doc_len", [])
    avgdl = float(getattr(bm25, "avgdl", 0.0) or 0.0)
    k1 = float(getattr(bm25, "k1", 1.5))
    b = float(getattr(bm25, "b", 0.75))
    if index >= len(doc_freqs) or index >= len(doc_len) or avgdl <= 0:
        return 0.0
    score = 0.0
    freqs = doc_freqs[index]
    length = float(doc_len[index] or 0)
    denominator_base = k1 * (1 - b + b * length / avgdl)
    for token in query_tokens:
        freq = float(freqs.get(token, 0) or 0)
        if freq <= 0:
            continue
        score += float(idf.get(token, 0.0) or 0.0) * (freq * (k1 + 1) / (freq + denominator_base))
    return score


def _pickle_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
