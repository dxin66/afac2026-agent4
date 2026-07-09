from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from agent.answer_utils import option_text
from agent.models import RetrievalHit
from agent.question_plan import build_question_plan
from agent.retrieval import SparseRetriever
from agent.text_utils import compact_for_query


# Per-domain context budgets. Documents vary enormously in size (insurance/research
# docs run ~20-40 pages while financial_contracts/financial_reports run ~300 pages),
# and the 5,000,000 token evaluation budget has ~25x headroom over what a full run
# actually consumes, so budgets are sized to keep coherent, uncompressed evidence
# spans rather than the small snippets that fragment connected clauses.
DOMAIN_CONTEXT_BUDGETS: dict[str, dict[str, int]] = {
    "insurance": {"max_chars": 14000, "max_item_chars": 900, "top_global": 12, "top_per_option": 4, "top_per_doc": 3},
    "research": {"max_chars": 13000, "max_item_chars": 850, "top_global": 12, "top_per_option": 4, "top_per_doc": 3},
    "regulatory": {"max_chars": 12000, "max_item_chars": 750, "top_global": 12, "top_per_option": 4, "top_per_doc": 3},
    "financial_contracts": {"max_chars": 13000, "max_item_chars": 750, "top_global": 12, "top_per_option": 4, "top_per_doc": 3},
    "financial_reports": {"max_chars": 12000, "max_item_chars": 700, "top_global": 12, "top_per_option": 4, "top_per_doc": 3},
}


class EvidenceBuilder:
    def __init__(
        self,
        retriever: SparseRetriever,
        *,
        top_global: int = 10,
        top_per_option: int = 3,
        top_docs_without_doc_ids: int = 20,
        candidate_chunk_hits: int = 600,
        max_chars: int = 9000,
        max_item_chars: int = 520,
        top_per_doc: int = 2,
    ) -> None:
        self.retriever = retriever
        self.top_global = top_global
        self.top_per_option = top_per_option
        self.top_docs_without_doc_ids = top_docs_without_doc_ids
        self.candidate_chunk_hits = candidate_chunk_hits
        self.max_chars = max_chars
        self.max_item_chars = max_item_chars
        self.top_per_doc = top_per_doc
        self.chunk_index_by_id = {chunk.chunk_id: index for index, chunk in enumerate(self.retriever.chunks)}
        self.chunks_by_domain: dict[str, list[Any]] = defaultdict(list)
        self.chunks_by_doc: dict[tuple[str, str], list[Any]] = defaultdict(list)
        self.doc_titles: dict[tuple[str, str], str] = {}
        for chunk in self.retriever.chunks:
            self.chunks_by_domain[chunk.domain].append(chunk)
            self.chunks_by_doc[(chunk.domain, chunk.doc_id)].append(chunk)
            self.doc_titles.setdefault((chunk.domain, chunk.doc_id), str(chunk.title or ""))

    def build(self, question: dict[str, Any]) -> dict[str, Any]:
        budget = DOMAIN_CONTEXT_BUDGETS.get(str(question.get("domain") or ""), {})
        saved_budget = (self.max_chars, self.max_item_chars, self.top_global, self.top_per_option, self.top_per_doc)
        self.max_chars = budget.get("max_chars", self.max_chars)
        self.max_item_chars = budget.get("max_item_chars", self.max_item_chars)
        self.top_global = budget.get("top_global", self.top_global)
        self.top_per_option = budget.get("top_per_option", self.top_per_option)
        self.top_per_doc = budget.get("top_per_doc", self.top_per_doc)
        try:
            return self._build(question)
        finally:
            self.max_chars, self.max_item_chars, self.top_global, self.top_per_option, self.top_per_doc = saved_budget

    def _build(self, question: dict[str, Any]) -> dict[str, Any]:
        query = _augment_query(question, build_query(question))
        doc_ids = [str(item) for item in question.get("doc_ids") or []]
        if doc_ids:
            candidate_doc_ids = doc_ids
        else:
            candidate_doc_ids = self.infer_doc_ids(question)
        if not candidate_doc_ids:
            raise ValueError(f"no candidate documents found for {question['qid']}")

        hits: list[RetrievalHit] = []
        hits.extend(
            self.retriever.search(
                query,
                domain=question["domain"],
                doc_ids=candidate_doc_ids,
                top_k=self.top_global,
            )
        )
        hits.extend(self._keyword_hits(question, query, candidate_doc_ids, top_k=12))
        hits.extend(self._must_cover_hits(question, candidate_doc_ids))
        hits.extend(self._doc_coverage_hits(question, query, candidate_doc_ids))
        for key, value in sorted((question.get("options") or {}).items()):
            option_query = f"{question['question']}\n{key}. {value}"
            hits.extend(
                self.retriever.search(
                    compact_for_query(option_query),
                    domain=question["domain"],
                    doc_ids=candidate_doc_ids,
                    top_k=self.top_per_option,
                )
            )
            hits.extend(self._keyword_hits(question, compact_for_query(option_query), candidate_doc_ids, top_k=4))
            hits.extend(self._must_cover_hits(question, candidate_doc_ids, option_key=str(key), option_text_value=str(value)))
        hits.extend(self._option_doc_keyword_coverage_hits(question, candidate_doc_ids))

        selected = self._dedupe_hits(hits)
        selected = self._dedupe_hits([*selected, *self._context_hits(selected)])
        selected = self._prioritize_doc_coverage(question, selected, candidate_doc_ids)
        if not selected:
            raise ValueError(f"no evidence chunks found for {question['qid']}")
        evidence_items = self._compress_items([hit.to_evidence_item() for hit in selected], query)
        if not evidence_items:
            raise ValueError(f"evidence is empty after trimming for {question['qid']}")
        question_plan = build_question_plan(question, candidate_doc_ids=candidate_doc_ids)
        option_evidence = self._build_option_evidence_matrix(question, evidence_items, candidate_doc_ids, question_plan)
        retrieval_evidence_matrix = self._build_retrieval_evidence_matrix(question, option_evidence, question_plan)
        return {
            "qid": question["qid"],
            "domain": question["domain"],
            "query": query,
            "candidate_doc_ids": candidate_doc_ids,
            "query_terms": _expand_query_terms(question, query),
            "question_plan": question_plan,
            "items": evidence_items,
            "option_evidence": option_evidence,
            "retrieval_evidence_matrix": retrieval_evidence_matrix,
            "candidate_coverage": _candidate_coverage(question_plan, retrieval_evidence_matrix),
            "matrix_gaps": _candidate_matrix_gaps(question_plan, retrieval_evidence_matrix),
        }

    def infer_doc_ids(self, question: dict[str, Any]) -> list[str]:
        query = build_query(question)
        explicit = _extract_explicit_doc_ids(question)
        inferred = self._infer_doc_ids(question, query)
        merged: list[str] = []
        for doc_id in [*explicit, *inferred]:
            if doc_id not in merged:
                merged.append(doc_id)
        return merged[: self.top_docs_without_doc_ids]

    def _infer_doc_ids(self, question: dict[str, Any], query: str) -> list[str]:
        hits = self.retriever.search(query, domain=question["domain"], top_k=self.candidate_chunk_hits)
        scores: dict[str, float] = defaultdict(float)
        coverage: dict[str, int] = defaultdict(int)
        for hit in hits:
            scores[hit.chunk.doc_id] += hit.score
            coverage[hit.chunk.doc_id] += 1
        keyword_scores = self._keyword_doc_scores(question, query)
        for doc_id, score in keyword_scores.items():
            scores[doc_id] += score
            coverage[doc_id] += 1
        ranked = sorted(scores.items(), key=lambda item: (item[1], coverage[item[0]]), reverse=True)
        return [doc_id for doc_id, _ in ranked[: self.top_docs_without_doc_ids]]

    def _keyword_doc_scores(self, question: dict[str, Any], query: str) -> dict[str, float]:
        terms = _expand_query_terms(question, query)
        if not terms:
            return {}
        per_doc: dict[str, list[float]] = defaultdict(list)
        domain = question["domain"]
        for chunk in self.chunks_by_domain.get(domain, []):
            text = " ".join(
                item
                for item in [chunk.title, chunk.section_title, chunk.clause_no, chunk.row_key, chunk.text]
                if item
            )
            score = _keyword_chunk_score(text, terms)
            if score > 0:
                per_doc[chunk.doc_id].append(score)
        doc_scores: dict[str, float] = {}
        for doc_id, values in per_doc.items():
            top_values = sorted(values, reverse=True)[:5]
            doc_scores[doc_id] = sum(top_values) * 0.01
        for (domain, doc_id), title in self.doc_titles.items():
            if domain != question["domain"] or not title:
                continue
            if title in query or _title_alias_match(title, query):
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 10.0
        return doc_scores

    def _keyword_hits(self, question: dict[str, Any], query: str, doc_ids: list[str], top_k: int) -> list[RetrievalHit]:
        terms = _expand_query_terms(question, query)
        if not terms:
            return []
        doc_set = set(doc_ids)
        scored: list[RetrievalHit] = []
        for chunk in self._candidate_chunks(question["domain"], doc_set):
            score = _keyword_chunk_score(_chunk_evidence_text(chunk), terms)
            if score <= 0:
                continue
            scored.append(RetrievalHit(chunk=chunk, score=score))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    def _candidate_chunks(self, domain: str, doc_ids: set[str]) -> list[Any]:
        chunks: list[Any] = []
        for doc_id in doc_ids:
            chunks.extend(self.chunks_by_doc.get((domain, doc_id), []))
        return chunks

    def _doc_coverage_hits(self, question: dict[str, Any], query: str, doc_ids: list[str]) -> list[RetrievalHit]:
        explicit_doc_ids = [str(item) for item in question.get("doc_ids") or []]
        coverage_doc_ids = explicit_doc_ids or doc_ids[:4]
        hits: list[RetrievalHit] = []
        for doc_id in coverage_doc_ids:
            hits.extend(
                self.retriever.search(
                    query,
                    domain=question["domain"],
                    doc_ids=[doc_id],
                    top_k=self.top_per_doc,
                )
            )
        return hits

    def _option_doc_keyword_coverage_hits(self, question: dict[str, Any], doc_ids: list[str]) -> list[RetrievalHit]:
        if not doc_ids:
            return []
        hits: list[RetrievalHit] = []
        domain = str(question.get("domain") or "")
        for _, value in sorted((question.get("options") or {}).items()):
            option_query = compact_for_query(str(value or ""))
            terms = _option_matrix_terms(domain, question, option_query)
            if not terms:
                terms = _expand_query_terms(question, option_query)
            if not terms:
                continue
            for doc_id in doc_ids:
                scored: list[RetrievalHit] = []
                for chunk in self.chunks_by_doc.get((question["domain"], str(doc_id)), []):
                    text = _chunk_evidence_text(chunk)
                    matched = [term for term in terms if term and term in text]
                    score = (len(set(matched)) * 100.0) + _keyword_chunk_score(text, terms)
                    if score > 0:
                        scored.append(RetrievalHit(chunk=chunk, score=score + 50.0))
                scored.sort(key=lambda hit: hit.score, reverse=True)
                hits.extend(scored[:2])
        return hits

    def _must_cover_hits(
        self,
        question: dict[str, Any],
        doc_ids: list[str],
        *,
        option_key: str = "",
        option_text_value: str = "",
    ) -> list[RetrievalHit]:
        domain = str(question.get("domain") or "")
        text = build_query(question)
        if option_text_value:
            text = f"{text} {option_key} {option_text_value}"
        specs = _domain_must_cover_specs(domain, text)
        if not specs:
            return []
        doc_set = set(doc_ids)
        hits: list[RetrievalHit] = []
        candidate_chunks = self._candidate_chunks(question["domain"], doc_set)
        for index, spec in enumerate(specs):
            spec_hits: list[RetrievalHit] = []
            for chunk in candidate_chunks:
                chunk_text = _chunk_evidence_text(chunk)
                if _matches_must_cover_spec(chunk_text, spec):
                    spec_score = 1000.0 - index + _keyword_chunk_score(chunk_text, list(spec))
                    spec_hits.append(RetrievalHit(chunk=chunk, score=spec_score))
            spec_hits.sort(key=lambda hit: hit.score, reverse=True)
            hits.extend(spec_hits[:4])
        return hits

    def _context_hits(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        context: list[RetrievalHit] = []
        for hit in hits[:24]:
            chunk = hit.chunk
            if not _needs_neighbor_context(chunk.block_type, chunk.text):
                continue
            index = self.chunk_index_by_id.get(chunk.chunk_id)
            if index is None:
                continue
            for neighbor_index in (index - 2, index - 1, index + 1):
                if neighbor_index < 0 or neighbor_index >= len(self.retriever.chunks):
                    continue
                neighbor = self.retriever.chunks[neighbor_index]
                if neighbor.domain != chunk.domain or neighbor.doc_id != chunk.doc_id:
                    continue
                context.append(RetrievalHit(chunk=neighbor, score=hit.score * 0.85))
        return context

    @staticmethod
    def _dedupe_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
        best: dict[str, RetrievalHit] = {}
        for hit in hits:
            existing = best.get(hit.chunk.chunk_id)
            if existing is None or hit.score > existing.score:
                best[hit.chunk.chunk_id] = hit
        return sorted(best.values(), key=lambda item: item.score, reverse=True)

    def _prioritize_doc_coverage(
        self,
        question: dict[str, Any],
        hits: list[RetrievalHit],
        candidate_doc_ids: list[str],
    ) -> list[RetrievalHit]:
        explicit_doc_ids = [str(item) for item in question.get("doc_ids") or []]
        if not explicit_doc_ids:
            return hits

        per_doc_slots = _doc_slot_count(str(question.get("domain") or ""), len(explicit_doc_ids))
        priority: list[RetrievalHit] = []
        used: set[str] = set()
        by_doc: dict[str, list[RetrievalHit]] = defaultdict(list)
        for hit in hits:
            by_doc[hit.chunk.doc_id].append(hit)

        for doc_id in explicit_doc_ids:
            doc_hits = by_doc.get(doc_id, [])
            if len(doc_hits) < per_doc_slots:
                doc_hits = self._dedupe_hits(
                    [
                        *doc_hits,
                        *self.retriever.search(
                            build_query(question),
                            domain=question["domain"],
                            doc_ids=[doc_id],
                            top_k=max(per_doc_slots * 2, 4),
                        ),
                    ]
                )
            for hit in doc_hits[:per_doc_slots]:
                if hit.chunk.chunk_id not in used:
                    priority.append(hit)
                    used.add(hit.chunk.chunk_id)

        # Keep explicit document coverage before global score ordering so compression
        # cannot drop low-scoring but necessary documents in multi-document questions.
        tail = [hit for hit in hits if hit.chunk.chunk_id not in used]
        return [*priority, *tail]

    def _compress_items(self, items: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        used = 0
        for item in items:
            original_text = str(item["text"])
            item_max_chars = self._item_char_budget(item)
            text, windows = compress_evidence_text(original_text, query, max_chars=item_max_chars)
            remaining = self.max_chars - used
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining].rstrip()
            if text:
                copied = dict(item)
                copied["text"] = text
                copied["compressed"] = True
                copied["original_text_chars"] = len(original_text)
                copied["windows"] = windows
                kept.append(copied)
                used += len(text)
        return kept

    def _item_char_budget(self, item: dict[str, Any]) -> int:
        block_type = str(item.get("block_type") or "")
        if block_type in {"table_row", "table_record"}:
            return self.max_item_chars * 2
        if item.get("clause_no") or item.get("article_no") or block_type in {"clause", "article", "mixed"}:
            return int(self.max_item_chars * 1.5)
        return self.max_item_chars

    def _build_option_evidence_matrix(
        self,
        question: dict[str, Any],
        evidence_items: list[dict[str, Any]],
        candidate_doc_ids: list[str],
        question_plan: dict[str, Any],
    ) -> dict[str, Any]:
        matrix: dict[str, Any] = {}
        domain = str(question.get("domain") or "")
        expected_docs = [str(item) for item in question.get("doc_ids") or candidate_doc_ids]
        option_doc_targets = question_plan.get("option_doc_targets") or {}

        for letter, option_value in sorted((question.get("options") or {}).items()):
            letter = str(letter).upper()
            option_value = str(option_value or "")
            terms = _option_matrix_terms(domain, question, option_value)
            target_docs = [str(item) for item in option_doc_targets.get(letter) or []]
            by_doc: dict[str, list[dict[str, Any]]] = {doc_id: [] for doc_id in target_docs}
            scored: list[tuple[float, str, dict[str, Any]]] = []
            for item in evidence_items:
                item_text = _evidence_item_search_text(item)
                matched_terms = [term for term in terms if term and term in item_text]
                if not matched_terms:
                    continue
                score = _keyword_chunk_score(item_text, matched_terms)
                doc_id = str(item.get("doc_id") or "")
                scored.append(
                    (
                        score,
                        doc_id,
                        {
                            "chunk_id": str(item.get("chunk_id") or ""),
                            "doc_id": doc_id,
                            "page_start": item.get("page_start"),
                            "page_end": item.get("page_end"),
                            "block_type": str(item.get("block_type") or ""),
                            "section_title": str(item.get("section_title") or ""),
                            "clause_no": str(item.get("clause_no") or ""),
                            "row_key": str(item.get("row_key") or ""),
                            "matched_terms": matched_terms[:10],
                            "text": _short_snippet(str(item.get("text") or ""), matched_terms, max_chars=260),
                            "score": round(float(score), 4),
                        },
                    )
                )
            scored.sort(key=lambda row: row[0], reverse=True)
            for _, doc_id, payload in scored:
                by_doc.setdefault(doc_id, [])
                if len(by_doc[doc_id]) < 2 and payload["chunk_id"] not in {row["chunk_id"] for row in by_doc[doc_id]}:
                    by_doc[doc_id].append(payload)

            for doc_id in target_docs:
                if by_doc.get(doc_id):
                    continue
                fallback_items = _fallback_option_doc_items(question, evidence_items, doc_id)
                by_doc[doc_id] = fallback_items[:2]

            matrix[letter] = {
                "option": option_value,
                "terms": terms[:24],
                "target_docs": target_docs,
                "by_doc": {doc_id: rows for doc_id, rows in by_doc.items() if rows or doc_id in target_docs},
                "missing_docs": [doc_id for doc_id in target_docs if not by_doc.get(doc_id)],
            }
        return matrix

    def _build_retrieval_evidence_matrix(
        self,
        question: dict[str, Any],
        option_evidence: dict[str, Any],
        question_plan: dict[str, Any],
    ) -> dict[str, Any]:
        matrix: dict[str, Any] = {}
        option_doc_targets = question_plan.get("option_doc_targets") or {}
        for letter, row in sorted(option_evidence.items()):
            letter = str(letter).upper()
            by_doc = row.get("by_doc") if isinstance(row, dict) else {}
            doc_ids = set(str(doc_id) for doc_id in (by_doc or {}))
            doc_ids.update(str(doc_id) for doc_id in option_doc_targets.get(letter) or [])
            matrix[letter] = {}
            for doc_id in sorted(doc_ids):
                candidates: list[dict[str, Any]] = []
                seen: set[str] = set()
                for index, item in enumerate((by_doc or {}).get(doc_id) or [], start=1):
                    chunk_id = str(item.get("chunk_id") or "")
                    if chunk_id in seen:
                        continue
                    seen.add(chunk_id)
                    candidate = dict(item)
                    candidate["evidence_id"] = f"{question['qid']}:{letter}:{doc_id}:{index}"
                    candidate["option"] = letter
                    candidate["candidate_source"] = "retrieval_evidence"
                    candidate["match_reason"] = _candidate_match_reason(candidate)
                    candidates.append(candidate)
                matrix[letter][doc_id] = {
                    "candidates": candidates,
                    "direct_support": [],
                    "direct_refute": [],
                    "derived_support": [],
                    "derived_refute": [],
                    "support": [],
                    "refute": [],
                    "neutral": [],
                }
        return matrix


def build_query(question: dict[str, Any]) -> str:
    parts = [
        str(question.get("domain", "")),
        str(question.get("type", "")),
        str(question.get("answer_format", "")),
        str(question.get("question", "")),
        option_text(question),
    ]
    return compact_for_query("\n".join(parts))


def _augment_query(question: dict[str, Any], query: str) -> str:
    terms = _expand_query_terms(question, query)
    if not terms:
        return query
    return compact_for_query(f"{query}\n" + " ".join(terms))


def compress_evidence_text(text: str, query: str, *, max_chars: int = 520) -> tuple[str, list[dict[str, Any]]]:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text, [{"start": 0, "end": len(text), "kind": "chunk"}]

    terms = _compression_terms(query)
    units = _split_units(text)
    scored: list[tuple[float, int, str]] = []
    for index, unit in enumerate(units):
        score = _unit_score(unit, terms)
        if score > 0:
            scored.append((score, index, unit))

    windows: list[tuple[int, int, str]] = []
    if scored:
        for _, index, _ in sorted(scored, key=lambda item: item[0], reverse=True)[:3]:
            start = max(0, index - 1)
            end = min(len(units), index + _forward_window_size(units[index]))
            window_text = " ".join(unit for unit in units[start:end] if unit.strip())
            if window_text:
                windows.append((start, end, window_text))
    else:
        for term in terms[:4]:
            pos = text.find(term)
            if pos >= 0:
                start = max(0, pos - 160)
                end = min(len(text), pos + len(term) + 180)
                windows.append((start, end, text[start:end]))

    if not windows:
        return text[:max_chars].rstrip(), [{"start": 0, "end": max_chars, "kind": "prefix"}]

    snippets: list[str] = []
    window_meta: list[dict[str, Any]] = []
    used = 0
    seen: set[str] = set()
    for start, end, window_text in windows:
        window_text = window_text.strip()
        if not window_text or window_text in seen:
            continue
        seen.add(window_text)
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(window_text) > remaining:
            window_text = window_text[:remaining].rstrip()
        snippets.append(window_text)
        used += len(window_text)
        window_meta.append({"start": start, "end": end, "kind": "hit_window"})
    compressed = "\n".join(snippets).strip()
    if len(compressed) > max_chars:
        compressed = compressed[:max_chars].rstrip()
    return compressed, window_meta


def _split_units(text: str) -> list[str]:
    lines = [line.strip() for line in re.split(r"[\n\r]+", text) if line.strip()]
    if len(lines) >= 2:
        return lines
    parts = re.split(r"(?<=[。！？；;])\s*|\s{2,}", text)
    units = [part.strip() for part in parts if part and part.strip()]
    if units:
        return units
    return [text]


def _query_terms(query: str) -> list[str]:
    raw = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,12}", query)
    ignored = {"正确", "错误", "下列", "哪些", "关于", "根据", "依据", "结合", "文档", "信息", "描述", "判断", "是否", "以下", "选项"}
    terms: list[str] = []
    for term in raw:
        term = term.strip()
        if len(term) < 2 or term in ignored:
            continue
        if term not in terms:
            terms.append(term)
    return terms[:80]


def _compression_terms(query: str) -> list[str]:
    terms = [term for term in _query_terms(query) if not _is_low_value_keyword(term)]
    metric_terms = [
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
        "身故保险金",
        "养老年金",
        "保单账户价值",
        "现金价值",
        "违约责任",
        "付款期限",
        "交付",
        "受益所有人",
        "客户尽职调查",
        "身份资料",
        "交易记录",
        "分类评价",
        "行政处罚",
        "生效时点",
        "投资建议",
        "风险提示",
    ]
    for term in metric_terms:
        if term in query and term not in terms:
            terms.insert(0, term)
    return terms[:80]


def _forward_window_size(unit: str) -> int:
    table_markers = ("营业收入", "研发投入", "现金流量净额", "净利润", "现金分红", "毛利率", "项目")
    if any(marker in unit for marker in table_markers):
        return 8
    return 3


def _expand_query_terms(question: dict[str, Any], query: str) -> list[str]:
    text = build_query(question)
    metric_terms = [
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
        "双位数",
        "现金分红",
        "分红",
        "营业成本",
        "毛利率",
        "免赔额",
        "等待期",
        "责任免除",
        "保险责任",
        "身故保险金",
        "养老年金",
        "保单账户价值",
        "现金价值",
        "保单贷款",
        "犹豫期",
        "宽限期",
        "效力中止",
        "违约责任",
        "付款期限",
        "交付",
        "合同金额",
        "受益所有人",
        "客户尽职调查",
        "身份资料",
        "交易记录",
        "内部审批",
        "报告",
        "分类评价",
        "行政处罚",
        "生效时点",
        "投资建议",
        "风险提示",
        "渠道贡献",
        "会计师事务所",
        "签字注册会计师",
        "未勤勉尽责",
        "处罚时效",
        "连续继续状态",
        "信息披露违法",
        "董事候选人",
        "公司章程",
        "股东会",
    ]
    if question.get("domain") == "financial_reports":
        terms = [term for term in metric_terms if term in text]
        if "研发投入强度" in text and "研发投入占营业收入" not in terms:
            terms.append("研发投入占营业收入")
        if "归母净利润" in text and "归属于上市公司股东的净利润" not in terms:
            terms.append("归属于上市公司股东的净利润")
        if "经营活动现金流净额" in text and "经营活动产生的现金流量净额" not in terms:
            terms.append("经营活动产生的现金流量净额")
        for term in _domain_synonym_terms("financial_reports", text):
            if term not in terms:
                terms.append(term)
        return list(dict.fromkeys(terms))

    terms = [term for term in _query_terms(query) if not _is_low_value_keyword(term)]
    for term in _domain_synonym_terms(str(question.get("domain") or ""), text):
        if term not in terms:
            terms.insert(0, term)
    for term in metric_terms:
        if term in text and term not in terms:
            terms.insert(0, term)
    for option_text_value in (question.get("options") or {}).values():
        for term in re.findall(r"[\u4e00-\u9fff]{4,18}", str(option_text_value)):
            if term not in terms:
                terms.append(term)
    return terms[:120]


def _domain_synonym_terms(domain: str, text: str) -> list[str]:
    terms: list[str] = []
    if domain == "financial_contracts":
        if "违约责任" in text or "违约利息" in text:
            terms.extend(["违约金", "本金和利息", "实际偿付之日"])
        if "资产减值补偿" in text:
            terms.extend(["减值测试报告", "出具之日起10日", "股份回购", "支付现金"])
        if "兑付日" in text:
            terms.extend(["兑付日", "付息日"])
    if domain == "research":
        if "服务零售" in text or "商品零售" in text:
            terms.extend(["服务消费", "服务性消费", "社会消费品零售", "限额以上商品零售"])
        if "保费贡献率" in text or "渠道贡献" in text or "银保" in text:
            terms.extend(["保费贡献", "银保渠道", "渠道贡献"])
        if "居民可支配收入" in text or "居民收入增速" in text:
            terms.extend(["居民可支配收入增长", "居民人均可支配", "居民收入增速", "居民人均可支配总收入"])
        if "人均名义GDP" in text or "人均名义 GDP" in text:
            terms.extend(["人均GDP", "人均名义GDP", "人均名义 GDP"])
    if domain == "financial_reports":
        if "营业收入" in text or "营收" in text:
            terms.extend(["营业收入", "营业总收入", "营收"])
        if "归母净利润" in text or "归属于上市公司股东的净利润" in text:
            terms.extend(["归母净利润", "归属于母公司股东的净利润", "归属于上市公司股东的净利润"])
        if "经营活动现金流净额" in text:
            terms.extend(["经营活动现金流净额", "经营活动产生的现金流量净额"])
    return list(dict.fromkeys(terms))


def _domain_must_cover_specs(domain: str, text: str) -> list[tuple[str, ...]]:
    if domain == "insurance":
        return _term_specs(
            text,
            [
                "保险责任",
                "责任免除",
                "等待期",
                "免赔额",
                "赔付比例",
                "身故保险金",
                "养老年金",
                "已领养老年金",
                "保单账户价值",
                "个人账户价值",
                "现金价值",
                "保单贷款",
                "犹豫期",
                "宽限期",
                "效力中止",
            ],
        )
    if domain == "financial_reports":
        specs = _term_specs(
            text,
            [
                "营业收入",
                "研发投入",
                "研发投入占营业收入",
                "经营活动产生的现金流量净额",
                "经营活动现金流净额",
                "归属于上市公司股东的净利润",
                "归母净利润",
                "净利润",
                "现金分红",
                "毛利率",
                "同比增长",
                "同比增减",
            ],
        )
        if "研发投入" in text and "营业收入" in text:
            specs.append(("研发投入", "营业收入"))
        if "营业收入" in text or "营收" in text:
            specs.extend([("营业总收入",), ("营收",)])
        if "经营活动" in text and "现金流" in text:
            specs.append(("经营活动", "现金流"))
        return list(dict.fromkeys(specs))
    if domain == "financial_contracts":
        specs = _term_specs(
            text,
            [
                "违约利息",
                "本金和利息",
                "兑付日",
                "资产减值补偿",
                "通知期限",
                "报告出具之日",
                "合同主体",
                "发行主体",
                "发行规模",
                "合同金额",
                "付款期限",
                "交付期限",
            ],
        )
        if "违约利息" in text or "违约责任" in text:
            specs.append(("违约金", "本金和利息"))
        if "资产减值补偿" in text:
            specs.append(("减值测试报告", "出具之日起10日"))
        return list(dict.fromkeys(specs))
    if domain == "research":
        specs = _term_specs(
            text,
            [
                "投资建议",
                "风险提示",
                "渠道贡献",
                "保费贡献率",
                "服务零售",
                "商品零售",
                "居民可支配收入",
                "居民收入增速",
                "人均名义GDP",
                "人均名义 GDP",
                "市场规模",
                "复合增速",
                "毛利率",
                "净利润",
                "营业收入",
            ],
        )
        if "服务零售" in text or "商品零售" in text:
            specs.extend([("服务消费",), ("服务性消费",), ("社会消费品零售",)])
        if "保费贡献率" in text or "渠道贡献" in text or "银保" in text:
            specs.extend([("保费贡献",), ("银保渠道", "贡献")])
        if "居民可支配收入" in text or "居民收入增速" in text:
            specs.extend([("居民可支配收入增长",), ("居民人均可支配",), ("居民收入增速",)])
        if "人均名义GDP" in text or "人均名义 GDP" in text:
            specs.append(("人均GDP",))
        return list(dict.fromkeys(specs))
    return []


def _term_specs(text: str, terms: list[str]) -> list[tuple[str, ...]]:
    return [(term,) for term in terms if term in text]


def _matches_must_cover_spec(text: str, spec: tuple[str, ...]) -> bool:
    return all(term in text for term in spec)


def _title_alias_match(title: str, query: str) -> bool:
    aliases = {
        "上市公司治理准则": ("上市公司治理规范", "治理规范"),
        "上市公司章程指引": ("章程指引", "公司章程指引"),
    }
    for canonical, query_aliases in aliases.items():
        if title.strip("《》") == canonical and any(alias in query for alias in query_aliases):
            return True
    return False


def _doc_slot_count(domain: str, doc_count: int) -> int:
    if doc_count <= 1:
        return 0
    if domain in {"insurance", "research"}:
        return 3
    if domain in {"financial_reports", "financial_contracts", "regulatory"}:
        return 2
    return 2


def _chunk_evidence_text(chunk: Any) -> str:
    return " ".join(
        item
        for item in [chunk.title, chunk.section_title, chunk.clause_no, chunk.row_key, chunk.text]
        if item
    )


def _evidence_item_search_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(value)
        for value in [
            item.get("title"),
            item.get("section_title"),
            item.get("clause_no"),
            item.get("row_key"),
            item.get("text"),
        ]
        if value
    )


def _option_matrix_terms(domain: str, question: dict[str, Any], option_text_value: str) -> list[str]:
    terms: list[str] = []
    text = str(option_text_value or "")
    for term in _option_entity_terms(text):
        if term not in terms:
            terms.append(term)
    for term in _option_role_terms(domain, text):
        if term not in terms:
            terms.append(term)
    for term in _option_numeric_terms(text):
        if term not in terms:
            terms.append(term)
    for term in _option_predicate_terms(text):
        if term not in terms:
            terms.append(term)
    for group in _domain_option_term_groups(domain, text):
        for term in group:
            if term not in terms:
                terms.append(term)
    for term in _domain_synonym_terms(domain, text):
        if term not in terms:
            terms.append(term)
    for term in re.findall(r"第[一二三四五六七八九十百千万\d]+[条章节款项]", text):
        if term not in terms:
            terms.append(term)
    if not _option_entity_terms(text):
        # Cross-product comparison options are often a bare brand/product name
        # (e.g. "平安智盈金生") with no company-legal suffix, so the entity
        # regex above never fires and this option gets zero terms, which left
        # the whole question's retrieval_evidence_matrix empty. Fall back to
        # the option text itself as a brand-name proxy.
        for term in re.findall(r"[一-鿿]{3,18}", text):
            if term not in terms:
                terms.append(term)
    # The attribute actually being compared (等待期/宽限期/施救费用...) often
    # appears once in the shared question stem rather than repeated inside
    # each option. Safe to share across every option in the same question
    # (unlike entity terms, which must stay option-specific) since it's a
    # closed, domain-scoped term list, not open-ended text extraction.
    stem = str(question.get("question") or "")
    for group in _domain_option_term_groups(domain, stem):
        for term in group:
            if term not in terms:
                terms.append(term)
    if not terms and question.get("answer_format") == "tf":
        for term in _expand_query_terms(question, build_query(question))[:20]:
            if term not in terms:
                terms.append(term)
    return terms[:40]


def _option_entity_terms(text: str) -> list[str]:
    terms: list[str] = []
    # Keep the regexes separate; the first pattern intentionally captures long
    # legal names, while the second catches shorter named entities.
    regexes = [
        r"[\u4e00-\u9fff]{2,30}(?:股份有限公司|有限责任公司|有限公司|控股集团有限公司|集团有限公司|证券股份有限公司)",
        r"[\u4e00-\u9fff]{2,16}(?:证券|银行|交易所|公司|集团|租赁|投资)",
    ]
    for regex in regexes:
        for match in re.findall(regex, text):
            cleaned = _trim_entity_term(match)
            if cleaned and cleaned not in terms:
                terms.append(cleaned)
    return terms


def _trim_entity_term(term: str) -> str:
    prefixes = ["第一份文档的", "第二份文档的", "两份文档的", "文档", "第一份", "第二份"]
    value = str(term or "")
    for prefix in prefixes:
        if value.startswith(prefix):
            value = value[len(prefix) :]
    for separator in ["明确指定", "名称为", "指定", "为", "是"]:
        if separator in value:
            value = value.split(separator)[-1]
    return value.strip("，。；：: 的")


def _option_role_terms(domain: str, text: str) -> list[str]:
    role_terms = [
        "发行人名称",
        "发行人",
        "发行主体",
        "发行金额上限",
        "发行金额",
        "发行规模",
        "注册金额上限",
        "注册金额",
        "主体信用评级",
        "主体评级",
        "债项信用评级",
        "债项评级",
        "受托管理人",
        "牵头主承销商",
        "联席主承销商",
        "簿记管理人",
        "资产负债率",
        "净利润",
        "合并口径",
        "交易结构",
        "发行股份购买资产",
        "募集配套资金",
        "标的公司",
        "控股股东",
        "违约",
        "补偿",
        "违约责任",
        "资产减值补偿",
    ]
    if domain == "financial_reports":
        role_terms.extend(["营业收入", "营业总收入", "研发投入", "研发费用", "归母净利润", "现金分红", "经营活动现金流"])
    return [term for term in role_terms if term in text]


def _option_numeric_terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in re.findall(r"(?:主体信用评级AAA|主体信用评级AA\+|主体信用评级AA|主体评级AAA|主体评级AA\+|主体评级AA|AAA|AA\+|AA|A\+)", text):
        if match not in terms:
            terms.append(match)
    for match in re.findall(r"\d[\d,]*(?:\.\d+)?\s*(?:亿元|万元|元|%|％|年|级|百分点)?", text):
        value = match.strip()
        if value and value not in terms:
            terms.append(value)
    return terms


def _option_predicate_terms(text: str) -> list[str]:
    predicates = [
        "高于",
        "低于",
        "超过",
        "不超过",
        "均为",
        "均达到",
        "明确指定",
        "指定",
        "属于",
        "涉及",
        "提及",
        "披露",
        "列出",
        "提供",
        "显示",
        "成立",
        "符合",
    ]
    return [term for term in predicates if term in text]


def _option_target_docs(domain: str, question: dict[str, Any], letter: str, expected_docs: list[str]) -> list[str]:
    if domain != "insurance" or not expected_docs:
        return []
    option_letters = sorted(str(key).upper() for key in (question.get("options") or {}))
    if len(option_letters) != len(expected_docs):
        return []
    try:
        index = option_letters.index(letter)
    except ValueError:
        return []
    return [expected_docs[index]]


def _fallback_option_doc_items(
    question: dict[str, Any],
    evidence_items: list[dict[str, Any]],
    doc_id: str,
) -> list[dict[str, Any]]:
    terms = _expand_query_terms(question, build_query(question))[:40]
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in evidence_items:
        if str(item.get("doc_id") or "") != doc_id:
            continue
        text = _evidence_item_search_text(item)
        matched_terms = [term for term in terms if term in text]
        score = _keyword_chunk_score(text, matched_terms) if matched_terms else 0.0
        scored.append(
            (
                score,
                {
                    "chunk_id": str(item.get("chunk_id") or ""),
                    "doc_id": doc_id,
                    "page_start": item.get("page_start"),
                    "page_end": item.get("page_end"),
                    "block_type": str(item.get("block_type") or ""),
                    "section_title": str(item.get("section_title") or ""),
                    "clause_no": str(item.get("clause_no") or ""),
                    "row_key": str(item.get("row_key") or ""),
                    "matched_terms": matched_terms[:10],
                    "text": _short_snippet(str(item.get("text") or ""), matched_terms, max_chars=260),
                    "score": round(float(score), 4),
                    "fallback": True,
                },
            )
        )
    scored.sort(key=lambda row: row[0], reverse=True)
    return [payload for _, payload in scored]


def _candidate_match_reason(candidate: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if candidate.get("matched_terms"):
        reasons.append("option_keyword_match")
    if candidate.get("fallback"):
        reasons.append("doc_fallback_candidate")
    if candidate.get("block_type") in {"table_row", "table_record"}:
        reasons.append("table_record")
    if candidate.get("clause_no") or candidate.get("article_no"):
        reasons.append("article")
    if not reasons:
        reasons.append("retrieval_candidate")
    return reasons


def _candidate_coverage(question_plan: dict[str, Any], retrieval_matrix: dict[str, Any]) -> dict[str, Any]:
    required_docs = [str(doc_id) for doc_id in question_plan.get("required_docs") or []]
    covered_docs = sorted(
        {
            str(doc_id)
            for option_docs in retrieval_matrix.values()
            if isinstance(option_docs, dict)
            for doc_id, bucket in option_docs.items()
            if isinstance(bucket, dict) and bucket.get("candidates")
        }
    )
    return {
        "required_docs": required_docs,
        "covered_docs": covered_docs,
        "missing_docs": [doc_id for doc_id in required_docs if doc_id not in covered_docs],
    }


def _candidate_matrix_gaps(question_plan: dict[str, Any], retrieval_matrix: dict[str, Any]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    coverage = _candidate_coverage(question_plan, retrieval_matrix)
    for doc_id in coverage["missing_docs"]:
        gaps.append(
            {
                "type": "question_doc_missing_candidate",
                "level": "question_doc",
                "doc_id": doc_id,
                "reason": "explicit doc has no candidate evidence in entire question",
            }
        )
    option_doc_targets = question_plan.get("option_doc_targets") or {}
    for letter, target_docs in sorted(option_doc_targets.items()):
        for doc_id in target_docs:
            bucket = ((retrieval_matrix.get(str(letter).upper()) or {}).get(str(doc_id)) or {})
            if not bucket.get("candidates"):
                gaps.append(
                    {
                        "type": "option_doc_missing_candidate",
                        "level": "option_doc",
                        "option": str(letter).upper(),
                        "doc_id": str(doc_id),
                        "reason": "option appears to refer to this doc but no candidate evidence was found",
                    }
                )
    return gaps


def _domain_option_term_groups(domain: str, option_text_value: str) -> list[tuple[str, ...]]:
    groups_by_domain: dict[str, list[tuple[str, tuple[str, ...]]]] = {
        "insurance": [
            ("保险责任", ("保险责任",)),
            ("责任免除", ("责任免除",)),
            ("等待期", ("等待期",)),
            ("免赔额", ("免赔额",)),
            ("赔付比例", ("赔付比例", "给付比例")),
            ("身故保险金", ("身故保险金",)),
            ("养老年金", ("养老年金",)),
            ("保单账户价值", ("保单账户价值",)),
            ("个人账户价值", ("个人账户价值",)),
            ("现金价值", ("现金价值",)),
            ("宽限期", ("宽限期",)),
            ("效力中止", ("效力中止", "合同效力中止")),
            ("犹豫期", ("犹豫期",)),
            ("施救费用", ("施救费用",)),
            ("退保费用", ("退保费用", "退保手续费")),
            ("已交保险费", ("已交保险费", "已交保费")),
        ],
        "regulatory": [
            ("股东会", ("股东会", "股东大会")),
            ("普通决议", ("普通决议",)),
            ("特别决议", ("特别决议",)),
            ("变更募集资金用途", ("变更募集资金用途",)),
            ("独立董事", ("独立董事",)),
            ("公司章程", ("公司章程", "本章程")),
            ("资产负债率", ("资产负债率",)),
            ("行政处罚", ("行政处罚",)),
            ("生效", ("施行", "生效")),
        ],
        "financial_contracts": [
            ("合同金额", ("合同金额", "交易价格", "发行规模")),
            ("付款", ("付款", "支付")),
            ("交付", ("交付", "交割")),
            ("违约责任", ("违约责任", "违约金")),
            ("违约利息", ("违约利息", "违约金", "本金和利息")),
            ("本金和利息", ("本金和利息",)),
            ("兑付日", ("兑付日",)),
            ("资产减值补偿", ("资产减值补偿", "减值测试报告")),
            ("通知期限", ("通知期限", "通知之日起", "出具之日起10日")),
        ],
        "financial_reports": [
            ("营业收入", ("营业收入", "营业总收入", "营收")),
            ("营收", ("营业收入", "营业总收入", "营收")),
            ("研发投入", ("研发投入",)),
            ("研发投入占营业收入", ("研发投入占营业收入",)),
            ("经营活动产生的现金流量净额", ("经营活动产生的现金流量净额", "经营活动现金流净额")),
            ("经营活动现金流净额", ("经营活动产生的现金流量净额", "经营活动现金流净额")),
            ("归属于上市公司股东的净利润", ("归属于上市公司股东的净利润", "归属于母公司股东的净利润")),
            ("归母净利润", ("归母净利润", "归属于上市公司股东的净利润", "归属于母公司股东的净利润")),
            ("净利润", ("净利润",)),
            ("现金分红", ("现金分红", "分红")),
            ("毛利率", ("毛利率",)),
        ],
        "research": [
            ("投资建议", ("投资建议",)),
            ("风险提示", ("风险提示",)),
            ("渠道贡献", ("渠道贡献", "渠道保费", "银保渠道")),
            ("保费贡献率", ("保费贡献率", "保费贡献", "银保渠道")),
            ("服务零售", ("服务零售", "服务消费", "服务性消费")),
            ("商品零售", ("商品零售", "社会消费品零售", "限额以上商品零售")),
            ("居民可支配收入", ("居民可支配收入", "居民可支配收入增长", "居民人均可支配")),
            ("居民收入增速", ("居民收入增速", "收入增速")),
            ("人均名义GDP", ("人均名义GDP", "人均名义 GDP", "人均GDP")),
            ("市场规模", ("市场规模",)),
            ("复合增速", ("复合增速",)),
            ("芯片定制服务", ("芯片定制服务", "芯片定制")),
            ("检测规则", ("检测规则", "内置检测规则")),
            ("解析规则", ("解析规则",)),
        ],
    }
    return [terms for trigger, terms in groups_by_domain.get(domain, []) if trigger in option_text_value]


def _short_snippet(text: str, terms: list[str], *, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    positions = [text.find(term) for term in terms if term and text.find(term) >= 0]
    if not positions:
        return text[:max_chars].rstrip()
    center = min(positions)
    start = max(0, center - max_chars // 3)
    end = min(len(text), start + max_chars)
    return text[start:end].strip()


def _keyword_chunk_score(text: str, terms: list[str]) -> float:
    score = 0.0
    for term in terms:
        count = text.count(term)
        if count:
            score += count * (2.0 + min(len(term), 14))
    if score <= 0:
        return 0.0
    metric_bonus_terms = ("营业收入", "研发投入", "现金流量", "净利润", "同比", "增长", "分红")
    metric_bonus_terms = (
        "营业收入",
        "研发投入",
        "现金流量",
        "净利润",
        "同比",
        "增长",
        "分红",
        "免赔额",
        "等待期",
        "责任免除",
        "违约",
        "报告",
        "处罚",
    )
    score += sum(6.0 for term in metric_bonus_terms if term in text)
    return score / max(len(text) ** 0.35, 1.0)


def _is_low_value_keyword(term: str) -> bool:
    if re.fullmatch(r"\d+(?:\.\d+)?", term):
        return True
    if re.fullmatch(r"20\d{2}", term):
        return True
    if term in {"比亚迪", "美的集团", "宁德时代", "中国建筑", "中国移动", "年度报告"}:
        return True
    return False


def _unit_score(unit: str, terms: list[str]) -> float:
    score = 0.0
    for term in terms:
        if term in unit:
            score += min(len(term), 8)
    return score / max(len(unit), 40)


def _needs_neighbor_context(block_type: str, text: str) -> bool:
    if block_type in {"clause", "article", "table_row", "table_record", "mixed"}:
        return True
    return any(marker in text for marker in ("【条款】", "表格行", "责任免除", "等待期", "违约责任", "监管", "处罚"))


def _extract_explicit_doc_ids(question: dict[str, Any]) -> list[str]:
    text = build_query(question)
    domain = str(question.get("domain") or "")
    doc_ids: list[str] = []
    if domain == "financial_contracts":
        for match in re.findall(r"fc[_-]?text[_-]?0*([0-9]{1,2})", text, flags=re.IGNORECASE):
            doc_ids.append(f"text{int(match):02d}")
    if domain == "research":
        for match in re.findall(r"pack2[_-]?text[_-]?0*([0-9]{1,2})", text, flags=re.IGNORECASE):
            doc_ids.append(f"pack2_text{int(match):02d}")
    if domain == "insurance":
        for match in re.findall(r"(?:保险|条款|产品)?(?:^|[^0-9])([1-9]|1[0-6])(?:号|\\b)", text):
            doc_ids.append(str(int(match)))
    return list(dict.fromkeys(doc_ids))
