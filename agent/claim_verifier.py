from __future__ import annotations

import re
from typing import Any

from agent.answer_utils import AnswerFormatError, normalize_answer
from agent.finance_solver import classify_financial_candidate
from agent.question_plan import option_has_claim_negation


CLASSIFIED_BUCKETS = [
    "candidates",
    "direct_support",
    "direct_refute",
    "derived_support",
    "derived_refute",
    "support",
    "refute",
    "neutral",
]

STRONG_CORE_EVIDENCE_SCORE = 0.7
CORE_CLASSIFICATION_REASONS = {
    "number_match",
    "article_ref_match",
    "financial_number_match",
}
CORE_FINANCIAL_RESULTS = {
    "matched_claim_number",
}


def classify_retrieval_evidence_matrix(
    question: dict[str, Any],
    question_plan: dict[str, Any],
    retrieval_matrix: dict[str, Any],
) -> dict[str, Any]:
    classified: dict[str, Any] = {}
    for letter, option_docs in sorted(retrieval_matrix.items()):
        letter = str(letter).upper()
        option_text = str((question.get("options") or {}).get(letter) or "")
        classified[letter] = {}
        for doc_id, bucket in sorted((option_docs or {}).items()):
            candidates = list((bucket or {}).get("candidates") or [])
            classified[letter][doc_id] = classify_evidence_items(question_plan, option_text, str(doc_id), candidates)
    return classified


def classify_evidence_items(
    question_plan: dict[str, Any],
    option_text: str,
    doc_id: str,
    candidates: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    bucket = {key: [] for key in CLASSIFIED_BUCKETS}
    bucket["candidates"] = [dict(candidate) for candidate in candidates]
    for candidate in candidates:
        label, extra = classify_candidate(question_plan, option_text, candidate)
        payload = dict(candidate)
        payload.update(extra)
        if label in {"direct_support", "derived_support"}:
            bucket[label].append(payload)
            bucket["support"].append(payload)
        elif label in {"direct_refute", "derived_refute"}:
            bucket[label].append(payload)
            bucket["refute"].append(payload)
        else:
            bucket["neutral"].append(payload)
    return bucket


def classify_candidate(
    question_plan: dict[str, Any],
    option_text: str,
    candidate: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if question_plan.get("domain") == "financial_reports":
        financial_label = classify_financial_candidate(option_text, candidate)
        if financial_label is not None:
            return financial_label

    text = " ".join(
        str(candidate.get(key) or "")
        for key in ["section_title", "clause_no", "row_key", "text"]
    )
    matched_terms = [str(item) for item in candidate.get("matched_terms") or [] if str(item)]
    if not matched_terms:
        matched_terms = _claim_terms(option_text, text)
    if not matched_terms:
        return "neutral", {"classification_reason": "no_claim_term_match", "classification_score": 0.0}

    option_numbers = _numbers(option_text)
    evidence_numbers = _numbers(text)
    if option_numbers and evidence_numbers:
        if any(_number_equal(left, right) for left in option_numbers for right in evidence_numbers):
            return "direct_support", _classification_payload("number_match", candidate, 0.86)
        if len(matched_terms) >= 1:
            return "direct_refute", _classification_payload("number_conflict", candidate, 0.74)

    claim_refs = _article_refs(option_text)
    evidence_refs = _article_refs(text)
    if claim_refs:
        if set(claim_refs).intersection(evidence_refs):
            return "direct_support", _classification_payload("article_ref_match", candidate, 0.83)
        if evidence_refs and matched_terms:
            return "direct_refute", _classification_payload("article_ref_conflict", candidate, 0.7)

    if option_has_claim_negation(option_text):
        if any(word in text for word in ["不得", "不能", "无需", "不应", "禁止", "未", "无"]):
            return "direct_support", _classification_payload("negation_alignment", candidate, 0.78)
        return "neutral", _classification_payload("claim_negation_without_refute", candidate, 0.42)

    raw_score = float(candidate.get("score") or 0.0)
    if len(set(matched_terms)) >= 2:
        return "direct_support", _classification_payload("multi_term_match", candidate, max(0.75, min(raw_score / 10.0, 0.9)))
    if raw_score >= 5.0:
        return "direct_support", _classification_payload("strong_single_term_match", candidate, min(raw_score / 10.0, 0.78))
    return "neutral", _classification_payload("weak_single_term_match", candidate, min(raw_score / 10.0, 0.49))


def normalize_option_judgement(
    raw: Any,
    question: dict[str, Any],
    classified_matrix: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    normalized: dict[str, Any] = {}
    valid_by_option = {
        letter: {
            "support": set(_evidence_ids(_collect_option_items(doc_buckets, "support"))),
            "refute": set(_evidence_ids(_collect_option_items(doc_buckets, "refute"))),
        }
        for letter, doc_buckets in classified_matrix.items()
    }
    for letter in sorted(str(key).upper() for key in question.get("options") or {}):
        value = raw.get(letter) if letter in raw else raw.get(letter.lower())
        judgement = _raw_judgement_value(value)
        confidence = _raw_confidence(value)
        support_ids = _raw_id_list(value, "support_evidence_ids", "evidence_ids")
        refute_ids = _raw_id_list(value, "refute_evidence_ids")
        support_ids = [item for item in support_ids if item in valid_by_option.get(letter, {}).get("support", set())]
        refute_ids = [item for item in refute_ids if item in valid_by_option.get(letter, {}).get("refute", set())]

        if judgement == "true" and not support_ids:
            support_ids = list(valid_by_option.get(letter, {}).get("support", set()))[:3]
        if judgement == "false" and not refute_ids:
            refute_ids = list(valid_by_option.get(letter, {}).get("refute", set()))[:3]
        if judgement == "true" and not support_ids:
            judgement = "unknown"
        if judgement == "false" and not refute_ids:
            judgement = "unknown"

        normalized[letter] = {
            "judgement": judgement,
            "confidence": confidence if judgement != "unknown" else 0.0,
            "support_evidence_ids": support_ids,
            "refute_evidence_ids": refute_ids,
            "reason": str(value.get("reason") if isinstance(value, dict) else ""),
        }
    return normalized


def normalize_llm_option_judgement(
    raw: Any,
    question: dict[str, Any],
    valid_evidence_ids: set[str],
) -> dict[str, Any] | None:
    """Validate an LLM-produced option_judgement against the real evidence pool.

    Unlike ``normalize_option_judgement``, evidence references are checked against
    every evidence_id actually shown to the model, not against the rule
    classifier's own support/refute buckets -- the rule classifier is a hint
    generator here, not a gatekeeper of what the model is allowed to cite.
    """
    if not isinstance(raw, dict):
        return None
    normalized: dict[str, Any] = {}
    for letter in sorted(str(key).upper() for key in question.get("options") or {}):
        value = raw.get(letter) if letter in raw else raw.get(letter.lower())
        judgement = _raw_judgement_value(value)
        confidence = _raw_confidence(value)
        evidence_ids = _raw_id_list(value, "evidence_ids", "support_evidence_ids", "refute_evidence_ids")
        evidence_ids = [item for item in evidence_ids if item in valid_evidence_ids]
        reasoning = ""
        if isinstance(value, dict):
            reasoning = str(value.get("reasoning") or value.get("reason") or "")
        if judgement in {"true", "false"} and not evidence_ids:
            judgement = "unknown"
            confidence = 0.0
        normalized[letter] = {
            "judgement": judgement,
            "confidence": confidence if judgement != "unknown" else 0.0,
            "support_evidence_ids": evidence_ids if judgement == "true" else [],
            "refute_evidence_ids": evidence_ids if judgement == "false" else [],
            "reason": reasoning,
        }
    return normalized


def derive_answer(
    question: dict[str, Any],
    question_plan: dict[str, Any],
    option_judgement: dict[str, Any],
    *,
    fallback_answer: str | None = None,
) -> dict[str, Any]:
    letters = sorted(str(key).upper() for key in question.get("options") or {})
    desired = "false" if question_plan.get("is_negative") else "true"
    evidence_key = "refute_evidence_ids" if desired == "false" else "support_evidence_ids"
    candidates = [
        (letter, _confidence(option_judgement.get(letter)))
        for letter in letters
        if _judgement(option_judgement.get(letter)) == desired
        and _ids(option_judgement.get(letter), evidence_key)
    ]
    if not candidates:
        status = _no_candidate_status(desired, option_judgement, letters)
        return _fallback_result(question, status, f"no {desired} option with required evidence", fallback_answer, option_judgement, desired=desired)

    if question.get("answer_format") == "multi":
        unknown_letters = [letter for letter in letters if _judgement(option_judgement.get(letter)) == "unknown"]
        if unknown_letters:
            return _fallback_result(
                question,
                "partial_true_with_unknown" if desired == "true" else "partial_false_with_unknown",
                "multi answer has verified options but also unknown options",
                fallback_answer,
                option_judgement,
                desired=desired,
            )
        answer = "".join(letter for letter, _ in candidates)
        return _ok_result(question, answer, "multi answer derived from verified option judgement")

    ranked = sorted(candidates, key=lambda item: (item[1], item[0]), reverse=True)
    if len(ranked) >= 2 and ranked[0][1] - ranked[1][1] <= 0.03:
        return _fallback_result(question, "tie", "top option confidence scores are too close", fallback_answer, option_judgement, desired=desired)
    if ranked[0][1] < 0.5:
        return _fallback_result(question, "weak_support", "top option has weak verified evidence", fallback_answer, option_judgement, desired=desired)
    return _ok_result(question, ranked[0][0], "single answer derived from verified option judgement")


def compute_matrix_gaps(
    question_plan: dict[str, Any],
    retrieval_matrix: dict[str, Any],
    classified_matrix: dict[str, Any],
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for doc_id in question_plan.get("required_docs") or []:
        if _doc_candidate_count(retrieval_matrix, doc_id) <= 0:
            gaps.append(
                {
                    "type": "question_doc_missing_candidate",
                    "level": "question_doc",
                    "doc_id": doc_id,
                    "reason": "explicit doc has no candidate evidence in entire question",
                }
            )
        elif _doc_verified_count(classified_matrix, doc_id) <= 0:
            gaps.append(
                {
                    "type": "question_doc_weak_verified",
                    "level": "question_doc",
                    "doc_id": doc_id,
                    "reason": "explicit doc has candidates but no verified support/refute evidence",
                }
            )

    option_doc_targets = question_plan.get("option_doc_targets") or {}
    for letter, targets in sorted(option_doc_targets.items()):
        for doc_id in targets:
            if _option_doc_candidate_count(retrieval_matrix, letter, doc_id) <= 0:
                gaps.append(
                    {
                        "type": "option_doc_missing_candidate",
                        "level": "option_doc",
                        "option": letter,
                        "doc_id": doc_id,
                        "reason": "option appears to refer to this doc but no candidate evidence was found",
                    }
                )
            elif _option_doc_verified_count(classified_matrix, letter, doc_id) <= 0:
                gaps.append(
                    {
                        "type": "option_doc_weak_verified",
                        "level": "option_doc",
                        "option": letter,
                        "doc_id": doc_id,
                        "reason": "option has candidates for this doc but no verified support/refute evidence",
                    }
                )
    return gaps


def build_dual_evidence_summary(question_plan: dict[str, Any], classified_matrix: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for letter, doc_buckets in sorted(classified_matrix.items()):
        letter = str(letter).upper()
        support = _collect_option_items(doc_buckets, "support")
        refute = _collect_option_items(doc_buckets, "refute")
        core_support = [item for item in support if _is_core_evidence(item)]
        core_refute = [item for item in refute if _is_core_evidence(item)]
        support_score = _max_score(support)
        refute_score = _max_score(refute)
        core_support_score = _max_score(core_support)
        core_refute_score = _max_score(core_refute)
        strong_core_support = core_support_score >= STRONG_CORE_EVIDENCE_SCORE
        strong_core_refute = core_refute_score >= STRONG_CORE_EVIDENCE_SCORE
        summary[letter] = {
            "support_count": len(support),
            "refute_count": len(refute),
            "support_score": round(float(support_score), 4),
            "refute_score": round(float(refute_score), 4),
            "core_support_count": len(core_support),
            "core_refute_count": len(core_refute),
            "core_support_score": round(float(core_support_score), 4),
            "core_refute_score": round(float(core_refute_score), 4),
            "has_core_support": bool(core_support),
            "has_core_refute": bool(core_refute),
            "has_strong_core_support": strong_core_support,
            "has_strong_core_refute": strong_core_refute,
            "contradiction": bool(support and refute),
            "core_contradiction": bool(core_support and core_refute),
            "strong_core_contradiction": bool(strong_core_support and strong_core_refute),
            "support_evidence_ids": _evidence_ids(support),
            "refute_evidence_ids": _evidence_ids(refute),
            "core_support_evidence_ids": _evidence_ids(core_support),
            "core_refute_evidence_ids": _evidence_ids(core_refute),
            "top_support_reasons": _top_reasons(support),
            "top_refute_reasons": _top_reasons(refute),
            "domain": str(question_plan.get("domain") or ""),
        }
    return summary


def prune_answer_by_counter_evidence(
    question: dict[str, Any],
    question_plan: dict[str, Any],
    classified_matrix: dict[str, Any],
    answer: str,
) -> dict[str, Any]:
    normalized = normalize_answer(answer, question["answer_format"], question["options"])
    summary = build_dual_evidence_summary(question_plan, classified_matrix)
    result = {
        "original_answer": normalized,
        "pruned_answer": normalized,
        "changed": False,
        "applied": False,
        "skipped_reason": "",
        "removed_options": [],
        "kept_contradictions": [],
        "counter_evidence_direction": "support" if question_plan.get("is_negative") else "refute",
    }
    if question.get("answer_format") != "multi":
        result["skipped_reason"] = "non_multi_answer"
        return result

    counter_key = "support" if question_plan.get("is_negative") else "refute"
    shield_key = "refute" if question_plan.get("is_negative") else "support"
    kept: list[str] = []
    removed: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []
    for letter in normalized:
        row = summary.get(letter, {})
        has_counter = bool(row.get(f"has_strong_core_{counter_key}"))
        has_shield = bool(row.get(f"has_strong_core_{shield_key}"))
        if has_counter and has_shield:
            kept.append(letter)
            contradictions.append(
                {
                    "option": letter,
                    "counter_direction": counter_key,
                    "counter_score": row.get(f"core_{counter_key}_score", 0.0),
                    "shield_score": row.get(f"core_{shield_key}_score", 0.0),
                    "counter_evidence_ids": row.get(f"core_{counter_key}_evidence_ids", []),
                    "shield_evidence_ids": row.get(f"core_{shield_key}_evidence_ids", []),
                    "reason": "strong_core_contradiction_kept",
                }
            )
            continue
        if has_counter:
            removed.append(
                {
                    "option": letter,
                    "counter_direction": counter_key,
                    "counter_score": row.get(f"core_{counter_key}_score", 0.0),
                    "counter_evidence_ids": row.get(f"core_{counter_key}_evidence_ids", []),
                    "reason": "strong_core_counter_evidence",
                }
            )
            continue
        kept.append(letter)

    result["removed_options"] = removed
    result["kept_contradictions"] = contradictions
    if not removed:
        result["skipped_reason"] = "no_strong_core_counter_evidence"
        return result
    if not kept:
        result["skipped_reason"] = "pruning_would_empty_answer"
        return result

    pruned_answer = normalize_answer("".join(kept), question["answer_format"], question["options"])
    if len(normalized) > 1 and len(pruned_answer) < 2:
        result["skipped_reason"] = "pruning_would_collapse_multi_to_single"
        return result
    result["pruned_answer"] = pruned_answer
    result["changed"] = pruned_answer != normalized
    result["applied"] = result["changed"]
    if not result["changed"]:
        result["skipped_reason"] = "answer_unchanged_after_pruning"
    return result


def _classification_payload(reason: str, candidate: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "classification_reason": reason,
        "classification_score": round(float(score), 4),
    }


def _claim_terms(option_text: str, evidence_text: str) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9_.-]+|[\u4e00-\u9fff]{2,14}", option_text)
    ignored = {"正确", "错误", "下列", "哪些", "关于", "根据", "依据", "以下", "选项", "说法"}
    unique: list[str] = []
    for term in terms:
        if term in ignored or len(term) < 2:
            continue
        if term in evidence_text and term not in unique:
            unique.append(term)
    return unique[:12]


def _numbers(text: str) -> list[float]:
    values: list[float] = []
    for raw in re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(text or "")):
        try:
            value = float(raw.replace(",", ""))
            if 1900 <= value <= 2099 and float(value).is_integer():
                continue
            values.append(value)
        except ValueError:
            pass
    return values


def _number_equal(left: float, right: float) -> bool:
    if abs(left - right) <= 0.02:
        return True
    base = max(abs(left), abs(right), 1.0)
    return abs(left - right) / base <= 0.015


def _article_refs(text: str) -> list[str]:
    return re.findall(r"第[一二三四五六七八九十百千万\d]+[条章节款项]", str(text or ""))


def _collect_option_items(doc_buckets: dict[str, Any], key: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not isinstance(doc_buckets, dict):
        return items
    for bucket in doc_buckets.values():
        if isinstance(bucket, dict):
            items.extend(item for item in bucket.get(key, []) if isinstance(item, dict))
    return items


def _evidence_ids(items: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for item in items:
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id and evidence_id not in ids:
            ids.append(evidence_id)
    return ids[:6]


def _top_reasons(items: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for item in sorted(items, key=_item_score, reverse=True):
        reason = str(item.get("classification_reason") or item.get("result") or "")
        if reason and reason not in reasons:
            reasons.append(reason)
        if len(reasons) >= 3:
            break
    return reasons


def _is_core_evidence(item: dict[str, Any]) -> bool:
    if bool(item.get("core_evidence")):
        return True
    reason = str(item.get("classification_reason") or "")
    if reason in CORE_CLASSIFICATION_REASONS:
        return True
    result = str(item.get("result") or "")
    if result in CORE_FINANCIAL_RESULTS:
        return True
    return False


def _item_score(item: dict[str, Any]) -> float:
    return float(item.get("classification_score") or item.get("score") or 0.0)


def _max_score(items: list[dict[str, Any]]) -> float:
    scores = [_item_score(item) for item in items]
    return max(scores) if scores else 0.0


def _raw_judgement_value(value: Any) -> str:
    if isinstance(value, dict):
        raw = str(value.get("judgement") or value.get("label") or "").strip().lower()
    elif isinstance(value, bool):
        return "true" if value else "false"
    else:
        raw = str(value or "").strip().lower()
    if raw in {"true", "support", "supported", "yes", "正确", "是"}:
        return "true"
    if raw in {"false", "refute", "refuted", "no", "错误", "否"}:
        return "false"
    return "unknown"


def _raw_confidence(value: Any) -> float:
    if isinstance(value, dict):
        try:
            return max(0.0, min(1.0, float(value.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            return 0.0
    if isinstance(value, bool):
        return 0.65
    return 0.0


def _raw_id_list(value: Any, *keys: str) -> list[str]:
    if not isinstance(value, dict):
        return []
    result: list[str] = []
    for key in keys:
        raw = value.get(key)
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            for item in raw:
                item = str(item)
                if item and item not in result:
                    result.append(item)
    return result


def _judgement(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("judgement") or "unknown")
    return "unknown"


def _confidence(row: Any) -> float:
    if isinstance(row, dict):
        return float(row.get("confidence") or 0.0)
    return 0.0


def _ids(row: Any, key: str) -> list[str]:
    if isinstance(row, dict):
        raw = row.get(key)
        return [str(item) for item in raw] if isinstance(raw, list) else []
    return []


def _no_candidate_status(desired: str, option_judgement: dict[str, Any], letters: list[str]) -> str:
    labels = [_judgement(option_judgement.get(letter)) for letter in letters]
    if all(label == "unknown" for label in labels):
        return "all_unknown"
    return "no_false" if desired == "false" else "no_true"


def _ok_result(question: dict[str, Any], answer: str, reason: str) -> dict[str, Any]:
    return {
        "answer": normalize_answer(answer, question["answer_format"], question["options"]),
        "derive_status": "ok",
        "derive_reason": reason,
        "need_repair": False,
        "answer_source": "derived",
    }


def _fallback_result(
    question: dict[str, Any],
    status: str,
    reason: str,
    fallback_answer: str | None,
    option_judgement: dict[str, Any],
    *,
    desired: str = "true",
) -> dict[str, Any]:
    answer = _fallback_answer(question, fallback_answer, option_judgement, desired=desired)
    return {
        "answer": answer,
        "derive_status": status,
        "derive_reason": reason,
        "need_repair": True,
        "answer_source": "fallback",
    }


def _fallback_answer(
    question: dict[str, Any],
    fallback_answer: str | None,
    option_judgement: dict[str, Any],
    *,
    desired: str,
) -> str:
    if question.get("answer_format") == "multi":
        return _multi_expanded_fallback_answer(question, fallback_answer, option_judgement, desired=desired)
    if fallback_answer:
        try:
            return normalize_answer(fallback_answer, question["answer_format"], question["options"])
        except AnswerFormatError:
            pass
    letters = sorted(str(key).upper() for key in question.get("options") or {})
    ranked = sorted(
        ((letter, _confidence(option_judgement.get(letter))) for letter in letters),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    if ranked and ranked[0][1] > 0:
        return normalize_answer(ranked[0][0], question["answer_format"], question["options"])
    return normalize_answer(letters[0], question["answer_format"], question["options"])


def _multi_expanded_fallback_answer(
    question: dict[str, Any],
    fallback_answer: str | None,
    option_judgement: dict[str, Any],
    *,
    desired: str,
) -> str:
    letters = sorted(str(key).upper() for key in question.get("options") or {})
    if fallback_answer:
        try:
            normalized = normalize_answer(fallback_answer, question["answer_format"], question["options"])
            if len(normalized) > 1:
                return normalized
        except AnswerFormatError:
            pass

    selected: list[str] = []
    for letter in letters:
        label = _judgement(option_judgement.get(letter))
        if label == desired:
            selected.append(letter)
        elif desired == "true" and label == "unknown":
            # Positive multi-choice fallback is recall-oriented: keep unknown
            # options instead of collapsing to a single verified option.
            selected.append(letter)
    if selected:
        return normalize_answer("".join(selected), question["answer_format"], question["options"])
    return normalize_answer("".join(letters), question["answer_format"], question["options"])


def _doc_candidate_count(matrix: dict[str, Any], doc_id: str) -> int:
    return sum(_option_doc_candidate_count(matrix, letter, doc_id) for letter in matrix)


def _doc_verified_count(matrix: dict[str, Any], doc_id: str) -> int:
    return sum(_option_doc_verified_count(matrix, letter, doc_id) for letter in matrix)


def _option_doc_candidate_count(matrix: dict[str, Any], letter: str, doc_id: str) -> int:
    bucket = ((matrix.get(str(letter).upper()) or {}).get(str(doc_id)) or {})
    return len(bucket.get("candidates") or []) if isinstance(bucket, dict) else 0


def _option_doc_verified_count(matrix: dict[str, Any], letter: str, doc_id: str) -> int:
    bucket = ((matrix.get(str(letter).upper()) or {}).get(str(doc_id)) or {})
    if not isinstance(bucket, dict):
        return 0
    return len(bucket.get("support") or []) + len(bucket.get("refute") or [])
