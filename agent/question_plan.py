from __future__ import annotations

import re
from typing import Any


NEGATIVE_QUESTION_PATTERNS = [
    "不正确的是",
    "错误的是",
    "不符合的是",
    "不属于的是",
    "不包括的是",
    "除外的是",
    "说法错误",
    "说法不正确",
    "哪项不属于",
    "哪项不符合",
]

CLAIM_NEGATION_WORDS = [
    "不得",
    "不能",
    "无需",
    "不应",
    "禁止",
    "未",
    "无",
    "不得不",
]


FINANCE_CALC_TERMS = [
    "同比",
    "增长",
    "下降",
    "增速",
    "占比",
    "比例",
    "超过",
    "高于",
    "低于",
    "大于",
    "小于",
    "减少",
    "增加",
    "现金分红",
    "每股",
    "ROE",
    "净资产收益率",
]


DOC_ID_COMPANY_TERMS: dict[str, tuple[str, ...]] = {
    "byd": ("比亚迪",),
    "catl": ("宁德时代",),
    "midea": ("美的集团", "美的"),
    "cscec": ("中国建筑",),
    "chinamobile": ("中国移动",),
}


def build_question_plan(
    question: dict[str, Any],
    *,
    candidate_doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    required_docs = [str(item) for item in question.get("doc_ids") or []]
    doc_ids = candidate_doc_ids or required_docs
    question_text = str(question.get("question") or "")
    all_text = " ".join(
        [
            question_text,
            str(question.get("type") or ""),
            " ".join(str(value) for value in (question.get("options") or {}).values()),
        ]
    )

    is_negative = any(pattern in question_text for pattern in NEGATIVE_QUESTION_PATTERNS)
    is_multi = str(question.get("answer_format") or "") == "multi"
    is_finance_calc = str(question.get("domain") or "") == "financial_reports" and any(term in all_text for term in FINANCE_CALC_TERMS)
    option_doc_targets = _option_doc_targets(question, doc_ids)

    risk_flags: list[str] = []
    if is_negative:
        risk_flags.append("negative_question")
    if is_multi:
        risk_flags.append("multi_choice")
    if len(required_docs) >= 2:
        risk_flags.append("multi_document")
    if is_finance_calc:
        risk_flags.append("finance_calc")

    return {
        "qid": str(question.get("qid") or ""),
        "domain": str(question.get("domain") or ""),
        "answer_format": str(question.get("answer_format") or ""),
        "is_negative": is_negative,
        "is_multi": is_multi,
        "is_finance_calc": is_finance_calc,
        "required_docs": required_docs,
        "candidate_doc_ids": [str(item) for item in doc_ids],
        "option_doc_targets": option_doc_targets,
        "risk_flags": risk_flags,
        "negative_question_patterns": [pattern for pattern in NEGATIVE_QUESTION_PATTERNS if pattern in question_text],
    }


def option_has_claim_negation(text: str) -> bool:
    return any(word in str(text or "") for word in CLAIM_NEGATION_WORDS)


def _option_doc_targets(question: dict[str, Any], doc_ids: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    options = question.get("options") or {}
    for letter, value in sorted(options.items()):
        option_text = str(value or "")
        targets: list[str] = []
        for doc_id in _ordinal_doc_targets(option_text, doc_ids):
            if doc_id not in targets:
                targets.append(doc_id)
        for doc_id in doc_ids:
            doc_id = str(doc_id)
            if _option_mentions_doc(option_text, doc_id):
                if doc_id not in targets:
                    targets.append(doc_id)
                continue
            if _option_mentions_financial_report_doc(option_text, doc_id):
                if doc_id not in targets:
                    targets.append(doc_id)
        result[str(letter).upper()] = targets
    return result


def _ordinal_doc_targets(option_text: str, doc_ids: list[str]) -> list[str]:
    targets: list[str] = []
    if not doc_ids:
        return targets
    if any(marker in option_text for marker in ["两份文档", "两份材料", "两份募集说明书", "两份报告", "两家公司", "双方"]):
        targets.extend(str(doc_id) for doc_id in doc_ids[:2])
    ordinal_to_index = {
        "第一份": 0,
        "第一份文档": 0,
        "第一份材料": 0,
        "第一份报告": 0,
        "第二份": 1,
        "第二份文档": 1,
        "第二份材料": 1,
        "第二份报告": 1,
        "第三份": 2,
        "第三份文档": 2,
        "第四份": 3,
        "第四份文档": 3,
    }
    for marker, index in ordinal_to_index.items():
        if marker in option_text and index < len(doc_ids):
            targets.append(str(doc_ids[index]))
    return list(dict.fromkeys(targets))


def _option_mentions_doc(option_text: str, doc_id: str) -> bool:
    if len(doc_id) >= 4 and doc_id in option_text:
        return True
    if doc_id.startswith("text") and doc_id in option_text:
        return True
    match = re.fullmatch(r"text0*(\d+)", doc_id)
    if match:
        number = match.group(1).lstrip("0") or "0"
        padded = number.zfill(3)
        return any(pattern in option_text for pattern in [f"fc_text_{padded}", f"fc_text_{number}", f"text_{padded}", f"text{padded}"])
    return False


def _option_mentions_financial_report_doc(option_text: str, doc_id: str) -> bool:
    if not doc_id.startswith("annual_"):
        return False
    parts = doc_id.split("_")
    if len(parts) < 3:
        return False
    company_key = parts[1]
    year = parts[2]
    company_terms = DOC_ID_COMPANY_TERMS.get(company_key, ())
    mentions_company = any(term in option_text for term in company_terms)
    mentions_year = bool(re.search(rf"(?<!\d){re.escape(year)}(?!\d)", option_text))
    return mentions_company or mentions_year
