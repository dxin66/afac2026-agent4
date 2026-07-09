from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any


METRIC_DEFINITIONS: dict[str, dict[str, list[str]]] = {
    "net_profit": {
        "aliases": ["净利润"],
        "forbidden_aliases": ["归母净利润", "归属于上市公司股东的净利润", "扣非归母净利润", "扣除非经常性损益"],
    },
    "parent_net_profit": {
        "aliases": ["归母净利润", "归属于上市公司股东的净利润", "归属于母公司股东的净利润"],
        "forbidden_aliases": ["扣非归母净利润", "扣除非经常性损益", "净利润率"],
    },
    "deducted_parent_net_profit": {
        "aliases": ["扣非归母净利润", "扣除非经常性损益后的净利润", "扣非净利润"],
        "forbidden_aliases": ["净利润", "归母净利润"],
    },
    "operating_revenue": {
        "aliases": ["营业收入"],
        "forbidden_aliases": ["营业总收入", "其他业务收入", "主营业务收入"],
    },
    "total_operating_revenue": {
        "aliases": ["营业总收入", "营收"],
        "forbidden_aliases": ["其他业务收入", "营业外收入"],
    },
    "rd_investment": {
        "aliases": ["研发投入", "研发投入占营业收入"],
        "forbidden_aliases": ["研发费用", "开发支出"],
    },
    "rd_expense": {
        "aliases": ["研发费用"],
        "forbidden_aliases": ["研发投入", "研发投入占营业收入"],
    },
    "operating_cash_flow": {
        "aliases": ["经营活动产生的现金流量净额", "经营活动现金流净额"],
        "forbidden_aliases": ["现金及现金等价物净增加额", "现金及现金等价物"],
    },
    "cash_equivalent_increase": {
        "aliases": ["现金及现金等价物净增加额", "现金及现金等价物"],
        "forbidden_aliases": ["经营活动产生的现金流量净额", "经营活动现金流净额"],
    },
    "basic_eps": {"aliases": ["基本每股收益"], "forbidden_aliases": ["稀释每股收益"]},
    "diluted_eps": {"aliases": ["稀释每股收益"], "forbidden_aliases": ["基本每股收益"]},
    "total_assets": {"aliases": ["总资产"], "forbidden_aliases": ["净资产"]},
    "net_assets": {"aliases": ["净资产"], "forbidden_aliases": ["总资产", "归属于上市公司股东的净资产"]},
    "parent_net_assets": {"aliases": ["归属于上市公司股东的净资产"], "forbidden_aliases": ["总资产"]},
    "debt_ratio": {"aliases": ["资产负债率", "负债率"], "forbidden_aliases": []},
    "monetary_funds": {"aliases": ["货币资金"], "forbidden_aliases": []},
    "accounts_receivable": {"aliases": ["应收账款"], "forbidden_aliases": []},
    "inventory": {"aliases": ["存货"], "forbidden_aliases": []},
    "selling_expense": {"aliases": ["销售费用"], "forbidden_aliases": []},
    "admin_expense": {"aliases": ["管理费用"], "forbidden_aliases": []},
    "finance_expense": {"aliases": ["财务费用"], "forbidden_aliases": []},
    "investment_income": {"aliases": ["投资收益"], "forbidden_aliases": []},
    "operating_cost": {"aliases": ["营业成本"], "forbidden_aliases": []},
    "gross_profit": {"aliases": ["毛利润", "毛利率"], "forbidden_aliases": []},
    "dividend_ratio": {"aliases": ["分红比例", "现金分红占", "现金分红比例"], "forbidden_aliases": []},
    "cash_dividend_per_share": {"aliases": ["每股现金分红", "每 10 股派发", "每10股派发"], "forbidden_aliases": []},
    "roe": {"aliases": ["加权平均净资产收益率", "净资产收益率", "ROE"], "forbidden_aliases": []},
}


COMPARISON_TERMS = ["同比", "增长", "下降", "高于", "低于", "超过", "大于", "小于", "占比", "比例", "减少", "增加"]
PERCENT_REQUIRED_MARGIN = 0.005


@dataclass
class FinanceCalc:
    option: str
    formula_evidence_id: str
    judgement: str
    metric: str
    metric_match: str
    company_match: str
    year_match: str
    unit_normalized: bool
    formula_inputs_complete: bool
    has_forbidden_alias: bool
    has_ambiguous_metric: bool
    has_ambiguous_year: bool
    margin: float
    required_margin: float
    depends_on: list[str] = field(default_factory=list)
    confidence: float = 0.92
    formula: str = ""
    value: float | None = None
    claim_value: float | None = None
    relation: str = ""
    diagnostic_reason: str = ""


def can_hard_override_finance(calc: FinanceCalc) -> bool:
    return (
        calc.metric_match in {"exact", "allowed_alias"}
        and calc.company_match == "exact_or_doc_bound"
        and calc.year_match == "exact"
        and calc.unit_normalized is True
        and calc.formula_inputs_complete is True
        and calc.has_forbidden_alias is False
        and calc.has_ambiguous_metric is False
        and calc.has_ambiguous_year is False
        and calc.margin >= calc.required_margin
        and len(calc.depends_on) >= 1
    )


def build_finance_solver_artifacts(
    question: dict[str, Any],
    retrieval_matrix: dict[str, Any],
) -> dict[str, Any]:
    if str(question.get("domain") or "") != "financial_reports":
        return {
            "raw_table_evidence": [],
            "formula_evidence": [],
            "solver_votes": {},
            "hard_override_trace": [],
        }

    raw_by_id: dict[str, dict[str, Any]] = {}
    facts_by_option: dict[str, list[dict[str, Any]]] = {}
    for letter, option_docs in sorted((retrieval_matrix or {}).items()):
        letter = str(letter).upper()
        option_text = str((question.get("options") or {}).get(letter) or "")
        metric_key = detect_metric(option_text)
        if metric_key is None:
            continue
        facts_by_option.setdefault(letter, [])
        for doc_id, bucket in sorted((option_docs or {}).items()):
            for candidate in list((bucket or {}).get("candidates") or []):
                raw_rows = extract_raw_table_evidence(
                    option=letter,
                    option_text=option_text,
                    metric_key=metric_key,
                    doc_id=str(doc_id),
                    candidate=candidate,
                )
                for raw in raw_rows:
                    raw_by_id.setdefault(str(raw["raw_evidence_id"]), raw)
                    facts_by_option[letter].extend(raw.get("facts") or [])

    formula_evidence: list[dict[str, Any]] = []
    solver_votes: dict[str, list[dict[str, Any]]] = {}
    for letter, option_text in sorted((question.get("options") or {}).items()):
        letter = str(letter).upper()
        calc = build_finance_calc_for_option(
            option=letter,
            option_text=str(option_text or ""),
            facts=facts_by_option.get(letter, []),
            formula_index=len(formula_evidence) + 1,
        )
        if calc is None:
            continue
        valid_raw_ids = set(raw_by_id)
        if any(raw_id not in valid_raw_ids for raw_id in calc.depends_on):
            calc.formula_inputs_complete = False
            calc.diagnostic_reason = "formula_depends_on_missing_raw_table_evidence"
        formula = finance_calc_to_formula_evidence(calc)
        formula_evidence.append(formula)
        can_override = can_hard_override_finance(calc)
        layer = "hard_override" if can_override else ("soft_vote" if calc.formula_inputs_complete else "diagnostic")
        solver_votes.setdefault(letter, []).append(
            {
                "solver": "finance",
                "layer": layer,
                "judgement": calc.judgement if calc.judgement in {"true", "false"} else "unknown",
                "confidence": calc.confidence if can_override else min(calc.confidence, 0.49),
                "formula_evidence_id": calc.formula_evidence_id,
                "evidence_ids": [calc.formula_evidence_id],
                "can_override": can_override,
                "gate_checks": finance_gate_checks(calc),
                "reason": calc.diagnostic_reason or calc.relation,
            }
        )

    return {
        "raw_table_evidence": list(raw_by_id.values()),
        "formula_evidence": formula_evidence,
        "solver_votes": solver_votes,
        "hard_override_trace": [],
    }


def apply_finance_hard_votes(
    option_judgement: dict[str, Any],
    solver_votes: dict[str, list[dict[str, Any]]],
    *,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized_mode = mode if mode in {"dry_run", "enabled"} else "off"
    updated = {str(letter).upper(): dict(value or {}) for letter, value in (option_judgement or {}).items()}
    trace: list[dict[str, Any]] = []
    for letter, votes in sorted((solver_votes or {}).items()):
        hard_votes = [
            vote
            for vote in votes
            if vote.get("solver") == "finance" and vote.get("layer") == "hard_override" and vote.get("can_override")
        ]
        if not hard_votes:
            continue
        vote = hard_votes[0]
        judgement = str(vote.get("judgement") or "unknown")
        before = _trace_judgement(updated.get(letter, {}))
        evidence_ids = [str(item) for item in vote.get("evidence_ids") or [] if str(item)]
        after = {
            "judgement": judgement,
            "confidence": float(vote.get("confidence") or 0.0),
            "support_evidence_ids": evidence_ids if judgement == "true" else [],
            "refute_evidence_ids": evidence_ids if judgement == "false" else [],
            "reason": "finance_hard_override",
        }
        trace.append(
            {
                "option": letter,
                "before": before,
                "after": _trace_judgement(after),
                "mode": normalized_mode,
                "can_override": True,
                "gate_checks": vote.get("gate_checks") or {},
                "formula_evidence_id": vote.get("formula_evidence_id"),
            }
        )
        if normalized_mode == "enabled":
            updated[letter] = after
    return updated, trace


def finance_calc_to_formula_evidence(calc: FinanceCalc) -> dict[str, Any]:
    row = asdict(calc)
    row["gate_checks"] = finance_gate_checks(calc)
    row["can_hard_override"] = can_hard_override_finance(calc)
    return row


def finance_gate_checks(calc: FinanceCalc) -> dict[str, bool]:
    return {
        "metric_match": calc.metric_match in {"exact", "allowed_alias"},
        "company_match": calc.company_match == "exact_or_doc_bound",
        "year_match": calc.year_match == "exact",
        "unit_normalized": calc.unit_normalized is True,
        "formula_inputs_complete": calc.formula_inputs_complete is True,
        "no_forbidden_alias": calc.has_forbidden_alias is False,
        "no_ambiguous_metric": calc.has_ambiguous_metric is False,
        "no_ambiguous_year": calc.has_ambiguous_year is False,
        "margin": calc.margin >= calc.required_margin,
        "depends_on": len(calc.depends_on) >= 1,
    }


def extract_raw_table_evidence(
    *,
    option: str,
    option_text: str,
    metric_key: str,
    doc_id: str,
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    text = " ".join(
        str(candidate.get(key) or "")
        for key in ["section_title", "row_key", "text"]
    )
    if has_forbidden_alias(metric_key, text) or not has_allowed_alias(metric_key, text):
        return []
    columns: dict[str, str] = {}
    facts, columns = _facts_from_structured_table_record(metric_key, doc_id, candidate)

    if not facts:
        cells = _table_cells(str(candidate.get("text") or ""))
        for cell in cells:
            key, value = _split_cell(cell)
            year = _first_year(key) or _first_year(value)
            number_text = value if _first_number_with_unit(value) is not None else cell
            parsed = _first_number_with_unit(number_text)
            if not year or parsed is None:
                continue
            normalized, unit, raw_number = parsed
            columns[year] = number_text.strip()
            facts.append(
                {
                    "year": year,
                    "metric": metric_key,
                    "value": normalized,
                    "raw_value": raw_number,
                    "unit": unit,
                    "doc_id": doc_id,
                }
            )

    if not facts:
        facts = _facts_from_one_column_text(metric_key, doc_id, text)
        columns = {fact["year"]: str(fact.get("raw_text") or fact.get("raw_value") or "") for fact in facts}

    if not facts:
        return []

    source_id = str(candidate.get("evidence_id") or candidate.get("chunk_id") or f"{option}:{doc_id}")
    raw_id = f"raw:{source_id}:1"
    for fact in facts:
        fact["raw_evidence_id"] = raw_id
        fact["source_evidence_id"] = source_id
    return [
        {
            "raw_evidence_id": raw_id,
            "option": option,
            "source_evidence_id": source_id,
            "chunk_id": str(candidate.get("chunk_id") or ""),
            "doc_id": doc_id,
            "page": candidate.get("page_start"),
            "row_key": str(candidate.get("row_key") or ""),
            "metric": metric_key,
            "metric_match": _metric_match(metric_key, text),
            "table_shape": "one_row_multi_columns" if len(facts) >= 2 else "one_column_multi_years",
            "columns": columns,
            "facts": facts,
            "text": str(candidate.get("text") or "")[:500],
        }
    ]


def build_finance_calc_for_option(
    *,
    option: str,
    option_text: str,
    facts: list[dict[str, Any]],
    formula_index: int,
) -> FinanceCalc | None:
    metric_key = detect_metric(option_text)
    if metric_key is None:
        return None
    metric_keys = detect_metrics(option_text)
    years = extract_years(option_text)
    relevant = [fact for fact in facts if fact.get("metric") == metric_key]
    calc = FinanceCalc(
        option=option,
        formula_evidence_id=f"formula:{option}:{formula_index:03d}",
        judgement="unknown",
        metric=metric_key,
        metric_match="exact" if _option_has_exact_alias(metric_key, option_text) else "allowed_alias",
        company_match="exact_or_doc_bound" if relevant else "missing",
        year_match="missing",
        unit_normalized=bool(relevant),
        formula_inputs_complete=False,
        has_forbidden_alias=has_forbidden_alias(metric_key, option_text),
        has_ambiguous_metric=len(metric_keys) > 1,
        has_ambiguous_year=False,
        margin=0.0,
        required_margin=0.0,
        depends_on=[],
        relation="unresolved",
    )
    if not relevant:
        calc.diagnostic_reason = "no_metric_fact"
        return calc

    amount_calc = _amount_or_threshold_calc(calc, option_text, years, relevant)
    if amount_calc is not None:
        return amount_calc
    comparison_calc = _comparison_calc(calc, option_text, years, relevant)
    if comparison_calc is not None:
        return comparison_calc
    calc.diagnostic_reason = "unsupported_finance_relation"
    return calc


def classify_financial_candidate(option_text: str, candidate: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    metric_key = detect_metric(option_text)
    if metric_key is None:
        return None
    text = " ".join(
        str(candidate.get(key) or "")
        for key in ["section_title", "row_key", "text"]
    )
    if has_forbidden_alias(metric_key, text):
        return (
            "neutral",
            {
                "classification_reason": "financial_forbidden_alias",
                "metric": metric_key,
            },
        )
    if not has_allowed_alias(metric_key, text):
        return None

    option_numbers = extract_numbers(option_text)
    evidence_numbers = extract_numbers(text)
    if not evidence_numbers:
        return None

    payload = {
        "source_evidence_ids": [str(candidate.get("evidence_id") or "")],
        "derivation_type": "financial_metric_match",
        "formula_or_rule": "metric alias + numeric/year consistency",
        "computed_value": evidence_numbers[:8],
        "claim_value": option_numbers[:8],
        "metric": metric_key,
    }
    if option_numbers:
        matched = any(_numbers_close(claim, observed) for claim in option_numbers for observed in evidence_numbers)
        payload["result"] = "matched_claim_number" if matched else "conflicting_claim_number"
        return ("derived_support" if matched else "derived_refute", payload)

    if any(term in option_text for term in COMPARISON_TERMS):
        payload["result"] = "metric_numeric_evidence_available"
        return ("derived_support", payload)
    return None


def detect_metric(text: str) -> str | None:
    for key, spec in METRIC_DEFINITIONS.items():
        if any(alias in text for alias in spec["aliases"]):
            return key
    return None


def has_allowed_alias(metric_key: str, text: str) -> bool:
    spec = METRIC_DEFINITIONS.get(metric_key)
    return bool(spec and any(alias in text for alias in spec["aliases"]))


def has_forbidden_alias(metric_key: str, text: str) -> bool:
    spec = METRIC_DEFINITIONS.get(metric_key)
    return bool(spec and any(alias in text for alias in spec["forbidden_aliases"]))


def extract_numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"(-?\d[\d,]*(?:\.\d+)?)\s*(万亿元|亿元|百万元|万元|千元|元|%|％|个百分点|百分点|亿|万)?", str(text or "")):
        parsed = _normalize_number(match.group(1), match.group(2) or "")
        if parsed is None:
            continue
        values.append(parsed[0])
    return values


def extract_years(text: str) -> list[str]:
    years: list[str] = []
    for year in re.findall(r"20\d{2}", str(text or "")):
        if year not in years:
            years.append(year)
    return years


def detect_metrics(text: str) -> list[str]:
    metrics: list[str] = []
    for key, spec in METRIC_DEFINITIONS.items():
        if any(alias in text for alias in spec["aliases"]):
            metrics.append(key)
    return metrics


def _amount_or_threshold_calc(
    calc: FinanceCalc,
    option_text: str,
    years: list[str],
    facts: list[dict[str, Any]],
) -> FinanceCalc | None:
    claim_values = extract_numbers(option_text)
    if not claim_values:
        return None
    if _contains_percent_claim(option_text) and any(term in option_text for term in ["同比", "增长", "下降", "下滑", "减少"]):
        return None
    target_year = _target_year(years, facts)
    if target_year is None:
        calc.has_ambiguous_year = True
        calc.diagnostic_reason = "missing_target_year"
        return calc
    fact = _single_fact_for_year(facts, target_year)
    if fact is None:
        calc.has_ambiguous_year = True
        calc.diagnostic_reason = "missing_or_ambiguous_year_fact"
        return calc

    claim_value = claim_values[-1]
    fact_value = float(fact["value"])
    tolerance = _amount_tolerance(fact_value, claim_value)
    if any(term in option_text for term in ["超过", "高于", "大于"]):
        margin = fact_value - claim_value
        judgement = "true" if margin > 0 else "false"
        required_margin = tolerance
        formula = f"{fact_value} > {claim_value}"
        relation = "amount_greater_than"
    elif any(term in option_text for term in ["低于", "小于", "少于"]):
        margin = claim_value - fact_value
        judgement = "true" if margin > 0 else "false"
        required_margin = tolerance
        formula = f"{fact_value} < {claim_value}"
        relation = "amount_less_than"
    else:
        diff = abs(fact_value - claim_value)
        judgement = "true" if diff <= tolerance else "false"
        margin = tolerance - diff if judgement == "true" else diff - tolerance
        required_margin = 0.0 if judgement == "true" else tolerance
        formula = f"abs({fact_value} - {claim_value})"
        relation = "amount_exact"

    _complete_calc(
        calc,
        judgement=judgement,
        year_match="exact",
        formula_inputs_complete=True,
        margin=margin,
        required_margin=required_margin,
        depends_on=[str(fact["raw_evidence_id"])],
        formula=formula,
        value=fact_value,
        claim_value=claim_value,
        relation=relation,
    )
    return calc


def _comparison_calc(
    calc: FinanceCalc,
    option_text: str,
    years: list[str],
    facts: list[dict[str, Any]],
) -> FinanceCalc | None:
    if not any(term in option_text for term in COMPARISON_TERMS + ["同比", "上年", "双位数", "优于", "下滑"]):
        return None
    current_year, previous_year = _comparison_years(years, facts, option_text)
    if current_year is None or previous_year is None:
        calc.has_ambiguous_year = True
        calc.diagnostic_reason = "missing_comparison_years"
        return calc
    current = _single_fact_for_year(facts, current_year)
    previous = _single_fact_for_year(facts, previous_year)
    if current is None or previous is None:
        calc.has_ambiguous_year = True
        calc.diagnostic_reason = "missing_or_ambiguous_comparison_facts"
        return calc
    current_value = float(current["value"])
    previous_value = float(previous["value"])
    if previous_value == 0:
        calc.diagnostic_reason = "zero_previous_value"
        return calc

    growth = (current_value - previous_value) / abs(previous_value)
    threshold = _percentage_threshold(option_text)
    required_margin = PERCENT_REQUIRED_MARGIN
    relation = "comparison"
    if "双位数" in option_text:
        threshold = 0.10
        margin = growth - threshold
        judgement = "true" if margin >= 0 else "false"
        relation = "double_digit_growth"
    elif threshold is not None and any(term in option_text for term in ["超过", "高于", "大于"]):
        margin = growth - threshold
        judgement = "true" if margin >= 0 else "false"
        relation = "growth_greater_than"
    elif threshold is not None and any(term in option_text for term in ["低于", "小于"]):
        margin = threshold - growth
        judgement = "true" if margin >= 0 else "false"
        relation = "growth_less_than"
    elif threshold is not None and any(term in option_text for term in ["减少", "下降", "下滑"]):
        diff = abs(abs(growth) - threshold)
        judgement = "true" if growth < 0 and diff <= PERCENT_REQUIRED_MARGIN else "false"
        margin = PERCENT_REQUIRED_MARGIN - diff if judgement == "true" else max(diff, abs(growth))
        required_margin = 0.0 if judgement == "true" else PERCENT_REQUIRED_MARGIN
        relation = "negative_growth_exact"
    elif any(term in option_text for term in ["减少", "下降", "下滑", "低于"]):
        margin = -growth
        judgement = "true" if growth < 0 else "false"
        relation = "negative_growth"
    else:
        margin = growth
        judgement = "true" if growth > 0 else "false"
        relation = "positive_growth"

    _complete_calc(
        calc,
        judgement=judgement,
        year_match="exact",
        formula_inputs_complete=True,
        margin=margin,
        required_margin=required_margin,
        depends_on=[str(current["raw_evidence_id"]), str(previous["raw_evidence_id"])],
        formula=f"({current_value} - {previous_value}) / abs({previous_value})",
        value=growth,
        claim_value=threshold,
        relation=relation,
    )
    return calc


def _complete_calc(
    calc: FinanceCalc,
    *,
    judgement: str,
    year_match: str,
    formula_inputs_complete: bool,
    margin: float,
    required_margin: float,
    depends_on: list[str],
    formula: str,
    value: float,
    claim_value: float | None,
    relation: str,
) -> None:
    calc.judgement = judgement
    calc.year_match = year_match
    calc.formula_inputs_complete = formula_inputs_complete
    calc.margin = round(float(margin), 8)
    calc.required_margin = round(float(required_margin), 8)
    calc.depends_on = list(dict.fromkeys(depends_on))
    calc.formula = formula
    calc.value = value
    calc.claim_value = claim_value
    calc.relation = relation


def _table_cells(text: str) -> list[str]:
    value = str(text or "")
    if "表格行:" in value:
        value = value.split("表格行:", 1)[1]
    if "表格记录:" in value:
        value = value.split("表格记录:", 1)[1]
    return [cell.strip() for cell in re.split(r"\s*\|\s*", value) if cell.strip()]


def _facts_from_structured_table_record(metric_key: str, doc_id: str, candidate: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    record = metadata.get("table_record") if isinstance(metadata.get("table_record"), dict) else {}
    values = record.get("values") if isinstance(record.get("values"), dict) else {}
    if not values:
        return [], {}
    facts: list[dict[str, Any]] = []
    columns: dict[str, str] = {}
    row_key = str(record.get("row_key") or candidate.get("row_key") or "")
    for key, value in values.items():
        cell_text = f"{row_key} {key} {value}"
        year = _first_year(str(key)) or _first_year(str(value)) or _first_year(cell_text)
        parsed = _first_number_with_unit(str(value)) or _first_number_with_unit(cell_text)
        if not year or parsed is None:
            continue
        normalized, unit, raw_number = parsed
        columns[year] = str(value).strip()
        facts.append(
            {
                "year": year,
                "metric": metric_key,
                "value": normalized,
                "raw_value": raw_number,
                "unit": unit,
                "doc_id": doc_id,
                "raw_text": cell_text,
            }
        )
    return _dedupe_facts(facts), columns


def _split_cell(cell: str) -> tuple[str, str]:
    if ":" in cell:
        key, value = cell.split(":", 1)
        return key.strip(), value.strip()
    if "：" in cell:
        key, value = cell.split("：", 1)
        return key.strip(), value.strip()
    return "", cell.strip()


def _facts_from_one_column_text(metric_key: str, doc_id: str, text: str) -> list[dict[str, Any]]:
    aliases = METRIC_DEFINITIONS.get(metric_key, {}).get("aliases") or []
    facts: list[dict[str, Any]] = []
    for alias in aliases:
        patterns = [
            rf"(?P<year>20\d{{2}})\s*年?.{{0,30}}?{re.escape(alias)}.{{0,20}}?(?P<num>-?\d[\d,]*(?:\.\d+)?)\s*(?P<unit>万亿元|亿元|百万元|万元|千元|元|%|％|个百分点|百分点|亿|万)?",
            rf"{re.escape(alias)}.{{0,30}}?(?P<year>20\d{{2}})\s*年?.{{0,20}}?(?P<num>-?\d[\d,]*(?:\.\d+)?)\s*(?P<unit>万亿元|亿元|百万元|万元|千元|元|%|％|个百分点|百分点|亿|万)?",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                parsed = _normalize_number(match.group("num"), match.group("unit") or "")
                if parsed is None:
                    continue
                normalized, unit, raw_number = parsed
                facts.append(
                    {
                        "year": match.group("year"),
                        "metric": metric_key,
                        "value": normalized,
                        "raw_value": raw_number,
                        "unit": unit,
                        "doc_id": doc_id,
                        "raw_text": match.group(0),
                    }
                )
    return _dedupe_facts(facts)


def _first_number_with_unit(text: str) -> tuple[float, str, float] | None:
    match = re.search(r"(-?\d[\d,]*(?:\.\d+)?)\s*(万亿元|亿元|百万元|万元|千元|元|%|％|个百分点|百分点|亿|万)?", str(text or ""))
    if not match:
        return None
    return _normalize_number(match.group(1), match.group(2) or "")


def _normalize_number(raw: str, unit: str) -> tuple[float, str, float] | None:
    try:
        raw_value = float(str(raw).replace(",", ""))
    except ValueError:
        return None
    if not unit and 1900 <= raw_value <= 2099 and float(raw_value).is_integer():
        return None
    value = raw_value
    if unit == "万亿元":
        value *= 1e12
    elif unit in {"亿元", "亿"}:
        value *= 1e8
    elif unit == "百万元":
        value *= 1e6
    elif unit in {"万元", "万"}:
        value *= 1e4
    elif unit == "千元":
        value *= 1e3
    elif unit in {"%", "％", "个百分点", "百分点"}:
        value /= 100.0
    return value, unit, raw_value


def _first_year(text: str) -> str:
    match = re.search(r"20\d{2}", str(text or ""))
    return match.group(0) if match else ""


def _metric_match(metric_key: str, text: str) -> str:
    aliases = METRIC_DEFINITIONS.get(metric_key, {}).get("aliases") or []
    return "exact" if aliases and aliases[0] in text else "allowed_alias"


def _option_has_exact_alias(metric_key: str, text: str) -> bool:
    aliases = METRIC_DEFINITIONS.get(metric_key, {}).get("aliases") or []
    return bool(aliases and aliases[0] in text)


def _target_year(years: list[str], facts: list[dict[str, Any]]) -> str | None:
    if years:
        return max(years)
    fact_years = sorted({str(fact.get("year") or "") for fact in facts if fact.get("year")})
    if len(fact_years) == 1:
        return fact_years[0]
    return None


def _comparison_years(
    years: list[str],
    facts: list[dict[str, Any]],
    option_text: str,
) -> tuple[str | None, str | None]:
    if len(years) >= 2:
        ordered = sorted(years)
        return ordered[-1], ordered[-2]
    if len(years) == 1 and any(term in option_text for term in ["上年", "同比", "较上年", "相比"]):
        return years[0], str(int(years[0]) - 1)
    fact_years = sorted({str(fact.get("year") or "") for fact in facts if fact.get("year")})
    if not years and len(fact_years) == 2:
        return fact_years[-1], fact_years[-2]
    return None, None


def _single_fact_for_year(facts: list[dict[str, Any]], year: str) -> dict[str, Any] | None:
    matches = [fact for fact in facts if str(fact.get("year") or "") == str(year)]
    unique: dict[tuple[str, float], dict[str, Any]] = {}
    for fact in matches:
        unique[(str(fact.get("raw_evidence_id") or ""), float(fact.get("value") or 0.0))] = fact
    if len(unique) != 1:
        return None
    return next(iter(unique.values()))


def _amount_tolerance(left: float, right: float) -> float:
    return max(max(abs(left), abs(right), 1.0) * 0.005, 1.0)


def _contains_percent_claim(text: str) -> bool:
    return bool(re.search(r"\d[\d,]*(?:\.\d+)?\s*(?:%|％|百分点)", str(text or "")))


def _percentage_threshold(text: str) -> float | None:
    for raw, unit in re.findall(r"(-?\d[\d,]*(?:\.\d+)?)\s*(%|％|百分点)", str(text or "")):
        parsed = _normalize_number(raw, unit)
        if parsed is not None:
            return parsed[0]
    return None


def _dedupe_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float]] = set()
    for fact in facts:
        key = (str(fact.get("year") or ""), str(fact.get("metric") or ""), float(fact.get("value") or 0.0))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped


def _trace_judgement(value: dict[str, Any]) -> dict[str, Any]:
    support_ids = [str(item) for item in value.get("support_evidence_ids") or [] if str(item)]
    refute_ids = [str(item) for item in value.get("refute_evidence_ids") or [] if str(item)]
    return {
        "judgement": str(value.get("judgement") or "unknown"),
        "confidence": float(value.get("confidence") or 0.0),
        "evidence_ids": [*support_ids, *refute_ids],
    }


def _numbers_close(left: float, right: float) -> bool:
    if math.isclose(left, right, rel_tol=0.015, abs_tol=0.02):
        return True
    if left and right and math.isclose(left / 100.0, right, rel_tol=0.02, abs_tol=0.001):
        return True
    if left and right and math.isclose(left * 1e8, right, rel_tol=0.02, abs_tol=1.0):
        return True
    return False
