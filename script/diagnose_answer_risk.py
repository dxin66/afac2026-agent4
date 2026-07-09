from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.answer_utils import normalize_answer
from agent.evidence import EvidenceBuilder
from agent.io_utils import read_jsonl
from agent.io_utils import write_json
from agent.models import ChunkRecord
from agent.question_loader import load_questions
from agent.retrieval import SparseRetriever


def diagnose_answer_risk(
    *,
    question_dir: str | Path,
    answer_csv: str | Path,
    evidence_json: str | Path,
    logs_dir: str | Path,
    processed_dir: str | Path | None = None,
    rebuild_evidence: bool = False,
    qids: list[str] | None = None,
) -> dict[str, Any]:
    questions = load_questions(question_dir)
    if qids:
        requested = set(qids)
        questions = [question for question in questions if question["qid"] in requested]
        found = {question["qid"] for question in questions}
        missing = sorted(requested.difference(found))
        if missing:
            raise ValueError(f"requested qids not found: {missing}")
    answers = _load_answers(answer_csv)
    evidence_by_qid = {} if rebuild_evidence else json.loads(Path(evidence_json).read_text(encoding="utf-8"))
    evidence_builder = _load_evidence_builder(processed_dir) if rebuild_evidence else None

    rows: list[dict[str, Any]] = []
    for question in questions:
        qid = question["qid"]
        answer = answers.get(qid, "")
        build_error = ""
        evidence_record = evidence_by_qid.get(qid, {})
        model_output = evidence_record.get("model_output") if isinstance(evidence_record, dict) else {}
        if evidence_builder is not None:
            try:
                evidence = evidence_builder.build(question)
            except Exception as exc:
                evidence = {}
                build_error = str(exc)
        else:
            evidence = evidence_record.get("evidence") if isinstance(evidence_record, dict) else {}
        items = evidence.get("items") if isinstance(evidence, dict) else []
        score = 0
        reasons: list[str] = []

        try:
            normalized = normalize_answer(answer, question["answer_format"], question["options"])
            if normalized != answer:
                score += 50
                reasons.append(f"answer_not_normalized:{answer}->{normalized}")
        except Exception as exc:
            score += 100
            reasons.append(f"illegal_answer:{exc}")

        if not isinstance(items, list) or not items:
            score += 100
            reasons.append("empty_evidence")
            items = []
        if build_error:
            score += 100
            reasons.append("evidence_build_error:" + build_error[:160])

        expected_docs = [str(doc_id) for doc_id in question.get("doc_ids") or []]
        evidence_docs = sorted({str(item.get("doc_id") or "") for item in items})
        missing_docs = [doc_id for doc_id in expected_docs if doc_id not in evidence_docs]
        if missing_docs:
            score += 40 + 10 * len(missing_docs)
            reasons.append("missing_expected_docs:" + ",".join(missing_docs))

        evidence_text = "\n".join(str(item.get("text") or "") for item in items)
        if not rebuild_evidence:
            score += _check_model_output(question, answer, model_output, items, reasons)
        score += _check_selected_option_support(question, answer, evidence_text, reasons)
        score += _check_competing_option_support(question, answer, evidence_text, reasons)
        score += _check_domain_specific_risks(question, answer, evidence_text, evidence_docs, reasons)
        score += _check_option_matrix_risks(question, answer, evidence, reasons)

        rows.append(
            {
                "qid": qid,
                "domain": question["domain"],
                "answer_format": question["answer_format"],
                "answer": answer,
                "risk_score": score,
                "reasons": reasons,
                "evidence_items": len(items),
                "evidence_docs": evidence_docs,
                "evidence_source": "rebuilt" if rebuild_evidence else "evidence_json",
            }
        )

    rows.sort(key=lambda row: (row["risk_score"], row["qid"]), reverse=True)
    summary = {
        "metric": "answer_risk_diagnostics",
        "evidence_source": "rebuilt" if rebuild_evidence else str(evidence_json),
        "model_output_checked": not rebuild_evidence,
        "questions": len(rows),
        "flagged": sum(1 for row in rows if row["risk_score"] > 0),
        "top": rows[:30],
        "rows": rows,
    }
    logs = Path(logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    write_json(logs / "answer_risk_report.json", summary)
    _write_csv(logs / "answer_risk_report.csv", rows)
    return summary


def _load_evidence_builder(processed_dir: str | Path | None) -> EvidenceBuilder:
    if processed_dir is None:
        raise ValueError("--processed-dir is required when --rebuild-evidence is used")
    processed = Path(processed_dir)
    chunks = [ChunkRecord.from_dict(row) for row in read_jsonl(processed / "chunks.jsonl")]
    retriever = SparseRetriever.load(processed / "tfidf_index.pkl", chunks)
    return EvidenceBuilder(retriever)


def _load_answers(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return {
            str(row.get("qid") or ""): str(row.get("answer") or "")
            for row in csv.DictReader(f)
            if row.get("qid") and row.get("qid") != "summary"
        }


def _check_model_output(
    question: dict[str, Any],
    answer: str,
    model_output: Any,
    items: list[dict[str, Any]],
    reasons: list[str],
) -> int:
    if not isinstance(model_output, dict):
        reasons.append("missing_model_output")
        return 25
    score = 0
    model_answer = str(model_output.get("answer") or "")
    if model_answer and model_answer != answer:
        score += 30
        reasons.append(f"model_answer_mismatch:{model_answer}")
    if model_output.get("answer_source") == "fallback":
        score += 35
        derived = model_output.get("derived_answer") if isinstance(model_output.get("derived_answer"), dict) else {}
        status = str(derived.get("derive_status") or "unknown")
        reasons.append(f"answer_source_fallback:{status}")

    if model_output.get("answer_source") != "derived":
        inferred = _infer_from_judgement(question, model_output.get("option_judgement"))
        if inferred and inferred != answer:
            score += 60
            reasons.append(f"judgement_conflict:{inferred}")
        if not inferred:
            score += 15
            reasons.append("missing_or_invalid_option_judgement")

    refs = model_output.get("evidence_refs")
    if refs is not None:
        if isinstance(refs, str):
            refs = [refs]
        valid_refs = {str(item.get("chunk_id") or "") for item in items}
        if not isinstance(refs, list) or not refs:
            score += 15
            reasons.append("missing_evidence_refs")
        else:
            invalid = [str(ref) for ref in refs if str(ref) not in valid_refs]
            if invalid:
                score += 25
                reasons.append("invalid_evidence_refs:" + ",".join(invalid[:3]))
    return score


def _infer_from_judgement(question: dict[str, Any], judgements: Any) -> str:
    if not isinstance(judgements, dict):
        return ""
    true_letters = [
        letter
        for letter in sorted(str(key).upper() for key in question["options"])
        if _truthy(judgements.get(letter))
    ]
    if not true_letters:
        return ""
    if question["answer_format"] in {"mcq", "tf"} and len(true_letters) != 1:
        return ""
    try:
        return normalize_answer("".join(true_letters), question["answer_format"], question["options"])
    except Exception:
        return ""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1", "正确", "是"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def _check_selected_option_support(question: dict[str, Any], answer: str, evidence_text: str, reasons: list[str]) -> int:
    score = 0
    options = question.get("options") or {}
    for letter in answer:
        option_value = str(options.get(letter) or "")
        missing_groups = [
            group
            for group in _domain_required_option_terms(str(question.get("domain") or ""), option_value)
            if not any(term in evidence_text for term in group)
        ]
        if missing_groups:
            score += 12 * len(missing_groups)
            reasons.append(f"selected_{letter}_missing_terms:" + "|".join("/".join(group) for group in missing_groups[:4]))

        article_refs = _article_refs(option_value)
        missing_refs = [ref for ref in article_refs if ref not in evidence_text]
        if missing_refs:
            score += 15 * len(missing_refs)
            reasons.append(f"selected_{letter}_missing_article_refs:" + ",".join(missing_refs[:4]))
    return score


def _check_competing_option_support(question: dict[str, Any], answer: str, evidence_text: str, reasons: list[str]) -> int:
    score = 0
    domain = str(question.get("domain") or "")
    for letter, option_value in sorted((question.get("options") or {}).items()):
        letter = str(letter).upper()
        if letter in answer:
            continue
        groups = _domain_required_option_terms(domain, str(option_value))
        if len(groups) < 2:
            continue
        supported = sum(1 for group in groups if any(term in evidence_text for term in group))
        if supported == len(groups):
            score += 6
            reasons.append(f"unselected_{letter}_all_terms_supported")
    return score


def _check_domain_specific_risks(
    question: dict[str, Any],
    answer: str,
    evidence_text: str,
    evidence_docs: list[str],
    reasons: list[str],
) -> int:
    domain = str(question.get("domain") or "")
    question_text = str(question.get("question") or "")
    option_text = " ".join(str(value) for value in (question.get("options") or {}).values())
    text = question_text + " " + option_text
    score = 0

    if domain == "financial_reports":
        metric_terms = ["营业收入", "营业总收入", "营收", "研发投入", "经营活动产生的现金流量净额", "经营活动现金流净额", "归母净利润", "归属于上市公司股东的净利润", "现金分红", "毛利率"]
        if any(term in text for term in metric_terms) and not re.search(r"\d[\d,]*(?:\.\d+)?%?", evidence_text):
            score += 30
            reasons.append("financial_report_no_numeric_evidence")
        years = sorted(set(re.findall(r"20\d{2}", text)))
        missing_years = [year for year in years if year not in evidence_text]
        if missing_years:
            score += 8 * len(missing_years)
            reasons.append("financial_report_missing_years:" + ",".join(missing_years[:4]))

    if domain == "insurance":
        if any(term in text for term in ["计算", "金额", "排序", "赔付", "给付"]):
            needed = ["保险责任", "责任免除", "身故保险金", "现金价值", "免赔额", "等待期"]
            if not any(term in evidence_text for term in needed):
                score += 30
                reasons.append("insurance_calculation_lacks_rule_terms")
        expected_docs = [str(doc_id) for doc_id in question.get("doc_ids") or []]
        if len(expected_docs) >= 3 and len(evidence_docs) < len(expected_docs):
            score += 20
            reasons.append("insurance_multi_product_doc_gap")

    if domain in {"regulatory", "financial_contracts"}:
        selected_text = " ".join(str((question.get("options") or {}).get(letter) or "") for letter in answer)
        if "第" in selected_text:
            refs = _article_refs(selected_text)
            missing_refs = [ref for ref in refs if ref not in evidence_text]
            if missing_refs:
                score += 12 * len(missing_refs)
                reasons.append("legal_selected_article_gap:" + ",".join(missing_refs[:4]))
        if any(marker in selected_text for marker in ["且", "同时", "两项", "任一", "虽", "但", "无需", "不得"]):
            score += 4
            reasons.append("compound_legal_option")

    if domain == "research":
        expected_docs = [str(doc_id) for doc_id in question.get("doc_ids") or []]
        if len(expected_docs) >= 2 and len(evidence_docs) < len(expected_docs):
            score += 25
            reasons.append("research_multi_doc_gap")
        if any(term in text for term in ["同比", "增速", "贡献", "市场规模"]) and not re.search(r"\d[\d,]*(?:\.\d+)?%?", evidence_text):
            score += 20
            reasons.append("research_metric_no_numeric_evidence")

    if question.get("answer_format") == "multi" and len(answer) >= 3:
        score += 3
        reasons.append("multi_answer_three_or_more")
    return score


def _check_option_matrix_risks(
    question: dict[str, Any],
    answer: str,
    evidence: dict[str, Any],
    reasons: list[str],
) -> int:
    option_evidence = evidence.get("option_evidence") if isinstance(evidence, dict) else {}
    if not isinstance(option_evidence, dict) or not option_evidence:
        return 0

    selected = set(answer)
    score = 0
    for letter in selected:
        row = option_evidence.get(letter)
        if not isinstance(row, dict):
            score += 12
            reasons.append(f"selected_{letter}_missing_option_matrix")
            continue
        missing_docs = [str(doc_id) for doc_id in row.get("missing_docs") or []]
        if missing_docs:
            score += 10 * len(missing_docs)
            reasons.append(f"selected_{letter}_matrix_missing_docs:" + ",".join(missing_docs[:4]))
        hit_count = _option_matrix_hit_count(row)
        if hit_count == 0:
            score += 12
            reasons.append(f"selected_{letter}_matrix_no_hits")

    for letter, row in sorted(option_evidence.items()):
        letter = str(letter).upper()
        if letter in selected or not isinstance(row, dict):
            continue
        strong_hits = _option_matrix_strong_hit_count(row)
        if strong_hits >= 2:
            score += 6
            reasons.append(f"unselected_{letter}_matrix_strong_hits:{strong_hits}")
    return score


def _option_matrix_hit_count(row: dict[str, Any]) -> int:
    by_doc = row.get("by_doc")
    if not isinstance(by_doc, dict):
        return 0
    return sum(len(items) for items in by_doc.values() if isinstance(items, list))


def _option_matrix_strong_hit_count(row: dict[str, Any]) -> int:
    by_doc = row.get("by_doc")
    if not isinstance(by_doc, dict):
        return 0
    count = 0
    for items in by_doc.values():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            matched_terms = item.get("matched_terms")
            if isinstance(matched_terms, list) and len(matched_terms) >= 2:
                count += 1
    return count


def _article_refs(text: str) -> list[str]:
    refs = re.findall(r"第[一二三四五六七八九十百千万\d]+[条章节款项]", text)
    return list(dict.fromkeys(refs))


def _domain_required_option_terms(domain: str, option_text_value: str) -> list[tuple[str, ...]]:
    terms_by_domain: dict[str, list[tuple[str, tuple[str, ...]]]] = {
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
        ],
        "regulatory": [
            ("股东会", ("股东会", "股东大会")),
            ("普通决议", ("普通决议",)),
            ("特别决议", ("特别决议",)),
            ("变更募集资金用途", ("变更募集资金用途",)),
            ("独立董事", ("独立董事",)),
            ("公司章程", ("公司章程", "本章程")),
            ("资产负债率", ("资产负债率",)),
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
        ],
    }
    return [evidence_terms for trigger, evidence_terms in terms_by_domain.get(domain, []) if trigger in option_text_value]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["qid", "domain", "answer_format", "answer", "risk_score", "evidence_items", "evidence_docs", "reasons"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "qid": row["qid"],
                    "domain": row["domain"],
                    "answer_format": row["answer_format"],
                    "answer": row["answer"],
                    "risk_score": row["risk_score"],
                    "evidence_items": row["evidence_items"],
                    "evidence_docs": ",".join(row["evidence_docs"]),
                    "reasons": "; ".join(row["reasons"]),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Flag answer rows that deserve targeted accuracy review.")
    parser.add_argument("--question-dir", default=str(ROOT / "public_dataset_upload" / "questions" / "group_a"))
    parser.add_argument("--answer-csv", default=str(ROOT / "answer.csv"))
    parser.add_argument("--evidence-json", default=str(ROOT / "evidence.json"))
    parser.add_argument("--processed-dir", default=str(ROOT / "processed_data"))
    parser.add_argument("--logs-dir", default=str(ROOT / "logs"))
    parser.add_argument("--rebuild-evidence", action="store_true", help="rebuild evidence from processed_data instead of reading evidence.json")
    parser.add_argument("--qid", action="append", help="diagnose only this qid; may be provided multiple times")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    summary = diagnose_answer_risk(
        question_dir=args.question_dir,
        answer_csv=args.answer_csv,
        evidence_json=args.evidence_json,
        logs_dir=args.logs_dir,
        processed_dir=args.processed_dir,
        rebuild_evidence=args.rebuild_evidence,
        qids=args.qid,
    )
    print(
        f"answer_risk flagged={summary['flagged']}/{summary['questions']} "
        f"evidence_source={summary['evidence_source']}"
    )
    for row in summary["top"][: args.top]:
        if row["risk_score"] <= 0:
            continue
        print(f"{row['qid']} score={row['risk_score']} answer={row['answer']} reasons={'; '.join(row['reasons'][:4])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
