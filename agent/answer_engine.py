from __future__ import annotations

import json
import time
from typing import Any

from agent.answer_utils import normalize_answer, option_text
from agent.claim_verifier import (
    build_dual_evidence_summary,
    classify_retrieval_evidence_matrix,
    compute_matrix_gaps,
    derive_answer,
    normalize_llm_option_judgement,
)
from agent.finance_solver import build_finance_solver_artifacts
from agent.models import AnswerResult, ModelResponse, Usage
from agent.question_plan import build_question_plan
from agent.qwen_client import QwenClient


class AnswerEngine:
    """LLM-first answer engine.

    A single structured Qwen call is the primary reasoner for every question:
    it judges each option against the retrieved evidence directly. Rule-based
    signals (keyword/number/article matching, finance formula checks) are
    computed and handed to the model as labeled hints, never as a gate that
    can decide or veto an answer before the model is asked.
    """

    def __init__(self, client: QwenClient) -> None:
        self.client = client

    def answer(self, question: dict[str, Any], evidence: dict[str, Any]) -> AnswerResult:
        started = time.perf_counter()
        question_plan = evidence.get("question_plan")
        if not isinstance(question_plan, dict):
            question_plan = build_question_plan(question, candidate_doc_ids=evidence.get("candidate_doc_ids") or [])
        retrieval_matrix = evidence.get("retrieval_evidence_matrix")
        if not isinstance(retrieval_matrix, dict):
            retrieval_matrix = _legacy_retrieval_matrix(evidence)
        classified_matrix = classify_retrieval_evidence_matrix(question, question_plan, retrieval_matrix)
        matrix_gaps = compute_matrix_gaps(question_plan, retrieval_matrix, classified_matrix)
        finance_artifacts = build_finance_solver_artifacts(question, retrieval_matrix)
        dual_evidence_summary = build_dual_evidence_summary(question_plan, classified_matrix)
        valid_evidence_ids = {str(item.get("evidence_id") or "") for item in evidence.get("items") or []}
        valid_evidence_ids.discard("")

        usage = Usage()
        parsed: dict[str, Any] = {}
        raw_content = ""
        llm_error: dict[str, str] | None = None
        repair_attempted = False
        option_judgement: dict[str, Any] | None = None

        try:
            response = self._reason(question, evidence, question_plan, dual_evidence_summary, finance_artifacts)
            usage = usage.add(response.usage)
            parsed = response.parsed
            raw_content = response.content
            option_judgement = normalize_llm_option_judgement(parsed.get("option_judgement"), question, valid_evidence_ids)
        except Exception as exc:  # noqa: BLE001 - any LLM/parse failure triggers one repair attempt
            llm_error = {"type": exc.__class__.__name__, "message": str(exc)[:500]}

        if option_judgement is None:
            repair_attempted = True
            try:
                response = self._repair(
                    question,
                    evidence,
                    question_plan,
                    dual_evidence_summary,
                    finance_artifacts,
                    previous_error=llm_error.get("message") if llm_error else "option_judgement missing or not an object",
                )
                usage = usage.add(response.usage)
                parsed = response.parsed
                raw_content = response.content
                option_judgement = normalize_llm_option_judgement(parsed.get("option_judgement"), question, valid_evidence_ids)
                llm_error = None
            except Exception as exc:  # noqa: BLE001 - fall back to an all-unknown judgement rather than crash the run
                llm_error = {"type": exc.__class__.__name__, "message": str(exc)[:500]}
                option_judgement = None

        if option_judgement is None:
            option_judgement = {
                str(letter).upper(): {"judgement": "unknown", "confidence": 0.0, "support_evidence_ids": [], "refute_evidence_ids": [], "reason": "llm_unavailable"}
                for letter in question.get("options") or {}
            }

        derived_answer = derive_answer(question, question_plan, option_judgement, fallback_answer=str(parsed.get("answer") or ""))
        answer = normalize_answer(derived_answer["answer"], question["answer_format"], question["options"])

        evidence["question_plan"] = question_plan
        evidence["retrieval_evidence_matrix"] = retrieval_matrix
        evidence["classified_evidence_matrix"] = classified_matrix
        evidence["matrix_gaps"] = matrix_gaps
        evidence["dual_evidence_summary"] = dual_evidence_summary
        evidence["raw_table_evidence"] = finance_artifacts.get("raw_table_evidence") or []
        evidence["formula_evidence"] = finance_artifacts.get("formula_evidence") or []
        evidence["option_judgement"] = option_judgement
        evidence["derived_answer"] = derived_answer
        evidence["evidence_retrieval"] = parsed.get("evidence_retrieval") or []

        model_output: dict[str, Any] = {
            "answer": answer,
            "llm_option_judgement": parsed.get("option_judgement"),
            "option_judgement": option_judgement,
            "derived_answer": derived_answer,
            "derive_status": derived_answer.get("derive_status"),
            "answer_source": derived_answer.get("answer_source"),
            "evidence_retrieval": parsed.get("evidence_retrieval") or [],
            "raw_table_evidence": finance_artifacts.get("raw_table_evidence") or [],
            "formula_evidence": finance_artifacts.get("formula_evidence") or [],
            "repair_attempted": repair_attempted,
            "llm_error": llm_error,
            "timing": {"answer_seconds": round(time.perf_counter() - started, 4)},
        }
        return AnswerResult(
            qid=question["qid"],
            answer=answer,
            usage=usage,
            evidence=evidence,
            model_output=model_output,
            raw_model_content=raw_content,
        )

    def _reason(
        self,
        question: dict[str, Any],
        evidence: dict[str, Any],
        question_plan: dict[str, Any],
        dual_evidence_summary: dict[str, Any],
        finance_artifacts: dict[str, Any],
    ) -> ModelResponse:
        messages = [
            {"role": "system", "content": _system_prompt(question["domain"])},
            {
                "role": "user",
                "content": _user_prompt(question, evidence, question_plan, dual_evidence_summary, finance_artifacts),
            },
        ]
        return self.client.complete_json(messages, max_tokens=_primary_max_tokens(question, question_plan), temperature=0.0)

    def _repair(
        self,
        question: dict[str, Any],
        evidence: dict[str, Any],
        question_plan: dict[str, Any],
        dual_evidence_summary: dict[str, Any],
        finance_artifacts: dict[str, Any],
        *,
        previous_error: str,
    ) -> ModelResponse:
        payload = json.loads(_user_prompt(question, evidence, question_plan, dual_evidence_summary, finance_artifacts))
        payload["previous_error"] = previous_error
        payload["hard_requirement"] = (
            "上一次输出无法解析或不完整。必须重新输出合法 JSON，包含 option_judgement（覆盖全部选项字母）、"
            "answer、evidence_retrieval 三个字段。evidence_id 必须来自 evidence 列表中的真实 evidence_id，不能编造。"
        )
        messages = [
            {"role": "system", "content": _system_prompt(question["domain"])},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        return self.client.complete_json(messages, max_tokens=_primary_max_tokens(question, question_plan), temperature=0.0)


def _primary_max_tokens(question: dict[str, Any], question_plan: dict[str, Any]) -> int:
    tokens = 900
    if question.get("answer_format") == "multi":
        tokens += 300
    if question_plan.get("is_finance_calc"):
        tokens += 300
    if len(question_plan.get("required_docs") or []) >= 3:
        tokens += 200
    return min(tokens, 1800)


def _system_prompt(domain: str) -> str:
    return (
        "你是金融长文本问答专家。你会看到与题目相关的证据原文片段（已从长文档中检索定位），"
        "以及规则引擎生成的启发式信号（rule_hints/finance_hints，仅供参考、可能有误，一切以证据原文为准，不能替代你自己的判断）。\n"
        "任务：对每个选项逐一判断该选项陈述是 true（被证据支持）、false（被证据反驳）还是 unknown（证据不足以判断）。"
        "unknown 不等于 false，禁止把证据不足当成选项错误来处理。\n"
        "judgement 为 true 时必须在 evidence_ids 中引用至少一个真实存在的 evidence_id 作为支持证据；"
        "为 false 时必须引用至少一个 evidence_id 作为反驳证据；找不到对应证据时必须输出 unknown，不能凭常识或猜测判断，也不能编造 evidence_id。\n"
        "题目如果是要求找出不正确/不符合/错误的选项（negative），answer 应选出被判定为 false 的选项；"
        "否则（positive）answer 应选出被判定为 true 的选项。\n"
        "多选题（multi）的 answer 是所有被判定为 true（或 negative 题下的 false）选项字母的升序拼接，不使用分隔符。\n"
        "单选题（mcq）和判断题（tf）的 answer 只能是一个字母。\n"
        "必须输出合法 JSON，字段为 option_judgement、answer、evidence_retrieval，不要输出多余文本或 markdown。"
        f"{_domain_prompt(domain)}"
    )


def _domain_prompt(domain: str) -> str:
    prompts = {
        "insurance": (
            "保险题优先核对保险责任、责任免除、等待期、免赔额、赔付比例、已领金额、现金价值和账户价值；"
            "涉及计算或跨产品比较时逐项核对每个产品对应的证据，不能只凭产品名称判断。"
        ),
        "regulatory": (
            "监管题优先引用条款原文，重点核对金额门槛、日期、报告期限、审批/备案/披露义务和适用主体；"
            "相似法规条文冲突时以证据中的具体文件和条款为准，不要依赖常识推断法条内容。"
            "选项包含复合判断（多个子条件、多个法条引用、多个表决方式描述）时，只要其中一个子判断、法条号、"
            "表决方式或范围表述与证据不符，整体判定为 false。"
        ),
        "financial_contracts": (
            "合同题优先核对合同主体、金额、期限、交付/付款条件、违约责任、解除条件和编号条款；"
            "多合同比较时分别核对每份合同的证据，不要用一份合同的条款去回答另一份合同的问题。"
        ),
        "financial_reports": (
            "财报题优先核对公司、年份、指标口径和表格数值；同比、占比、双位数增长等结论必须基于同口径数据核实。"
            "finance_hints 中的候选数值来自规则引擎的表格/数值抽取，请结合证据原文核实后再采信，不要直接照搬。"
        ),
        "research": (
            "研报题优先核对行业、公司、指标时间范围、渠道贡献、投资观点和风险提示；"
            "多研报题目必须联合核对所有相关文档的证据。"
        ),
    }
    return prompts.get(domain, "")


def _user_prompt(
    question: dict[str, Any],
    evidence: dict[str, Any],
    question_plan: dict[str, Any],
    dual_evidence_summary: dict[str, Any],
    finance_artifacts: dict[str, Any],
) -> str:
    payload = {
        "qid": question["qid"],
        "domain": question["domain"],
        "answer_format": question["answer_format"],
        "type": question.get("type", ""),
        "is_negative": bool(question_plan.get("is_negative")),
        "question": question["question"],
        "options_text": option_text(question),
        "options": question["options"],
        "evidence": _evidence_payload(evidence),
        "rule_hints": _rule_hints_payload(dual_evidence_summary),
        "finance_hints": _finance_hints_payload(finance_artifacts),
        "output_schema": _output_schema(question),
    }
    return json.dumps(payload, ensure_ascii=False)


def _evidence_payload(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for item in evidence.get("items") or []:
        items.append(
            {
                "evidence_id": item.get("evidence_id"),
                "doc_id": item.get("doc_id"),
                "title": item.get("title"),
                "section_title": item.get("section_title") or "",
                "article_no": item.get("article_no") or item.get("clause_no") or "",
                "row_key": item.get("row_key") or "",
                "page": item.get("page_start"),
                "text": item.get("text"),
            }
        )
    return items


def _rule_hints_payload(dual_evidence_summary: dict[str, Any]) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    for letter, row in sorted((dual_evidence_summary or {}).items()):
        hints[letter] = {
            "support_evidence_ids": row.get("support_evidence_ids", [])[:4],
            "refute_evidence_ids": row.get("refute_evidence_ids", [])[:4],
            "top_support_reasons": row.get("top_support_reasons", []),
            "top_refute_reasons": row.get("top_refute_reasons", []),
            "contradiction": row.get("contradiction", False),
        }
    return hints


def _finance_hints_payload(finance_artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    hints = []
    for row in finance_artifacts.get("formula_evidence") or []:
        if not row.get("depends_on"):
            continue
        hints.append(
            {
                "option": row.get("option"),
                "metric": row.get("metric"),
                "formula": row.get("formula"),
                "computed_value": row.get("value"),
                "claim_value": row.get("claim_value"),
                "relation": row.get("relation"),
                "rule_judgement": row.get("judgement"),
                "depends_on_evidence": row.get("depends_on"),
            }
        )
    return hints


def _output_schema(question: dict[str, Any]) -> dict[str, Any]:
    return {
        "option_judgement": {
            str(letter).upper(): {
                "judgement": "true|false|unknown",
                "confidence": 0.0,
                "evidence_ids": ["最多3个真实存在的evidence_id"],
                "reasoning": "不超过40字的简短依据",
            }
            for letter in sorted(question.get("options") or {})
        },
        "answer": "最终答案字母（mcq/tf为单个字母，multi为升序字母拼接）",
        "evidence_retrieval": [
            {"doc_id": "...", "quoted_clause": "引用的原文片段", "reasoning": "该证据支持/反驳了哪个选项"}
        ],
    }


def _legacy_retrieval_matrix(evidence: dict[str, Any]) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    option_evidence = evidence.get("option_evidence") if isinstance(evidence, dict) else {}
    if not isinstance(option_evidence, dict):
        return matrix
    for letter, row in sorted(option_evidence.items()):
        letter = str(letter).upper()
        matrix[letter] = {}
        by_doc = row.get("by_doc") if isinstance(row, dict) else {}
        if not isinstance(by_doc, dict):
            continue
        for doc_id, items in sorted(by_doc.items()):
            candidates = []
            for index, item in enumerate(items or [], start=1):
                candidate = dict(item)
                candidate.setdefault("evidence_id", f"legacy:{letter}:{doc_id}:{index}")
                candidate.setdefault("option", letter)
                candidate.setdefault("candidate_source", "legacy_option_evidence")
                candidates.append(candidate)
            matrix[letter][str(doc_id)] = {
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
