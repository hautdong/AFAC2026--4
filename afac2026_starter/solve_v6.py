from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from openai import OpenAI

from .common import (
    build_question_text,
    ensure_dir,
    load_settings,
    normalize_question_items,
    read_json_any,
    safe_json_dump,
)
from .solve import write_answer_csv
from .solve_v4 import (
    ARTICLE_ANCHOR_RE,
    LETTERS,
    NUMBER_ANCHOR_RE,
    DocumentIndex,
    EvidencePack,
    PageChunk,
    add_usage,
    answer_protocol,
    build_document_indexes,
    call_json,
    collect_review_queries,
    confidence_values,
    domain_instructions,
    format_context,
    parse_answer,
    review_reasons,
    select_evidence,
    valid_citation_ratio,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:  # type: ignore
        return iterable


COMPARISON_WORDS = (
    "高于", "低于", "增加", "减少", "上升", "下降", "增长", "下滑",
    "超过", "不超过", "至少", "至多", "早于", "晚于", "同比", "环比",
    "均", "都", "全部", "分别", "排序", ">", "<", "=",
)
SCOPE_WORDS = (
    "本期", "本次", "注册", "主体信用评级", "债项信用评级", "年度",
    "特别", "拟", "已", "实施", "完成", "工作日", "自然日", "普通决议",
    "特别决议", "必须", "应当", "可以", "不得",
)
LITERAL_FIELD_TERMS = (
    "发行人名称", "发行主体", "发行金额", "发行规模", "注册金额", "注册规模",
    "上市地点", "主体信用评级", "债项信用评级", "营业收入", "净利润",
    "现金分红", "现金流量净额", "研发投入", "资产负债率", "免赔额",
    "保险金额", "现金价值", "保单账户价值", "施行日期", "工作日",
)


def question_profile(question: Dict[str, Any]) -> Dict[str, Any]:
    text = build_question_text(question)
    compact_numbers = list(dict.fromkeys(match.group(0).replace(" ", "") for match in NUMBER_ANCHOR_RE.finditer(text)))
    articles = list(dict.fromkeys(match.group(0).replace(" ", "") for match in ARTICLE_ANCHOR_RE.finditer(text)))
    comparisons = [word for word in COMPARISON_WORDS if word in text]
    scopes = [word for word in SCOPE_WORDS if word in text]
    return {
        "document_order": [
            f"DOCUMENT {index}={doc_id}"
            for index, doc_id in enumerate(question.get("doc_ids", []), start=1)
        ],
        "numbers_dates_units": compact_numbers[:16],
        "article_or_clause_numbers": articles[:10],
        "comparison_checks": comparisons,
        "scope_checks": scopes,
        "requires_cross_document_check": len(question.get("doc_ids", [])) > 1,
    }


def normalized_literal(text: str) -> str:
    return re.sub(r"[\s，。；：、（）()\[\]【】]", "", text).lower()


def literal_evidence_hints(
    question: Dict[str, Any],
    pack: EvidencePack,
    max_hints_per_option: int = 1,
) -> Dict[str, List[Dict[str, Any]]]:
    options = question.get("options")
    if not isinstance(options, dict):
        return {}
    hints: Dict[str, List[Dict[str, Any]]] = {}
    for letter in LETTERS:
        option = str(options.get(letter, ""))
        if not option:
            continue
        fields = [field for field in LITERAL_FIELD_TERMS if field in option]
        numbers = [match.group(0).replace(" ", "") for match in NUMBER_ANCHOR_RE.finditer(option)]
        quoted_values = re.findall(r"[“\"]([^”\"]{2,30})[”\"]", option)
        anchors = fields + numbers + quoted_values
        if not anchors:
            continue
        scored: List[Tuple[int, PageChunk]] = []
        for chunk in pack.chunks:
            normalized = normalized_literal(chunk.text)
            score = sum(3 for field in fields if normalized_literal(field) in normalized)
            score += sum(2 for number in numbers if normalized_literal(number) in normalized)
            score += sum(2 for value in quoted_values if normalized_literal(value) in normalized)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], -item[1].score))
        option_hints: List[Dict[str, Any]] = []
        for score, chunk in scored[:max_hints_per_option]:
            option_hints.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "page": chunk.page,
                    "matched_anchors": [
                        anchor for anchor in anchors
                        if normalized_literal(anchor) in normalized_literal(chunk.text)
                    ],
                    "excerpt": chunk.text[:500],
                    "match_score": score,
                }
            )
        if option_hints:
            hints[letter] = option_hints
    return hints


def semantic_guardrails(question: Dict[str, Any]) -> List[str]:
    text = build_question_text(question)
    guards: List[str] = []
    if "上市地点" in text:
        guards.append(
            "A title-page field such as '上市地点：深圳证券交易所' directly supports a literal claim about "
            "the securities listing place. Do not silently narrow '证券' to '债券' unless the option says 债券."
        )
    if "现金分红" in text and "50%" in text and "实施" in text:
        guards.append(
            "A source statement saying '延续高比例分红的政策，连续三年以净利润的50%实施现金分红' "
            "directly supports implementation of a 50% cash-dividend policy. Pending approval of the current "
            "year's payment plan does not negate policy implementation unless the option claims payment completed."
        )
    if re.search(r"\d+(?:\.\d+)?%?\s*至\s*\d+(?:\.\d+)?%?", text):
        guards.append(
            "For a stated numeric interval, test unrounded source values against both endpoints; a value such as "
            "66.38% is outside an upper bound of 66%."
        )
    if any(symbol in text for symbol in (">", "<", "排序")):
        guards.append("After calculating values, verify every displayed inequality in its written order.")
    if "持续放缓趋势" in text:
        guards.append(
            "When the source itself says growth momentum weakened / growth slowed and states that a rate fell "
            "from X to Y over the same multi-year interval, this supports a '持续放缓趋势' claim. Do not demand "
            "every intermediate annual value unless the option explicitly says '逐年'."
        )
    return guards


def apply_deterministic_checks(
    question: Dict[str, Any],
    pack: EvidencePack,
    parsed: Dict[str, Any],
) -> Dict[str, Any]:
    result = copy.deepcopy(parsed)
    options = question.get("options")
    judgments = result.get("judgments")
    if not isinstance(options, dict) or not isinstance(judgments, dict):
        return result

    changed = False
    for letter in LETTERS:
        option = str(options.get(letter, ""))
        if (
            "明确标注" not in option
            or "上市地点" not in option
            or any(negation in option for negation in ("未明确", "没有", "并非", "不是"))
        ):
            continue
        exchanges = re.findall(r"(?:上海|深圳|北京|香港)(?:证券)?交易所", option)
        if not exchanges:
            continue
        matching = [
            chunk for chunk in pack.chunks
            if "上市地点" in normalized_literal(chunk.text)
            and any(normalized_literal(exchange) in normalized_literal(chunk.text) for exchange in exchanges)
        ]
        if not matching:
            continue
        best = max(matching, key=lambda item: item.score)
        item = judgments.get(letter)
        if not isinstance(item, dict):
            item = {}
            judgments[letter] = item
        item.update(
            {
                "verdict": True,
                "confidence": 1.0,
                "citations": [best.chunk_id],
                "reasoning": (
                    "Deterministic title-field check: the same source chunk explicitly contains "
                    f"'上市地点' and '{exchanges[0]}'; no extra security-type qualifier may be added."
                ),
            }
        )
        changed = True

    for letter in LETTERS:
        option = str(options.get(letter, ""))
        if "持续放缓趋势" not in option or "逐年" in option:
            continue
        matching = [
            chunk for chunk in pack.chunks
            if re.search(r"20\d{2}\s*[-—至]\s*20\d{2}\s*年", chunk.text)
            and "降至" in chunk.text
            and any(phrase in chunk.text for phrase in ("增速放缓", "增长动能减弱", "增速趋缓"))
        ]
        if not matching:
            continue
        best = max(matching, key=lambda item: item.score)
        item = judgments.get(letter)
        if not isinstance(item, dict):
            item = {}
            judgments[letter] = item
        item.update(
            {
                "verdict": True,
                "confidence": 1.0,
                "citations": [best.chunk_id],
                "reasoning": (
                    "Deterministic trend check: one source passage explicitly characterizes slowing growth "
                    "and reports a decline from the start to the end of the stated multi-year interval."
                ),
            }
        )
        changed = True

    if changed:
        true_letters = [
            letter for letter in LETTERS
            if isinstance(judgments.get(letter), dict)
            and judgments[letter].get("verdict") is True
        ]
        answer_format = str(question.get("answer_format", "multi")).lower()
        result["answer"] = true_letters[0] if answer_format in {"mcq", "tf"} else "".join(true_letters)
        result.setdefault("deterministic_checks", []).append("grounded_literal_or_trend_check")
    return result


def retrieval_settings(question: Dict[str, Any], review: bool = False) -> Dict[str, Any]:
    # Accuracy-first mode deliberately preserves v4's full evidence depth. Token
    # compression is postponed until this version has an online accuracy score.
    return {
        "group_limit": 2,
        "selection_budget": 18 if review else 16,
        "continuation_limit": 3,
        "include_early_summary": True,
        "max_chars": 24000,
    }


def retrieve_v6(
    question: Dict[str, Any],
    indexes: Dict[str, DocumentIndex],
    review: bool = False,
    extra_queries: Optional[Sequence[str]] = None,
) -> EvidencePack:
    settings = retrieval_settings(question, review=review)
    return select_evidence(
        question,
        indexes,
        settings["max_chars"],
        extra_queries=extra_queries,
        group_limit=settings["group_limit"],
        selection_budget=settings["selection_budget"],
        continuation_limit=settings["continuation_limit"],
        include_early_summary=settings["include_early_summary"],
    )


def cited_chunk_ids(parsed: Dict[str, Any]) -> Set[str]:
    result: Set[str] = set()
    judgments = parsed.get("judgments")
    if not isinstance(judgments, dict):
        return result
    for letter in LETTERS:
        item = judgments.get(letter)
        if not isinstance(item, dict) or not isinstance(item.get("citations"), list):
            continue
        result.update(str(citation) for citation in item["citations"])
    return result


def compact_review_pack(
    question: Dict[str, Any],
    pack: EvidencePack,
    primary: Dict[str, Any],
    max_chars: int,
) -> EvidencePack:
    doc_ids = [str(doc_id) for doc_id in question.get("doc_ids", [])]
    required_ids = cited_chunk_ids(primary)
    selected: Dict[str, PageChunk] = {}

    def add(chunk: PageChunk) -> None:
        selected.setdefault(chunk.chunk_id, chunk)

    # Keep cited facts and targeted re-retrieval hits first.
    for chunk in pack.chunks:
        if chunk.chunk_id in required_ids or any(label.startswith("REVIEW_") for label in chunk.matched_for):
            add(chunk)

    # Every source document remains represented even after compression.
    for doc_id in doc_ids:
        doc_chunks = [chunk for chunk in pack.chunks if chunk.doc_id == doc_id]
        headers = [chunk for chunk in doc_chunks if "DOC_HEADER" in chunk.matched_for]
        if headers:
            add(headers[0])
        elif doc_chunks:
            add(max(doc_chunks, key=lambda item: item.score))

    # Preserve one strongest fact for each option. For two-document comparisons,
    # retain one per document so "both" claims remain auditable.
    two_doc_comparison = len(doc_ids) <= 2
    for letter in LETTERS:
        matches = [chunk for chunk in pack.chunks if letter in chunk.matched_for]
        if not matches:
            continue
        if two_doc_comparison:
            for doc_id in doc_ids:
                per_doc = [chunk for chunk in matches if chunk.doc_id == doc_id]
                if per_doc:
                    add(max(per_doc, key=lambda item: item.score))
        else:
            add(max(matches, key=lambda item: item.score))

    # Continuations are useful for split tables, but one per document is enough.
    for doc_id in doc_ids:
        continuations = [
            chunk for chunk in pack.chunks
            if chunk.doc_id == doc_id and "CONTINUATION" in chunk.matched_for
        ]
        if continuations:
            add(max(continuations, key=lambda item: item.score))

    ranked_remaining = sorted(
        pack.chunks,
        key=lambda item: (-len(item.matched_for), -item.score),
    )
    for chunk in ranked_remaining:
        candidate = list(selected.values()) + [chunk]
        context = format_context(candidate, doc_ids, max_chars)
        if chunk.chunk_id in context:
            add(chunk)
        if len(context) >= max_chars * 0.92:
            break

    chunks = sorted(selected.values(), key=lambda item: (item.doc_order, item.page, item.chunk_order))
    context = format_context(chunks, doc_ids, max_chars)
    diagnostics = dict(pack.diagnostics)
    diagnostics.update(
        {
            "pre_compaction_chunks": len(pack.chunks),
            "chunk_count": len(chunks),
            "context_chars": len(context),
            "selected_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "page": chunk.page,
                    "score": round(chunk.score, 3),
                    "matched_for": sorted(chunk.matched_for),
                    "text_preview": chunk.text[:260],
                }
                for chunk in chunks
            ],
        }
    )
    return EvidencePack(chunks=chunks, context=context, diagnostics=diagnostics)


def primary_messages_v6(question: Dict[str, Any], pack: EvidencePack) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a precise financial document analyst. Use only source excerpts. Build a compact shared "
                "fact table, then judge every option. Preserve numbers, units, years, negations, conditions, and "
                "document order. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": f"""
Question:
{build_question_text(question)}

Parsed checks:
{json.dumps(question_profile(question), ensure_ascii=False, separators=(',', ':'))}

Literal evidence hints (retrieval aids, not automatic verdicts):
{json.dumps(literal_evidence_hints(question, pack), ensure_ascii=False, separators=(',', ':'))}

Semantic guardrails:
{json.dumps(semantic_guardrails(question), ensure_ascii=False)}

Answer protocol:
{answer_protocol(question)}

Domain checks:
{domain_instructions(question)}

Source excerpts:
{pack.context}

Method:
1. Record each decisive source fact once. Use exact chunk_id citations.
2. Verify both/all claims in every relevant document and evaluate every displayed >, <, or = sign.
3. Treat numeric ranges literally: every observed decimal must fall inside the stated endpoints. Do not round 66.38% down to 66%.
4. Judge the option's literal wording. Do not invent an unstated qualifier such as "bond listing" when the option only says "securities listing".
5. A near-verbatim source statement is direct support; do not reject a faithful summary merely because the source also lists additional causes.
6. For calculations show only the formula and substituted values. Missing evidence is uncertainty, not contradiction.
7. If decisive evidence is absent, set needs_review=true and provide a short targeted query.

Return strict JSON:
{{
  "facts": [{{"id":"F1","doc_id":"...","chunk_id":"...","fact":"exact concise fact"}}],
  "judgments": {{
    "A": {{"verdict":true,"confidence":0.0,"citations":["chunk_id"],"reasoning":"concise"}},
    "B": {{"verdict":false,"confidence":0.0,"citations":["chunk_id"],"reasoning":"concise"}},
    "C": {{"verdict":false,"confidence":0.0,"citations":["chunk_id"],"reasoning":"concise"}},
    "D": {{"verdict":false,"confidence":0.0,"citations":["chunk_id"],"reasoning":"concise"}}
  }},
  "proposition_verdict": null,
  "answer": "A",
  "needs_review": false,
  "missing_evidence_queries": [],
  "reason": "one sentence"
}}
""".strip(),
        },
    ]


def compact_primary(primary: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        "answer": primary.get("answer"),
        "proposition_verdict": primary.get("proposition_verdict"),
        "needs_review": primary.get("needs_review"),
        "missing_evidence_queries": primary.get("missing_evidence_queries", []),
    }
    judgments = primary.get("judgments")
    if isinstance(judgments, dict):
        compact_judgments: Dict[str, Any] = {}
        for letter in LETTERS:
            item = judgments.get(letter)
            if not isinstance(item, dict):
                continue
            compact_judgments[letter] = {
                "verdict": item.get("verdict"),
                "confidence": item.get("confidence"),
                "citations": item.get("citations", []),
                "reasoning": str(item.get("reasoning", ""))[:320],
            }
        compact["judgments"] = compact_judgments
    return compact


def review_messages_v6(
    question: Dict[str, Any],
    pack: EvidencePack,
    primary: Dict[str, Any],
    reasons: Sequence[str],
) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a final financial QA verifier. Recalculate disputed claims from source excerpts and "
                "correct only evidence-backed errors. Return compact strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": f"""
Question:
{build_question_text(question)}

Parsed checks:
{json.dumps(question_profile(question), ensure_ascii=False, separators=(',', ':'))}

Literal evidence hints (retrieval aids, not automatic verdicts):
{json.dumps(literal_evidence_hints(question, pack), ensure_ascii=False, separators=(',', ':'))}

Semantic guardrails:
{json.dumps(semantic_guardrails(question), ensure_ascii=False)}

Answer protocol:
{answer_protocol(question)}

Domain checks:
{domain_instructions(question)}

Review triggers:
{', '.join(reasons)}

Primary result (untrusted):
{json.dumps(compact_primary(primary), ensure_ascii=False, separators=(',', ':'))}

Compressed source evidence:
{pack.context}

Independently verify all options. Keep years, units, scope words, inequalities, legal conditions, and exceptions exact. Citations must be supplied chunk_id values.
Treat ranges literally, judge only the option's stated qualifiers, and accept faithful near-verbatim summaries without demanding unnecessary extra detail.

Return strict JSON with keys judgments, proposition_verdict, answer, corrections, reason. Each judgment needs verdict (boolean), confidence, citations, and concise reasoning.
""".strip(),
        },
    ]


def call_required_json(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_completion_tokens: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    parsed, usage = call_json(
        client,
        model,
        messages,
        max_completion_tokens=max_completion_tokens,
    )
    if isinstance(parsed.get("judgments"), dict):
        return parsed, usage
    retry, retry_usage = call_json(
        client,
        model,
        messages,
        max_completion_tokens=max_completion_tokens + 600,
    )
    return retry, add_usage(usage, retry_usage)


def repair_messages_v6(
    question: Dict[str, Any],
    context: str,
    reviewed: Dict[str, Any],
    issues: Sequence[str],
) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Repair an internally inconsistent financial QA JSON result. Recheck source citations and return "
                "a self-consistent final JSON. Do not preserve the old answer when evidence contradicts it."
            ),
        },
        {
            "role": "user",
            "content": f"""
Question:
{build_question_text(question)}

Answer protocol:
{answer_protocol(question)}

Detected problems:
{', '.join(issues)}

Inconsistent result:
{json.dumps(compact_primary(reviewed), ensure_ascii=False, separators=(',', ':'))}

Source evidence:
{context}

Repair rules:
1. For mcq exactly one judgment verdict is true.
2. For tf, A=true proposition and B=false proposition are mutually exclusive; exactly one is true.
3. For multi, answer is exactly the sorted letters with true verdicts.
4. Every true verdict must cite a supplied chunk_id.

Return strict JSON with keys judgments, proposition_verdict, answer, corrections, reason. Each judgment needs verdict, confidence, citations, and concise reasoning.
""".strip(),
        },
    ]


def solve_question_v6(
    client: OpenAI,
    model: str,
    question: Dict[str, Any],
    indexes: Dict[str, DocumentIndex],
) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
    answer_format = str(question.get("answer_format", "multi")).lower()
    primary_pack = retrieve_v6(question, indexes)
    primary, primary_usage = call_required_json(
        client,
        model,
        primary_messages_v6(question, primary_pack),
        2400,
    )
    primary = apply_deterministic_checks(question, primary_pack, primary)
    primary_answer, primary_issues = parse_answer(primary, answer_format)
    reasons = review_reasons(question, primary, primary_issues, primary_pack)

    final = primary
    final_answer = primary_answer
    review_pack: Optional[EvidencePack] = None
    review_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    review_accepted = False
    if reasons:
        review_queries = collect_review_queries(question, primary)
        expanded_pack = retrieve_v6(
            question,
            indexes,
            review=True,
            extra_queries=review_queries,
        )
        review_pack = expanded_pack
        reviewed, review_usage = call_required_json(
            client,
            model,
            review_messages_v6(question, review_pack, primary, reasons),
            2200,
        )
        reviewed = apply_deterministic_checks(question, review_pack, reviewed)
        reviewed_answer, reviewed_issues = parse_answer(reviewed, answer_format)
        valid_ids = {chunk.chunk_id for chunk in review_pack.chunks}
        citation_ratio = valid_citation_ratio(reviewed, valid_ids)
        severe_issues = [
            issue for issue in reviewed_issues
            if issue.startswith(("mcq_true_count_", "tf_verdict_missing", "multi_no_true_judgment"))
            or issue == "answer_verdict_mismatch"
        ]
        if severe_issues or citation_ratio < 1.0:
            repair_triggers = severe_issues or ["true_option_missing_valid_citation"]
            repaired, repair_usage = call_required_json(
                client,
                model,
                repair_messages_v6(question, review_pack.context, reviewed, repair_triggers),
                2200,
            )
            repaired = apply_deterministic_checks(question, review_pack, repaired)
            review_usage = add_usage(review_usage, repair_usage)
            repaired_answer, repaired_issues = parse_answer(repaired, answer_format)
            repaired_ratio = valid_citation_ratio(repaired, valid_ids)
            if repaired_answer and not repaired_issues and repaired_ratio >= 1.0:
                reviewed = repaired
                reviewed_answer = repaired_answer
                reviewed_issues = repaired_issues
                citation_ratio = repaired_ratio
        remaining_severe_issues = [
            issue for issue in reviewed_issues
            if issue.startswith(("mcq_true_count_", "tf_verdict_missing", "multi_no_true_judgment"))
            or issue == "answer_verdict_mismatch"
        ]
        review_accepted = (
            bool(reviewed_answer)
            and citation_ratio >= 1.0
            and not remaining_severe_issues
        )
        if review_accepted:
            final = reviewed
            final_answer = reviewed_answer

    usage = add_usage(primary_usage, review_usage)
    trace = {
        "qid": str(question.get("qid", "")),
        "answer": final_answer,
        "primary_answer": primary_answer,
        "reviewed": bool(reasons),
        "review_accepted": review_accepted,
        "review_reasons": reasons,
        "question_profile": question_profile(question),
        "primary_retrieval": {
            **primary_pack.diagnostics,
            "context_chars": len(primary_pack.context),
        },
        "review_retrieval": review_pack.diagnostics if review_pack else None,
        "primary_output": primary,
        "final_output": final,
        "primary_usage": primary_usage,
        "review_usage": review_usage,
        "usage": usage,
    }
    return final_answer, usage, trace


def main() -> None:
    parser = argparse.ArgumentParser(description="AFAC2026 v6 token-efficient structured RAG solver.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--doc-meta", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-output", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--qid", default="")
    parser.add_argument("--qids", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    indexes = build_document_indexes(Path(args.doc_meta), Path(args.processed_dir))
    questions = normalize_question_items(read_json_any(args.questions))
    if args.qid:
        questions = [question for question in questions if str(question.get("qid", "")) == args.qid]
        if not questions:
            raise ValueError(f"Unknown qid: {args.qid}")
    if args.qids:
        requested_qids = {qid.strip() for qid in args.qids.split(",") if qid.strip()}
        questions = [question for question in questions if str(question.get("qid", "")) in requested_qids]
        found_qids = {str(question.get("qid", "")) for question in questions}
        missing_qids = requested_qids - found_qids
        if missing_qids:
            raise ValueError(f"Unknown qids: {sorted(missing_qids)}")
    if args.limit > 0:
        questions = questions[:args.limit]

    output_path = Path(args.output)
    evidence_path = Path(args.evidence_output) if args.evidence_output else None
    rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    summary = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8-sig", newline="") as source:
            for raw in csv.DictReader(source):
                if raw.get("qid") == "summary":
                    continue
                row = {
                    "qid": str(raw.get("qid", "")),
                    "answer": str(raw.get("answer", "")),
                    "prompt_tokens": int(raw.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(raw.get("completion_tokens", 0) or 0),
                    "total_tokens": int(raw.get("total_tokens", 0) or 0),
                }
                rows.append(row)
                for key in summary:
                    summary[key] += row[key]
        if evidence_path and evidence_path.exists():
            restored = read_json_any(evidence_path)
            if isinstance(restored, list):
                traces = [item for item in restored if isinstance(item, dict)]

    completed = {str(row["qid"]) for row in rows}
    pending = [question for question in questions if str(question.get("qid", "")) not in completed]
    for question in tqdm(pending, desc="solve_v6"):
        answer, usage, trace = solve_question_v6(
            client,
            settings.model,
            question,
            indexes,
        )
        for key in summary:
            summary[key] += usage[key]
        rows.append({"qid": str(question.get("qid", "")), "answer": answer, **usage})
        traces.append(trace)
        write_answer_csv(output_path, rows, summary)
        if evidence_path:
            ensure_dir(evidence_path.parent)
            safe_json_dump(evidence_path, traces)

    write_answer_csv(output_path, rows, summary)
    if evidence_path:
        ensure_dir(evidence_path.parent)
        safe_json_dump(evidence_path, traces)
    print(f"Solved questions: {len(rows)}/{len(questions)}")
    print(f"Reviewed questions: {sum(1 for trace in traces if trace['reviewed'])}")
    print(f"Accepted reviews: {sum(1 for trace in traces if trace['review_accepted'])}")
    print(f"Total tokens: {summary['total_tokens']}")
    print(f"Saved answer csv to: {output_path}")


if __name__ == "__main__":
    main()
