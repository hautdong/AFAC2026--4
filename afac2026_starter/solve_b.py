from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from openai import OpenAI

from .common import (
    build_question_text,
    ensure_dir,
    load_settings,
    normalize_question_items,
    read_json_any,
    safe_json_dump,
    tokenize,
)
from .solve_v4 import (
    DocumentIndex,
    PageChunk,
    add_usage,
    build_document_indexes,
    build_idf,
    call_json,
    domain_instructions,
    rank_document_chunks,
)
from .solve_v6 import retrieve_v6, solve_question_v6

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
EMPTY_USAGE = {key: 0 for key in USAGE_KEYS}
OPEN_TYPES = {"计算题", "抽取题"}


@dataclass
class CandidateDocument:
    doc_id: str
    title: str
    score: float
    chunks: List[PageChunk]


def normalized(text: str) -> str:
    return re.sub(r"[\s，。；：、（）()《》\[\]【】]", "", text).lower()


def load_template(path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    order: List[str] = []
    patterns: Dict[str, List[str]] = {}
    for row in rows:
        qid = str(row.get("qid", ""))
        if not qid or qid == "summary":
            continue
        order.append(qid)
        values = [str(row.get(f"answer_{index}", "")) for index in range(1, 5)]
        patterns[qid] = [value for value in values if value]
    return order, patterns


def metadata_domains(path: Path) -> Dict[str, str]:
    data = read_json_any(path)
    items = data if isinstance(data, list) else data.get("documents", [])
    return {
        str(item.get("doc_id", "")): str(item.get("domain", ""))
        for item in items
        if isinstance(item, dict) and item.get("doc_id")
    }


def build_domain_indexes(
    indexes: Dict[str, DocumentIndex],
    domains: Dict[str, str],
) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for doc_id in indexes:
        domain = domains.get(doc_id, "")
        grouped.setdefault(domain, []).append(doc_id)
    return grouped


def exact_title_bonus(question_text: str, document: DocumentIndex) -> float:
    haystack = normalized(f"{document.doc_id} {document.title}")
    bonus = 0.0
    for title in re.findall(r"《([^》]+)》", question_text):
        title_literal = normalized(title)
        if title_literal and (title_literal in haystack or haystack in title_literal):
            bonus += 800.0
    for phrase in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+(?:公司|集团|银行|证券|保险|报告|办法|规定)", question_text):
        phrase_literal = normalized(phrase)
        if len(phrase_literal) >= 4 and phrase_literal in haystack:
            bonus += 120.0
    return bonus


def rank_candidate_documents(
    question: Dict[str, Any],
    indexes: Dict[str, DocumentIndex],
    domain_doc_ids: Sequence[str],
    limit: int = 10,
) -> List[CandidateDocument]:
    documents = [indexes[doc_id] for doc_id in domain_doc_ids if doc_id in indexes]
    all_chunks = [chunk for document in documents for chunk in document.chunks]
    idf = build_idf(all_chunks)
    query = build_question_text(question)
    query_tokens = set(tokenize(query))
    candidates: List[CandidateDocument] = []
    for document in documents:
        ranked = rank_document_chunks(document, query, idf, 3)
        if not ranked:
            continue
        title_tokens = set(tokenize(f"{document.doc_id} {document.title}"))
        title_overlap = len(query_tokens & title_tokens)
        score = ranked[0].score + sum(chunk.score for chunk in ranked[1:]) * 0.25
        score += title_overlap * 8.0 + exact_title_bonus(query, document)
        candidates.append(CandidateDocument(document.doc_id, document.title, score, ranked))
    candidates.sort(key=lambda item: (-item.score, item.doc_id))
    return candidates[:limit]


def selector_messages(question: Dict[str, Any], candidates: Sequence[CandidateDocument]) -> List[Dict[str, str]]:
    blocks: List[str] = []
    for rank, candidate in enumerate(candidates, start=1):
        snippets = "\n".join(
            f"[{chunk.chunk_id}] {chunk.text[:700]}" for chunk in candidate.chunks[:2]
        )
        blocks.append(
            f"CANDIDATE {rank}\ndoc_id={candidate.doc_id}\ntitle={candidate.title}\n{snippets}"
        )
    return [
        {
            "role": "system",
            "content": (
                "You select source documents for financial long-document QA. Use only lexical candidate titles "
                "and excerpts supplied here. Select every document needed for all options or calculations, but no "
                "irrelevant document. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": f"""
Question:
{build_question_text(question)}

Candidates:
{chr(10).join(blocks)}

Rules:
1. Preserve company, product, regulation, report year, and document title exactly.
2. Cross-company/year questions need all corresponding reports.
3. Select between 1 and 4 doc_ids from the candidates only.
4. A calculation may need several documents even when one snippet already contains a partial value.

Return JSON:
{{"selected_doc_ids":["..."],"reason":"brief","missing_query":"optional"}}
""".strip(),
        },
    ]


def select_documents(
    client: OpenAI,
    model: str,
    question: Dict[str, Any],
    candidates: Sequence[CandidateDocument],
) -> Tuple[List[str], Dict[str, int], Dict[str, Any]]:
    if not candidates:
        raise ValueError(f"No document candidates for {question.get('qid')}")
    parsed, usage = call_json(client, model, selector_messages(question, candidates), max_completion_tokens=700)
    valid_ids = {candidate.doc_id for candidate in candidates}
    raw_ids = parsed.get("selected_doc_ids")
    if isinstance(raw_ids, str):
        raw_ids = re.split(r"[,，\s]+", raw_ids)
    selected: List[str] = []
    if isinstance(raw_ids, list):
        for item in raw_ids:
            doc_id = str(item).strip()
            if doc_id in valid_ids and doc_id not in selected:
                selected.append(doc_id)
    if not selected:
        selected = [candidates[0].doc_id]
    return selected[:4], usage, parsed


def expected_slot_patterns(question: Dict[str, Any], template_patterns: Dict[str, List[str]]) -> List[str]:
    qid = str(question.get("qid", ""))
    patterns = template_patterns.get(qid, [])
    return patterns or (["A"] if question.get("options") else ["999999.99"])


def normalize_open_answers(raw: Any, expected_count: int) -> List[str]:
    if isinstance(raw, (str, int, float)):
        values = [str(raw)]
    elif isinstance(raw, list):
        values = [str(item) for item in raw]
    else:
        values = []
    answers: List[str] = []
    for value in values:
        clean = value.strip().replace("＞", ">").replace("；", ";")
        clean = re.sub(r"\s*>\s*", ">", clean)
        if clean:
            answers.append(clean)
    return answers[:expected_count] if len(answers) >= expected_count else []


def open_primary_messages(
    question: Dict[str, Any],
    context: str,
    patterns: Sequence[str],
) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You solve financial extraction and calculation questions from source excerpts. Keep original "
                "units, perform arithmetic with unrounded source values, and return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": f"""
Question:
{question.get('question', '')}

Domain checks:
{domain_instructions(question)}

Required answer slots and examples from the official template:
{json.dumps(list(patterns), ensure_ascii=False)}

Source excerpts:
{context}

Method:
1. Extract every input value with doc_id, chunk_id, year, and unit.
2. Write formulas and substitute original values; do not round intermediate results.
3. Verify ordering, percentage-point versus percent change, dates, signs, and unit conversion.
4. Return exactly {len(patterns)} answers in the requested order. Do not put units in answers unless the question requires % or a Chinese date.
5. A semicolon in the question separates answer slots; put each part in a separate answers array element.

Return JSON:
{{
  "facts":[{{"doc_id":"...","chunk_id":"...","fact":"..."}}],
  "formula":"...",
  "calculation":"...",
  "answers":["..."],
  "needs_review":false,
  "missing_evidence_queries":[],
  "reason":"brief"
}}
""".strip(),
        },
    ]


def open_review_messages(
    question: Dict[str, Any],
    context: str,
    patterns: Sequence[str],
    primary: Dict[str, Any],
) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You independently verify a financial extraction/calculation result. Recompute from source values, "
                "correct unit or rounding errors, and return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": f"""
Question:
{question.get('question', '')}

Official answer slot examples:
{json.dumps(list(patterns), ensure_ascii=False)}

Untrusted primary result:
{json.dumps(primary, ensure_ascii=False)}

Source excerpts:
{context}

Independently extract inputs and recompute. Use no rounded intermediate values. Return exactly {len(patterns)} ordered answer strings.
Return JSON with keys facts, formula, calculation, answers, corrections, reason.
""".strip(),
        },
    ]


def solve_open_question(
    client: OpenAI,
    model: str,
    question: Dict[str, Any],
    indexes: Dict[str, DocumentIndex],
    patterns: Sequence[str],
) -> Tuple[List[str], Dict[str, int], Dict[str, Any]]:
    pack = retrieve_v6(question, indexes)
    primary, primary_usage = call_json(
        client,
        model,
        open_primary_messages(question, pack.context, patterns),
        max_completion_tokens=2400,
    )
    primary_answers = normalize_open_answers(primary.get("answers"), len(patterns))
    extra_queries = primary.get("missing_evidence_queries")
    if not isinstance(extra_queries, list):
        extra_queries = []
    review_pack = retrieve_v6(question, indexes, review=True, extra_queries=extra_queries[:4])
    reviewed, review_usage = call_json(
        client,
        model,
        open_review_messages(question, review_pack.context, patterns, primary),
        max_completion_tokens=2400,
    )
    reviewed_answers = normalize_open_answers(reviewed.get("answers"), len(patterns))
    answers = reviewed_answers or primary_answers
    if len(answers) != len(patterns):
        raise ValueError(f"Invalid open answers for {question.get('qid')}: {answers}")
    usage = add_usage(primary_usage, review_usage)
    return answers, usage, {
        "primary_retrieval": {**pack.diagnostics, "context_chars": len(pack.context)},
        "review_retrieval": {**review_pack.diagnostics, "context_chars": len(review_pack.context)},
        "primary_output": primary,
        "final_output": reviewed if reviewed_answers else primary,
        "primary_answers": primary_answers,
        "final_answers": answers,
        "primary_usage": primary_usage,
        "review_usage": review_usage,
    }


def write_b_csv(path: Path, rows: Sequence[Dict[str, Any]], summary: Dict[str, int], order: Sequence[str]) -> None:
    ensure_dir(path.parent)
    by_qid = {str(row["qid"]): row for row in rows}
    fieldnames = [
        "qid", "answer_1", "answer_2", "answer_3", "answer_4",
        "prompt_tokens", "completion_tokens", "total_tokens",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"qid": "summary", **summary})
        for qid in order:
            if qid in by_qid:
                writer.writerow(by_qid[qid])


def load_existing(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for raw in csv.DictReader(source):
            if raw.get("qid") == "summary":
                continue
            rows.append(
                {
                    "qid": str(raw.get("qid", "")),
                    **{f"answer_{i}": str(raw.get(f"answer_{i}", "")) for i in range(1, 5)},
                    **{key: int(raw.get(key, 0) or 0) for key in USAGE_KEYS},
                }
            )
    return rows


def row_from_answers(qid: str, answers: Sequence[str], usage: Dict[str, int]) -> Dict[str, Any]:
    return {
        "qid": qid,
        **{f"answer_{i}": answers[i - 1] if i <= len(answers) else "" for i in range(1, 5)},
        **{key: int(usage.get(key, 0)) for key in USAGE_KEYS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AFAC2026 B-board blind-document solver.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--doc-meta", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--qid", default="")
    parser.add_argument("--qids", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart-qid", default="")
    args = parser.parse_args()

    settings = load_settings()
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    questions = normalize_question_items(read_json_any(args.questions))
    template_order, template_patterns = load_template(Path(args.template))
    order_index = {qid: index for index, qid in enumerate(template_order)}
    questions.sort(key=lambda item: order_index.get(str(item.get("qid", "")), len(order_index)))
    if args.qid:
        questions = [item for item in questions if str(item.get("qid", "")) == args.qid]
    if args.qids:
        requested_qids = {
            item.strip() for item in args.qids.split(",") if item.strip()
        }
        questions = [
            item for item in questions
            if str(item.get("qid", "")) in requested_qids
        ]
    if args.limit:
        questions = questions[:args.limit]

    indexes = build_document_indexes(Path(args.doc_meta), Path(args.processed_dir))
    domains = metadata_domains(Path(args.doc_meta))
    domain_indexes = build_domain_indexes(indexes, domains)
    output_path = Path(args.output)
    evidence_path = Path(args.evidence_output)
    rows = load_existing(output_path) if args.resume else []
    traces: List[Dict[str, Any]] = []
    if args.resume and evidence_path.exists():
        restored = read_json_any(evidence_path)
        if isinstance(restored, list):
            traces = [item for item in restored if isinstance(item, dict)]

    if args.restart_qid:
        restart_index = order_index[args.restart_qid]
        rows = [row for row in rows if order_index.get(str(row["qid"]), 9999) < restart_index]
        kept = {str(row["qid"]) for row in rows}
        traces = [trace for trace in traces if str(trace.get("qid", "")) in kept]

    summary = {key: sum(int(row.get(key, 0)) for row in rows) for key in USAGE_KEYS}
    completed = {str(row["qid"]) for row in rows}
    pending = [item for item in questions if str(item.get("qid", "")) not in completed]
    write_b_csv(output_path, rows, summary, template_order)

    for question in tqdm(pending, desc="solve_b"):
        qid = str(question.get("qid", ""))
        domain = str(question.get("domain", ""))
        candidates = rank_candidate_documents(question, indexes, domain_indexes.get(domain, []), limit=10)
        selected_doc_ids, selector_usage, selector_output = select_documents(
            client, settings.model, question, candidates
        )
        working_question = copy.deepcopy(question)
        working_question["doc_ids"] = selected_doc_ids
        patterns = expected_slot_patterns(question, template_patterns)
        if str(question.get("type", "")) in OPEN_TYPES or not question.get("options"):
            answers, solve_usage, solve_trace = solve_open_question(
                client, settings.model, working_question, indexes, patterns
            )
        else:
            answer, solve_usage, solve_trace = solve_question_v6(
                client, settings.model, working_question, indexes
            )
            answers = [answer]
        usage = add_usage(selector_usage, solve_usage)
        for key in USAGE_KEYS:
            summary[key] += usage[key]
        rows.append(row_from_answers(qid, answers, usage))
        traces.append(
            {
                "qid": qid,
                "domain": domain,
                "type": question.get("type"),
                "candidate_documents": [
                    {
                        "doc_id": item.doc_id,
                        "title": item.title,
                        "score": round(item.score, 3),
                        "top_chunks": [chunk.chunk_id for chunk in item.chunks],
                    }
                    for item in candidates
                ],
                "selected_doc_ids": selected_doc_ids,
                "selector_output": selector_output,
                "selector_usage": selector_usage,
                "answers": answers,
                "solve": solve_trace,
                "usage": usage,
            }
        )
        write_b_csv(output_path, rows, summary, template_order)
        ensure_dir(evidence_path.parent)
        safe_json_dump(evidence_path, traces)

    print(f"Solved B questions: {len(rows)}/{len(questions)}")
    print(f"Total tokens: {summary['total_tokens']}")
    print(f"Saved B answer CSV to: {output_path}")


if __name__ == "__main__":
    main()
