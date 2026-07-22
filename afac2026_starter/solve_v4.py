from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from openai import OpenAI

from .common import (
    build_question_text,
    clean_layout_text,
    clean_text,
    ensure_dir,
    load_settings,
    normalize_answer,
    normalize_doc_items,
    normalize_question_items,
    read_json_any,
    safe_json_dump,
    tokenize,
)
from .solve import DOMAIN_GUIDANCE, write_answer_csv

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:  # type: ignore
        return iterable


LETTERS = ("A", "B", "C", "D")
NUMBER_ANCHOR_RE = re.compile(
    r"(?:\d{4}\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?)?"
    r"|\d+(?:\.\d+)?\s*(?:%|％|亿元|万元|千元|元|亿|万|年|个月|月|日|"
    r"个工作日|个自然日|倍|股|份|人|项|级|AAA|AA\+|AA))",
    re.IGNORECASE,
)
ARTICLE_ANCHOR_RE = re.compile(
    r"(?:第[一二三四五六七八九十百千零〇\d]+条|\d+(?:\.\d+){1,3})"
)
ENTITY_ANCHOR_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9（）()·]{2,24}(?:公司|集团|银行|证券|保险|办法|规定|条例|报告|法)"
)
CLAUSE_SPLIT_RE = re.compile(r"[，。；：、！？\s]+")
FINANCIAL_KEY_PHRASES = (
    "本期发行金额",
    "本次发行金额",
    "本期发行规模",
    "本次发行规模",
    "注册金额",
    "注册规模",
    "主体信用评级",
    "债项信用评级",
    "营业收入",
    "归属于上市公司股东的净利润",
    "经营活动产生的现金流量净额",
    "研发投入占营业收入",
    "年度现金分红",
    "特别现金分红",
    "现金价值",
    "保单账户价值",
    "保险责任",
    "责任免除",
    "受益所有人",
    "客户尽职调查",
)
SCOPE_TERM_GROUPS = (
    ("本期", "本次", "注册"),
    ("主体信用评级", "债项信用评级"),
    ("年度现金分红", "特别现金分红", "现金分红"),
    ("同比", "环比"),
    ("至少", "至多", "不超过", "超过"),
    ("工作日", "自然日", "日"),
)


@dataclass
class PageChunk:
    doc_id: str
    title: str
    doc_order: int
    page: int
    chunk_order: int
    chunk_id: str
    text: str
    tokens: Counter = field(default_factory=Counter)
    score: float = 0.0
    matched_for: Set[str] = field(default_factory=set)


@dataclass
class DocumentIndex:
    doc_id: str
    title: str
    pages: List[Dict[str, Any]]
    chunks: List[PageChunk]


@dataclass
class EvidencePack:
    chunks: List[PageChunk]
    context: str
    diagnostics: Dict[str, Any]


def split_page_text(text: str, target_size: int = 1150, overlap: int = 220) -> List[str]:
    text = clean_layout_text(text)
    if not text:
        return []
    if len(text) <= target_size:
        return [text]

    lines = text.splitlines()
    if len(lines) > 1:
        chunks: List[str] = []
        start_line = 0
        while start_line < len(lines):
            end_line = start_line
            current_size = 0
            while end_line < len(lines):
                next_size = len(lines[end_line]) + (1 if current_size else 0)
                if current_size and current_size + next_size > target_size:
                    break
                current_size += next_size
                end_line += 1
                if current_size >= target_size:
                    break
            if end_line == start_line:
                end_line += 1
            chunk = "\n".join(lines[start_line:end_line])
            if len(chunk) > target_size * 1.5 and end_line == start_line + 1:
                chunks.extend(split_page_text(clean_text(chunk), target_size=target_size, overlap=overlap))
            else:
                chunks.append(chunk)
            if end_line >= len(lines):
                break
            overlap_size = 0
            next_start = end_line
            while next_start > start_line and overlap_size < overlap:
                next_start -= 1
                overlap_size += len(lines[next_start]) + 1
            start_line = max(start_line + 1, next_start)
        return chunks

    chunks: List[str] = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + target_size)
        end = hard_end
        if hard_end < len(text):
            search_start = max(start + target_size // 2, hard_end - 180)
            boundary = max(
                text.rfind("。", search_start, hard_end),
                text.rfind("；", search_start, hard_end),
                text.rfind("，", search_start, hard_end),
            )
            if boundary >= search_start:
                end = boundary + 1
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def build_document_indexes(
    doc_meta_path: Path,
    processed_dir: Path,
) -> Dict[str, DocumentIndex]:
    metadata = normalize_doc_items(read_json_any(doc_meta_path))
    meta_by_id = {
        str(item.get("doc_id", "")).strip(): item
        for item in metadata
        if str(item.get("doc_id", "")).strip()
    }
    indexes: Dict[str, DocumentIndex] = {}
    for path in sorted(processed_dir.glob("*.json")):
        raw = read_json_any(path)
        doc_id = str(raw.get("doc_id", "")).strip()
        if not doc_id:
            continue
        meta = meta_by_id.get(doc_id, {})
        metadata_title = str(meta.get("title", "")).strip()
        extracted_title = str(raw.get("title", "")).strip()
        if metadata_title and metadata_title != doc_id:
            title = metadata_title
        else:
            title = extracted_title or metadata_title or doc_id
        pages = raw.get("pages") if isinstance(raw.get("pages"), list) else []
        if not pages:
            pages = [{"page": 1, "text": str(raw.get("text", ""))}]

        chunks: List[PageChunk] = []
        chunk_order = 0
        for page_item in pages:
            page_number = int(page_item.get("page", 0) or 0)
            page_text = str(page_item.get("search_text") or page_item.get("text", ""))
            for page_part in split_page_text(page_text):
                chunk_id = f"{doc_id}::p{page_number}::c{chunk_order}"
                chunks.append(
                    PageChunk(
                        doc_id=doc_id,
                        title=title,
                        doc_order=0,
                        page=page_number,
                        chunk_order=chunk_order,
                        chunk_id=chunk_id,
                        text=page_part,
                        tokens=Counter(tokenize(page_part)),
                    )
                )
                chunk_order += 1
        indexes[doc_id] = DocumentIndex(
            doc_id=doc_id,
            title=title,
            pages=pages,
            chunks=chunks,
        )
    return indexes


def extract_anchors(text: str) -> List[str]:
    text = clean_text(text)
    anchors: Set[str] = set()
    anchors.update(match.group(0).replace(" ", "") for match in NUMBER_ANCHOR_RE.finditer(text))
    anchors.update(match.group(0).replace(" ", "") for match in ARTICLE_ANCHOR_RE.finditer(text))
    anchors.update(match.group(0).strip() for match in ENTITY_ANCHOR_RE.finditer(text))
    anchors.update(phrase for phrase in FINANCIAL_KEY_PHRASES if phrase in text)
    for clause in CLAUSE_SPLIT_RE.split(text):
        clause = clause.strip()
        if 4 <= len(clause) <= 18 and not clause.startswith(("answer_format=", "domain=", "type=")):
            anchors.add(clause)
    return sorted(anchors, key=lambda item: (-len(item), item))


def build_idf(chunks: Sequence[PageChunk]) -> Dict[str, float]:
    document_frequency: Counter = Counter()
    for chunk in chunks:
        document_frequency.update(set(chunk.tokens))
    total = max(1, len(chunks))
    return {
        token: math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
        for token, frequency in document_frequency.items()
    }


def score_chunk(
    chunk: PageChunk,
    query_text: str,
    idf: Dict[str, float],
) -> float:
    query_tokens = Counter(tokenize(query_text))
    score = 0.0
    length_norm = 1.0 + max(0, len(chunk.text) - 900) / 3000.0
    for token, query_count in query_tokens.items():
        if len(token) <= 1 or token not in chunk.tokens:
            continue
        term_frequency = min(3, chunk.tokens[token])
        score += idf.get(token, 0.2) * term_frequency * min(2, query_count)

    normalized_text = chunk.text.replace(" ", "").lower()
    for anchor in extract_anchors(query_text):
        normalized_anchor = anchor.replace(" ", "").lower()
        if normalized_anchor and normalized_anchor in normalized_text:
            if NUMBER_ANCHOR_RE.fullmatch(anchor):
                score += 12.0
            elif ARTICLE_ANCHOR_RE.fullmatch(anchor):
                score += 11.0
            elif anchor.endswith(("公司", "集团", "银行", "证券", "保险", "办法", "规定", "条例", "报告", "法")):
                score += 8.0
            else:
                score += 5.0
    return score / length_norm


def rank_document_chunks(
    document: DocumentIndex,
    query_text: str,
    idf: Dict[str, float],
    limit: int,
) -> List[PageChunk]:
    ranked: List[PageChunk] = []
    for original in document.chunks:
        score = score_chunk(original, query_text, idf)
        if score <= 0:
            continue
        ranked.append(
            PageChunk(
                doc_id=original.doc_id,
                title=original.title,
                doc_order=original.doc_order,
                page=original.page,
                chunk_order=original.chunk_order,
                chunk_id=original.chunk_id,
                text=original.text,
                tokens=original.tokens,
                score=score,
            )
        )
    ranked.sort(key=lambda item: (-item.score, item.page, item.chunk_order))
    return ranked[:limit]


def question_doc_ids(
    question: Dict[str, Any],
    indexes: Dict[str, DocumentIndex],
) -> List[str]:
    explicit = [str(item) for item in question.get("doc_ids", [])]
    return [doc_id for doc_id in explicit if doc_id in indexes]


def clone_with_order(chunk: PageChunk, doc_order: int, matched_for: str) -> PageChunk:
    return PageChunk(
        doc_id=chunk.doc_id,
        title=chunk.title,
        doc_order=doc_order,
        page=chunk.page,
        chunk_order=chunk.chunk_order,
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        tokens=chunk.tokens,
        score=chunk.score,
        matched_for={matched_for},
    )


def select_evidence(
    question: Dict[str, Any],
    indexes: Dict[str, DocumentIndex],
    max_chars: int,
    extra_queries: Optional[Sequence[str]] = None,
    group_limit: int = 2,
    selection_budget: int = 16,
    continuation_limit: int = 3,
    include_early_summary: bool = True,
) -> EvidencePack:
    doc_ids = question_doc_ids(question, indexes)
    if not doc_ids:
        raise ValueError(f"No valid doc_ids for qid={question.get('qid', '')}")
    all_chunks = [chunk for doc_id in doc_ids for chunk in indexes[doc_id].chunks]
    idf = build_idf(all_chunks)
    options = question.get("options", {}) if isinstance(question.get("options"), dict) else {}
    stem = str(question.get("question", ""))
    global_query = build_question_text(question)
    query_groups: List[Tuple[str, str]] = [("GLOBAL", global_query)]
    for letter in LETTERS:
        if letter in options:
            query_groups.append((letter, f"{stem}\n{letter}. {options[letter]}"))
    for index, query in enumerate(extra_queries or []):
        if str(query).strip():
            query_groups.append((f"REVIEW_{index + 1}", f"{stem}\n{query}"))

    selected_by_doc: Dict[str, Dict[str, PageChunk]] = defaultdict(dict)
    per_doc_limit = max(3, min(7, selection_budget // max(1, len(doc_ids))))
    for doc_order, doc_id in enumerate(doc_ids, start=1):
        document = indexes[doc_id]
        candidate_map: Dict[str, PageChunk] = {}
        front_chunk = document.chunks[0] if document.chunks else None
        early_chunks = [chunk for chunk in document.chunks if chunk.page <= 5]
        early_ranked = (
            rank_document_chunks(
                DocumentIndex(document.doc_id, document.title, document.pages, early_chunks),
                global_query,
                idf,
                1,
            )
            if include_early_summary
            else []
        )
        for group_name, query_text in query_groups:
            for chunk in rank_document_chunks(document, query_text, idf, group_limit):
                existing = candidate_map.get(chunk.chunk_id)
                if existing is None:
                    candidate_map[chunk.chunk_id] = clone_with_order(chunk, doc_order, group_name)
                else:
                    existing.score = max(existing.score, chunk.score)
                    existing.matched_for.add(group_name)

        ranked_candidates = sorted(
            candidate_map.values(),
            key=lambda item: (-len(item.matched_for), -item.score, item.page, item.chunk_order),
        )
        if front_chunk is not None:
            existing_front = candidate_map.get(front_chunk.chunk_id)
            if existing_front is None:
                existing_front = clone_with_order(front_chunk, doc_order, "DOC_HEADER")
                existing_front.score = score_chunk(front_chunk, global_query, idf)
            else:
                existing_front.matched_for.add("DOC_HEADER")
            selected_by_doc[doc_id][existing_front.chunk_id] = existing_front
        if early_ranked:
            early = early_ranked[0]
            existing_early = candidate_map.get(early.chunk_id)
            if existing_early is None:
                existing_early = clone_with_order(early, doc_order, "DOC_SUMMARY")
            else:
                existing_early.matched_for.add("DOC_SUMMARY")
            selected_by_doc[doc_id][existing_early.chunk_id] = existing_early
        for chunk in ranked_candidates:
            if len(selected_by_doc[doc_id]) >= per_doc_limit:
                break
            selected_by_doc[doc_id][chunk.chunk_id] = chunk

        # Tables and long clauses often continue into the next extracted chunk/page.
        # Carry a few continuations so a hit on a heading cannot hide the decisive value.
        chunk_position = {
            chunk.chunk_id: position
            for position, chunk in enumerate(document.chunks)
        }
        neighbor_sources = sorted(
            selected_by_doc[doc_id].values(),
            key=lambda item: (-len(item.matched_for - {"GLOBAL", "DOC_HEADER"}), -item.score),
        )
        neighbors_added = 0
        for source in neighbor_sources:
            if not (source.matched_for - {"GLOBAL", "DOC_HEADER"}):
                continue
            position = chunk_position.get(source.chunk_id)
            if position is None or position + 1 >= len(document.chunks):
                continue
            neighbor = document.chunks[position + 1]
            if neighbor.chunk_id in selected_by_doc[doc_id]:
                continue
            carried = clone_with_order(neighbor, doc_order, "CONTINUATION")
            carried.score = source.score * 0.9
            selected_by_doc[doc_id][carried.chunk_id] = carried
            neighbors_added += 1
            if neighbors_added >= continuation_limit:
                break

    selected = [
        chunk
        for doc_id in doc_ids
        for chunk in sorted(
            selected_by_doc[doc_id].values(),
            key=lambda item: (item.page, item.chunk_order),
        )
    ]
    context = format_context(selected, doc_ids, max_chars)
    diagnostics = {
        "doc_ids": doc_ids,
        "chunk_count": len(selected),
        "chunks_per_doc": {
            doc_id: len(selected_by_doc[doc_id])
            for doc_id in doc_ids
        },
        "selected_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "page": chunk.page,
                "score": round(chunk.score, 3),
                "matched_for": sorted(chunk.matched_for),
                "text_preview": chunk.text[:260],
            }
            for chunk in selected
        ],
    }
    return EvidencePack(chunks=selected, context=context, diagnostics=diagnostics)


def format_context(chunks: Sequence[PageChunk], doc_ids: Sequence[str], max_chars: int) -> str:
    chunks_by_doc: Dict[str, List[PageChunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk.doc_id].append(chunk)

    parts: List[str] = []
    used = 0
    for doc_order, doc_id in enumerate(doc_ids, start=1):
        doc_chunks = chunks_by_doc.get(doc_id, [])
        title = doc_chunks[0].title if doc_chunks else doc_id
        heading = f"## DOCUMENT {doc_order}\ndoc_id={doc_id}\ntitle={title}\n"
        if used + len(heading) > max_chars:
            break
        parts.append(heading)
        used += len(heading)
        for chunk in doc_chunks:
            labels = ",".join(sorted(chunk.matched_for))
            block = (
                f"[chunk_id={chunk.chunk_id}] [page={chunk.page}] "
                f"[matched_for={labels}]\n{chunk.text}\n"
            )
            if used + len(block) > max_chars:
                continue
            parts.append(block)
            used += len(block)
    return "\n".join(parts)


def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def call_json(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_completion_tokens: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    response = None
    last_error: Optional[Exception] = None
    for attempt, delay in enumerate((0, 3, 10), start=1):
        if delay:
            time.sleep(delay)
        try:
            request: Dict[str, Any] = {
                "model": model,
                "temperature": 0,
                "messages": messages,
                "response_format": {"type": "json_object"},
            }
            if max_completion_tokens is not None:
                request["max_tokens"] = max_completion_tokens
            response = client.chat.completions.create(
                **request,
            )
            break
        except Exception as error:  # SDK exposes different transient error classes by version.
            last_error = error
            if attempt == 3:
                raise
    if response is None:
        raise RuntimeError("Qwen request failed") from last_error
    parsed = safe_json_loads(response.choices[0].message.content or "{}")
    usage = {
        "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
    }
    return parsed, usage


def domain_instructions(question: Dict[str, Any]) -> str:
    domain = str(question.get("domain", ""))
    base = DOMAIN_GUIDANCE.get(domain, "Compare every option literally against the supplied source text.")
    supplements = {
        "financial_contracts": (
            "Map 'first document' and 'second document' to DOCUMENT 1 and DOCUMENT 2. "
            "Do not confuse issuer rating with issue rating, total registration size with current issue size, "
            "or lead underwriter with trustee."
        ),
        "financial_reports": (
            "Build a year-by-year metric table before judging. Preserve units and signs. "
            "Distinguish a proposed dividend from a paid dividend and annual dividend from total dividend. "
            "However, 'implement a dividend policy/plan' does not mean every payment is already completed: "
            "an annual-report statement such as 'continue the policy' or 'for consecutive years implement "
            "cash dividends at X% of net profit' directly supports policy implementation, even when the current "
            "distribution proposal still awaits shareholder approval. Require completed payment only when the "
            "option explicitly says paid, distributed, or completed."
        ),
        "insurance": (
            "Write the applicable benefit formula before substituting numbers. Check trigger, waiting period, "
            "deductible, compensation from other insurers, exclusions, age, and policy stage. "
            "When an option displays a numeric ranking with >, <, or =, evaluate every displayed inequality "
            "after calculating the values. The products may be reordered; never treat the sequence as merely "
            "a product-to-value mapping. For example, 90 > 144 is false even if both labels have the right values."
        ),
        "regulatory": (
            "Quote the exact article condition, actor, deadline, modal verb, effective date, and exception. "
            "Absence in one regulation is not proof that a statement is false unless the claim requires both documents."
        ),
        "research": (
            "Keep actual values separate from forecasts and company metrics separate from industry metrics. "
            "Verify the forecast year, geography, currency, and CAGR interval."
        ),
    }
    return f"{base} {supplements.get(domain, '')}".strip()


def answer_protocol(question: Dict[str, Any]) -> str:
    answer_format = str(question.get("answer_format", "multi")).lower()
    if answer_format == "tf":
        return (
            "This is a true/false question. Judge the proposition in the question once. "
            "If true, proposition_verdict=true and answer=A. If false, proposition_verdict=false and answer=B. "
            "A and B are labels, not two independent factual claims."
        )
    if answer_format == "mcq":
        return (
            "This is single-choice. Compare all options and select exactly one best answer. "
            "Exactly one judgment verdict must be true."
        )
    return (
        "This is multiple-choice. Judge A/B/C/D independently and include every supported option. "
        "At least one option is expected to be correct; do not reject a derivable option merely because the source "
        "does not repeat its wording verbatim. Judge whether each option's own claims are true, not whether the "
        "option exhaustively restates every related condition in the source. An omitted additional cap, exception, "
        "or implementation detail does not make a true statement false unless the option claims exclusivity, "
        "sufficiency, or an unconditional result. A source phrase such as '专项资管计划等' supports an option that "
        "names 专项资管计划 as the implemented method; the option need not repeat '等'."
    )


def primary_messages(question: Dict[str, Any], context: str) -> List[Dict[str, str]]:
    question_text = build_question_text(question)
    doc_mapping = ", ".join(
        f"DOCUMENT {index}={doc_id}"
        for index, doc_id in enumerate(question.get("doc_ids", []), start=1)
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a financial long-document evidence analyst. Use only the supplied source excerpts. "
                "First compress exact facts into a small evidence memory, then reason. Preserve numbers, units, "
                "years, negations, conditions, and document identity. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": f"""
Question:
{question_text}

Document order:
{doc_mapping}

Domain checks:
{domain_instructions(question)}

Answer protocol:
{answer_protocol(question)}

Source excerpts:
{context}

Required reasoning:
1. Decompose each option into atomic claims. A claim containing 'both', 'all', or a comparison must be checked against every relevant document.
2. For numbers, write the formula and substitutions. For regulation/contract terms, preserve the actor, trigger, deadline, modal verb, and exception.
3. A missing excerpt is uncertainty, not negative evidence. Set needs_review=true and provide a targeted missing_evidence_query when decisive evidence is absent.
4. Citations must use supplied chunk_id values. Keep each fact and reasoning concise.

Return this JSON shape:
{{
  "memory": {{
    "A": [{{"doc_id": "...", "chunk_id": "...", "fact": "exact fact", "calculation": "optional"}}],
    "B": [], "C": [], "D": []
  }},
  "judgments": {{
    "A": {{"verdict": true, "confidence": 0.0, "citations": ["chunk_id"], "reasoning": "brief"}},
    "B": {{"verdict": false, "confidence": 0.0, "citations": ["chunk_id"], "reasoning": "brief"}},
    "C": {{"verdict": false, "confidence": 0.0, "citations": ["chunk_id"], "reasoning": "brief"}},
    "D": {{"verdict": false, "confidence": 0.0, "citations": ["chunk_id"], "reasoning": "brief"}}
  }},
  "proposition_verdict": null,
  "answer": "A",
  "needs_review": false,
  "missing_evidence_queries": [],
  "reason": "brief final reason"
}}
""".strip(),
        },
    ]


def judgment_letters(parsed: Dict[str, Any]) -> List[str]:
    judgments = parsed.get("judgments")
    if not isinstance(judgments, dict):
        return []
    result: List[str] = []
    for letter in LETTERS:
        item = judgments.get(letter)
        if isinstance(item, dict) and item.get("verdict") is True:
            result.append(letter)
    return result


def parse_answer(parsed: Dict[str, Any], answer_format: str) -> Tuple[str, List[str]]:
    issues: List[str] = []
    raw_answer = normalize_answer(str(parsed.get("answer", "")), answer_format)
    true_letters = judgment_letters(parsed)

    if answer_format == "tf":
        proposition = parsed.get("proposition_verdict")
        if isinstance(proposition, bool):
            derived = "A" if proposition else "B"
        elif len(true_letters) == 1 and true_letters[0] in {"A", "B"}:
            derived = true_letters[0]
        else:
            derived = raw_answer if raw_answer in {"A", "B"} else "A"
            issues.append("tf_verdict_missing")
        if raw_answer != derived:
            issues.append("answer_verdict_mismatch")
        return derived, issues

    if answer_format == "mcq":
        if len(true_letters) == 1:
            derived = true_letters[0]
        else:
            derived = raw_answer
            issues.append(f"mcq_true_count_{len(true_letters)}")
        if raw_answer != derived:
            issues.append("answer_verdict_mismatch")
        return derived, issues

    if true_letters:
        derived = "".join(true_letters)
    else:
        derived = raw_answer
        issues.append("multi_no_true_judgment")
    if raw_answer != derived:
        issues.append("answer_verdict_mismatch")
    return derived, issues


def confidence_values(parsed: Dict[str, Any]) -> List[float]:
    judgments = parsed.get("judgments")
    if not isinstance(judgments, dict):
        return []
    values: List[float] = []
    for letter in LETTERS:
        item = judgments.get(letter)
        if not isinstance(item, dict):
            continue
        try:
            values.append(float(item.get("confidence", 0)))
        except (TypeError, ValueError):
            values.append(0.0)
    return values


def valid_citation_ratio(parsed: Dict[str, Any], valid_chunk_ids: Set[str]) -> float:
    judgments = parsed.get("judgments")
    if not isinstance(judgments, dict):
        return 0.0
    supported = 0
    cited = 0
    for letter in LETTERS:
        item = judgments.get(letter)
        if not isinstance(item, dict):
            continue
        citations = item.get("citations") if isinstance(item.get("citations"), list) else []
        if item.get("verdict") is True:
            supported += 1
            if any(str(citation) in valid_chunk_ids for citation in citations):
                cited += 1
    return cited / supported if supported else 0.0


def scope_citation_issues(
    question: Dict[str, Any],
    parsed: Dict[str, Any],
    pack: EvidencePack,
) -> List[str]:
    options = question.get("options")
    judgments = parsed.get("judgments")
    if not isinstance(options, dict) or not isinstance(judgments, dict):
        return []
    text_by_id = {chunk.chunk_id: chunk.text for chunk in pack.chunks}
    issues: List[str] = []
    for letter in LETTERS:
        option = str(options.get(letter, ""))
        judgment = judgments.get(letter)
        if not option or not isinstance(judgment, dict):
            continue
        citations = judgment.get("citations") if isinstance(judgment.get("citations"), list) else []
        cited_text = " ".join(text_by_id.get(str(citation), "") for citation in citations)
        for group in SCOPE_TERM_GROUPS:
            required_terms = [term for term in group if term in option]
            if required_terms and not any(term in cited_text for term in required_terms):
                issues.append(f"scope_not_cited_{letter}_{required_terms[0]}")
                break
    return issues


def review_reasons(
    question: Dict[str, Any],
    parsed: Dict[str, Any],
    parse_issues: Sequence[str],
    pack: EvidencePack,
) -> List[str]:
    reasons = list(parse_issues)
    confidence = confidence_values(parsed)
    answer_format = str(question.get("answer_format", "multi")).lower()
    if answer_format == "tf":
        confidence_is_low = not confidence or max(confidence) < 0.68
    else:
        confidence_is_low = len(confidence) < 2 or min(confidence, default=0.0) < 0.68
    if confidence_is_low:
        reasons.append("low_confidence")
    if parsed.get("needs_review") is True:
        reasons.append("model_requested_review")
    valid_ids = {chunk.chunk_id for chunk in pack.chunks}
    if valid_citation_ratio(parsed, valid_ids) < 1.0:
        reasons.append("unsupported_true_option")
    reasons.extend(scope_citation_issues(question, parsed, pack))

    domain = str(question.get("domain", ""))
    question_text = build_question_text(question)
    has_calculation = bool(NUMBER_ANCHOR_RE.search(question_text))
    if domain in {"financial_reports", "insurance"} and has_calculation:
        reasons.append("numeric_domain_check")
    return sorted(set(reasons))


def collect_review_queries(question: Dict[str, Any], parsed: Dict[str, Any]) -> List[str]:
    queries: List[str] = []
    raw_queries = parsed.get("missing_evidence_queries")
    if isinstance(raw_queries, list):
        queries.extend(str(item).strip() for item in raw_queries if str(item).strip())

    judgments = parsed.get("judgments")
    options = question.get("options", {})
    if isinstance(judgments, dict) and isinstance(options, dict):
        ranked: List[Tuple[float, str]] = []
        for letter in LETTERS:
            item = judgments.get(letter)
            if letter not in options or not isinstance(item, dict):
                continue
            try:
                confidence = float(item.get("confidence", 0))
            except (TypeError, ValueError):
                confidence = 0.0
            ranked.append((confidence, f"{letter}. {options[letter]}"))
        queries.extend(text for _, text in sorted(ranked)[:2])
    return list(dict.fromkeys(queries))[:4]


def review_messages(
    question: Dict[str, Any],
    context: str,
    primary: Dict[str, Any],
    reasons: Sequence[str],
) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the final verifier for financial document QA. Re-evaluate the answer from source excerpts, "
                "not from the previous conclusion. Correct retrieval mistakes, arithmetic, year/unit confusion, "
                "scope errors, and answer-format inconsistency. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": f"""
Question:
{build_question_text(question)}

Answer protocol:
{answer_protocol(question)}

Domain checks:
{domain_instructions(question)}

Review triggers:
{', '.join(reasons)}

Primary analysis:
{json.dumps(primary, ensure_ascii=False, separators=(',', ':'))}

Source excerpts (including targeted re-retrieval):
{context}

Independently verify every decisive claim. A missing phrase is not a contradiction. For comparisons, show both values; for calculations, show the formula; for rules, retain all conditions and exceptions. Citations must be supplied chunk_id values.

Return strict JSON with keys: judgments, proposition_verdict, answer, corrections, reason.
Each judgment value must contain verdict (JSON boolean), confidence (0 to 1), citations (chunk_id list), and reasoning. Follow the question's tf/mcq/multi constraint exactly.
""".strip(),
        },
    ]


def add_usage(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    return {
        key: int(left.get(key, 0)) + int(right.get(key, 0))
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def solve_question(
    client: OpenAI,
    model: str,
    question: Dict[str, Any],
    indexes: Dict[str, DocumentIndex],
    max_context_chars: int,
) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
    answer_format = str(question.get("answer_format", "multi")).lower()
    primary_pack = select_evidence(question, indexes, max_context_chars)
    primary, usage = call_json(client, model, primary_messages(question, primary_pack.context))
    primary_answer, primary_issues = parse_answer(primary, answer_format)
    reasons = review_reasons(question, primary, primary_issues, primary_pack)

    final = primary
    final_answer = primary_answer
    review_pack: Optional[EvidencePack] = None
    review_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if reasons:
        review_queries = collect_review_queries(question, primary)
        review_pack = select_evidence(
            question,
            indexes,
            max_context_chars,
            extra_queries=review_queries,
        )
        reviewed, review_usage = call_json(
            client,
            model,
            review_messages(question, review_pack.context, primary, reasons),
        )
        reviewed_answer, reviewed_issues = parse_answer(reviewed, answer_format)
        if not reviewed_issues or reviewed_answer:
            final = reviewed
            final_answer = reviewed_answer
    total_usage = add_usage(usage, review_usage)
    trace = {
        "qid": str(question.get("qid", "")),
        "answer": final_answer,
        "primary_answer": primary_answer,
        "reviewed": bool(reasons),
        "review_reasons": reasons,
        "primary_retrieval": primary_pack.diagnostics,
        "review_retrieval": review_pack.diagnostics if review_pack else None,
        "primary_output": primary,
        "final_output": final,
        "usage": total_usage,
    }
    return final_answer, total_usage, trace


def main() -> None:
    parser = argparse.ArgumentParser(description="AFAC2026 v4 page-aware structured RAG solver.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--doc-meta", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-output", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--qid", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart-qid", default="")
    args = parser.parse_args()

    settings = load_settings()
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    indexes = build_document_indexes(Path(args.doc_meta), Path(args.processed_dir))
    questions = normalize_question_items(read_json_any(args.questions))
    if args.qid:
        questions = [question for question in questions if str(question.get("qid", "")) == args.qid]
        if not questions:
            raise ValueError(f"Unknown qid: {args.qid}")
    if args.limit > 0:
        questions = questions[:args.limit]

    output_path = Path(args.output)
    evidence_path = Path(args.evidence_output) if args.evidence_output else None
    rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    summary = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                if row.get("qid") == "summary":
                    continue
                restored = {
                    "qid": str(row.get("qid", "")),
                    "answer": str(row.get("answer", "")),
                    "prompt_tokens": int(row.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(row.get("completion_tokens", 0) or 0),
                    "total_tokens": int(row.get("total_tokens", 0) or 0),
                }
                rows.append(restored)
                for key in summary:
                    summary[key] += restored[key]
        if evidence_path and evidence_path.exists():
            restored_traces = read_json_any(evidence_path)
            if isinstance(restored_traces, list):
                traces = [item for item in restored_traces if isinstance(item, dict)]

    if args.restart_qid:
        question_order = {
            str(question.get("qid", "")): index
            for index, question in enumerate(questions)
        }
        if args.restart_qid not in question_order:
            raise ValueError(f"Unknown restart qid: {args.restart_qid}")
        restart_index = question_order[args.restart_qid]
        rows = [
            row for row in rows
            if question_order.get(str(row.get("qid", "")), len(questions)) < restart_index
        ]
        kept_qids = {str(row.get("qid", "")) for row in rows}
        traces = [trace for trace in traces if str(trace.get("qid", "")) in kept_qids]
        summary = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for row in rows:
            for key in summary:
                summary[key] += int(row.get(key, 0))
        write_answer_csv(output_path, rows, summary)
        if evidence_path:
            safe_json_dump(evidence_path, traces)

    completed_qids = {str(row["qid"]) for row in rows}
    remaining_questions = [
        question
        for question in questions
        if str(question.get("qid", "")) not in completed_qids
    ]
    for question in tqdm(remaining_questions, desc="solve_v4"):
        answer, usage, trace = solve_question(
            client,
            settings.model,
            question,
            indexes,
            settings.max_context_chars,
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
    print(f"Saved answer csv to: {args.output}")
    print(f"Reviewed questions: {sum(1 for trace in traces if trace['reviewed'])}/{len(traces)}")
    print(f"Total tokens: {summary['total_tokens']}")
    if evidence_path:
        ensure_dir(evidence_path.parent)
        safe_json_dump(evidence_path, traces)
        print(f"Saved evidence json to: {evidence_path}")


if __name__ == "__main__":
    main()
