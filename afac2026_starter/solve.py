from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from .common import (
    build_question_text,
    clean_text,
    ensure_dir,
    load_settings,
    normalize_answer,
    normalize_answer_from_judgments,
    normalize_doc_items,
    normalize_question_items,
    read_json_any,
    safe_json_dump,
    score_text,
    tokenize,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


@dataclass
class Chunk:
    doc_id: str
    title: str
    chunk_id: str
    text: str
    score: int = 0


@dataclass
class EvidencePack:
    global_chunks: List[Chunk]
    option_chunks: Dict[str, List[Chunk]]
    all_chunks: List[Chunk]


DOMAIN_GUIDANCE = {
    "financial_contracts": (
        "Bond and contract questions: verify issuer names, issue size, rating, "
        "interest dates, redemption/conversion terms, trustees, and approval clauses exactly."
    ),
    "financial_reports": (
        "Financial report questions: compare the exact years, units, signs, growth direction, "
        "cash flow, R&D ratio, dividends, and shareholder-return metrics. Calculate when needed."
    ),
    "insurance": (
        "Insurance questions: check trigger conditions, exclusions, paid premiums, cash value, "
        "account value, benefit formulas, surrender rules, and claim timing."
    ),
    "regulatory": (
        "Regulatory questions: base each option on explicit legal articles, effective dates, "
        "reporting deadlines, approval thresholds, and whether ordinary or special resolutions apply."
    ),
    "research": (
        "Research report questions: identify the exact report conclusion, compared companies, "
        "industry trend, metric definition, and whether the option overstates the report."
    ),
}


def load_processed_docs(processed_dir: Path) -> Dict[str, Dict[str, Any]]:
    docs: Dict[str, Dict[str, Any]] = {}
    for path in processed_dir.glob("*.json"):
        item = read_json_any(path)
        doc_id = str(item.get("doc_id", "")).strip()
        if doc_id:
            docs[doc_id] = item
    if not docs:
        raise ValueError(f"No processed docs found under: {processed_dir}")
    return docs


def build_doc_catalog(doc_meta_path: Path, processed_docs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    doc_items = normalize_doc_items(read_json_any(doc_meta_path))
    catalog: Dict[str, Dict[str, Any]] = {}
    for item in doc_items:
        doc_id = str(item.get("doc_id", "")).strip()
        if not doc_id:
            continue
        processed = processed_docs.get(doc_id, {})
        catalog[doc_id] = {
            "doc_id": doc_id,
            "title": str(item.get("title", doc_id)),
            "path": str(item.get("path", "")),
            "text": processed.get("text", ""),
            "pages": processed.get("pages", []),
        }
    return catalog


def split_into_chunks(doc_id: str, title: str, text: str, chunk_size: int = 1200, overlap: int = 200) -> List[Chunk]:
    text = clean_text(text)
    if not text:
        return []
    chunks: List[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk_text = text[start:end]
        chunks.append(
            Chunk(
                doc_id=doc_id,
                title=title,
                chunk_id=f"{doc_id}_chunk_{index:03d}",
                text=chunk_text,
            )
        )
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
        index += 1
    return chunks


def retrieve_docs(question: Dict[str, Any], catalog: Dict[str, Dict[str, Any]], top_k: int) -> List[str]:
    explicit_doc_ids = question.get("doc_ids") or []
    if explicit_doc_ids:
        return [doc_id for doc_id in explicit_doc_ids if doc_id in catalog]

    query_tokens = tokenize(build_question_text(question))
    scored: List[Tuple[int, str]] = []
    for doc_id, doc in catalog.items():
        searchable = f"{doc['title']}\n{doc['text'][:4000]}"
        score = score_text(query_tokens, searchable)
        if score > 0:
            scored.append((score, doc_id))
    scored.sort(reverse=True)
    return [doc_id for _, doc_id in scored[:top_k]]


def retrieve_chunks(question: Dict[str, Any], selected_doc_ids: List[str], catalog: Dict[str, Dict[str, Any]], top_k: int) -> List[Chunk]:
    return retrieve_evidence(question, selected_doc_ids, catalog, top_k).all_chunks


def retrieve_evidence(question: Dict[str, Any], selected_doc_ids: List[str], catalog: Dict[str, Dict[str, Any]], top_k: int) -> EvidencePack:
    def scored_for_query(query_text: str) -> List[Chunk]:
        query_tokens = tokenize(query_text)
        scored: List[Chunk] = []
        for doc_id in selected_doc_ids:
            doc = catalog[doc_id]
            chunks = split_into_chunks(doc_id, doc["title"], doc["text"])
            for chunk in chunks:
                chunk.score = score_text(query_tokens, chunk.text)
                if chunk.score > 0:
                    scored.append(chunk)
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored

    question_text = build_question_text(question)
    selected: Dict[str, Chunk] = {}
    option_chunks: Dict[str, List[Chunk]] = {}

    # Keep a few globally relevant chunks, then add option-specific chunks so
    # multi-choice questions do not lose evidence for minority options.
    global_chunks = scored_for_query(question_text)[:4]
    for chunk in global_chunks:
        selected[chunk.chunk_id] = chunk

    options = question.get("options", {})
    if isinstance(options, dict):
        stem = str(question.get("question", "")).strip()
        for letter in ("A", "B", "C", "D"):
            if letter not in options:
                continue
            option_query = "\n".join(
                [
                    stem,
                    f"{letter}. {options[letter]}",
                    f"type={question.get('type', '')}",
                    f"domain={question.get('domain', '')}",
                ]
            )
            chunks_for_option = scored_for_query(option_query)[:3]
            option_chunks[letter] = chunks_for_option
            for chunk in chunks_for_option:
                if chunk.chunk_id not in selected:
                    selected[chunk.chunk_id] = chunk
                else:
                    selected[chunk.chunk_id].score = max(selected[chunk.chunk_id].score, chunk.score)

    scored_chunks = sorted(selected.values(), key=lambda item: item.score, reverse=True)
    return EvidencePack(
        global_chunks=global_chunks,
        option_chunks=option_chunks,
        all_chunks=scored_chunks[:top_k],
    )


def build_context(chunks: List[Chunk], max_chars: int) -> str:
    parts: List[str] = []
    current_length = 0
    for chunk in chunks:
        block = (
            f"[doc_id={chunk.doc_id}] [title={chunk.title}] [chunk_id={chunk.chunk_id}]\n"
            f"{chunk.text}"
        )
        if current_length + len(block) > max_chars and parts:
            break
        parts.append(block)
        current_length += len(block)
    return "\n\n".join(parts)


def format_chunk(chunk: Chunk) -> str:
    return (
        f"[doc_id={chunk.doc_id}] [title={chunk.title}] [chunk_id={chunk.chunk_id}] [score={chunk.score}]\n"
        f"{chunk.text}"
    )


def build_grouped_context(evidence: EvidencePack, max_chars: int) -> str:
    sections: List[Tuple[str, List[Chunk]]] = [("GLOBAL_EVIDENCE", evidence.global_chunks)]
    for letter in ("A", "B", "C", "D"):
        chunks = evidence.option_chunks.get(letter, [])
        if chunks:
            sections.append((f"OPTION_{letter}_EVIDENCE", chunks))

    parts: List[str] = []
    current_length = 0
    for title, chunks in sections:
        body = "\n\n".join(format_chunk(chunk) for chunk in chunks)
        if not body:
            continue
        block = f"## {title}\n{body}"
        if current_length + len(block) > max_chars and parts:
            continue
        parts.append(block)
        current_length += len(block)

    if parts:
        return "\n\n".join(parts)
    return build_context(evidence.all_chunks, max_chars)


def get_domain_guidance(question: Dict[str, Any]) -> str:
    domain = str(question.get("domain", "")).strip()
    return DOMAIN_GUIDANCE.get(domain, "Use only the supplied evidence and compare each option literally.")


def ask_llm(client: OpenAI, model: str, question: Dict[str, Any], context: str) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
    question_text = build_question_text(question)
    answer_format = str(question.get("answer_format", "mcq")).strip().lower()

    system_prompt = (
        "You are a careful financial long-document QA agent. "
        "Only answer from the provided evidence. "
        "Judge every option independently before giving the final answer. "
        "Return strict JSON with keys: judgments, answer, reason, evidence_doc_ids."
    )
    user_prompt = f"""
Question:
{question_text}

Domain guidance:
{get_domain_guidance(question)}

Evidence:
{context}

Rules:
1. judgments must be an object with A/B/C/D keys when options exist.
2. Each judgment value must contain: verdict, evidence, reasoning.
3. verdict must be a JSON boolean true or false, not a string.
4. verdict is true only when the option is directly supported by evidence or a calculation from evidence.
5. If a statement is partially correct but one condition is wrong, verdict must be false.
6. If an option says increase/decrease, higher/lower, earlier/later, or ordinary/special, verify the direction exactly.
7. If an option contradicts the evidence or lacks enough evidence, verdict must be false.
8. Use OPTION_A_EVIDENCE for A, OPTION_B_EVIDENCE for B, OPTION_C_EVIDENCE for C, and OPTION_D_EVIDENCE for D when available.
9. If answer_format is mcq or tf, answer with one uppercase letter only.
10. If answer_format is multi, answer with sorted uppercase letters only, such as AC or BCD.
11. The final answer must exactly equal the letters whose verdict is true.
12. Do not output any text outside JSON.
""".strip()

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    raw_answer = str(parsed.get("answer", "")).strip()
    judgment_answer = normalize_answer_from_judgments(parsed.get("judgments"), answer_format)
    answer = judgment_answer or normalize_answer(raw_answer, answer_format)
    usage = {
        "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
    }
    return answer, usage, parsed


def write_answer_csv(path: Path, rows: List[Dict[str, Any]], summary: Dict[str, int]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        writer.writerow(["summary", "", summary["prompt_tokens"], summary["completion_tokens"], summary["total_tokens"]])
        for row in rows:
            writer.writerow(
                [
                    row["qid"],
                    row["answer"],
                    row["prompt_tokens"],
                    row["completion_tokens"],
                    row["total_tokens"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve AFAC2026 questions with a minimal starter.")
    parser.add_argument("--questions", required=True, help="Path to question json/jsonl")
    parser.add_argument("--doc-meta", required=True, help="Path to documents.json")
    parser.add_argument("--processed-dir", required=True, help="Directory created by preprocess.py")
    parser.add_argument("--output", required=True, help="Path to answer.csv")
    parser.add_argument("--evidence-output", default="", help="Optional evidence json output path")
    parser.add_argument("--limit", type=int, default=0, help="Only run first N questions")
    args = parser.parse_args()

    settings = load_settings()
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    processed_docs = load_processed_docs(Path(args.processed_dir))
    catalog = build_doc_catalog(Path(args.doc_meta), processed_docs)
    questions = normalize_question_items(read_json_any(args.questions))
    if args.limit > 0:
        questions = questions[: args.limit]

    rows: List[Dict[str, Any]] = []
    evidence_rows: List[Dict[str, Any]] = []
    summary = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for question in tqdm(questions, desc="solve"):
        qid = str(question.get("qid", "")).strip()
        selected_doc_ids = retrieve_docs(question, catalog, settings.top_k_docs)
        evidence = retrieve_evidence(question, selected_doc_ids, catalog, settings.top_k_chunks)
        context = build_grouped_context(evidence, settings.max_context_chars)
        answer, usage, parsed = ask_llm(client, settings.model, question, context)

        summary["prompt_tokens"] += usage["prompt_tokens"]
        summary["completion_tokens"] += usage["completion_tokens"]
        summary["total_tokens"] += usage["total_tokens"]

        rows.append(
            {
                "qid": qid,
                "answer": answer,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            }
        )
        evidence_rows.append(
            {
                "qid": qid,
                "answer": answer,
                "selected_doc_ids": selected_doc_ids,
                "selected_chunks": [
                    {
                        "doc_id": chunk.doc_id,
                        "title": chunk.title,
                        "chunk_id": chunk.chunk_id,
                        "score": chunk.score,
                        "text_preview": chunk.text[:300],
                    }
                    for chunk in evidence.all_chunks
                ],
                "option_chunks": {
                    letter: [
                        {
                            "doc_id": chunk.doc_id,
                            "title": chunk.title,
                            "chunk_id": chunk.chunk_id,
                            "score": chunk.score,
                            "text_preview": chunk.text[:300],
                        }
                        for chunk in chunks
                    ]
                    for letter, chunks in evidence.option_chunks.items()
                },
                "model_output": parsed,
            }
        )

    output_path = Path(args.output)
    write_answer_csv(output_path, rows, summary)
    print(f"Saved answer csv to: {output_path}")

    if args.evidence_output:
        evidence_path = Path(args.evidence_output)
        ensure_dir(evidence_path.parent)
        safe_json_dump(evidence_path, evidence_rows)
        print(f"Saved evidence json to: {evidence_path}")


if __name__ == "__main__":
    main()
