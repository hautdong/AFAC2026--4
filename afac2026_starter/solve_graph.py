from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from .common import (
    build_question_text,
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
from .solve import DOMAIN_GUIDANCE, Chunk, format_chunk, retrieve_docs

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


def load_graph(path: Path) -> Dict[str, Any]:
    graph = read_json_any(path)
    if graph.get("schema") != "afac_hierarchical_graph_v1":
        raise ValueError(f"Unsupported graph index: {path}")
    return graph


def build_doc_catalog(doc_meta_path: Path, graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    doc_items = normalize_doc_items(read_json_any(doc_meta_path))
    docs = graph["docs"]
    catalog: Dict[str, Dict[str, Any]] = {}
    for item in doc_items:
        doc_id = str(item.get("doc_id", "")).strip()
        if not doc_id or doc_id not in docs:
            continue
        chunk_ids = graph["doc_chunks"].get(doc_id, [])
        preview = "\n".join(graph["chunks"][chunk_id]["text"] for chunk_id in chunk_ids[:5])
        catalog[doc_id] = {
            "doc_id": doc_id,
            "title": str(item.get("title", doc_id)),
            "path": str(item.get("path", "")),
            "text": preview,
        }
    return catalog


def graph_chunk_to_chunk(graph_chunk: Dict[str, Any], score: int) -> Chunk:
    return Chunk(
        doc_id=str(graph_chunk["doc_id"]),
        title=str(graph_chunk["title"]),
        chunk_id=str(graph_chunk["chunk_id"]),
        text=str(graph_chunk["text"]),
        score=score,
    )


def rank_graph_chunks(
    graph: Dict[str, Any],
    doc_ids: List[str],
    query_text: str,
    limit: int,
) -> List[Chunk]:
    query_tokens = tokenize(query_text)
    query_terms = set(term for term in graph.get("term_chunks", {}) if term and term in query_text)
    candidates: Dict[str, int] = {}

    for doc_id in doc_ids:
        for chunk_id in graph["doc_chunks"].get(doc_id, []):
            candidates.setdefault(chunk_id, 0)

    # Token and term indexes are deterministic lexical retrieval, not embedding retrieval.
    for token in query_tokens:
        for chunk_id in graph.get("token_chunks", {}).get(token, [])[:80]:
            graph_chunk = graph["chunks"][chunk_id]
            if graph_chunk["doc_id"] in doc_ids:
                candidates[chunk_id] = candidates.get(chunk_id, 0) + 2
    for term in query_terms:
        for chunk_id in graph.get("term_chunks", {}).get(term, [])[:80]:
            graph_chunk = graph["chunks"][chunk_id]
            if graph_chunk["doc_id"] in doc_ids:
                candidates[chunk_id] = candidates.get(chunk_id, 0) + 8

    ranked: List[Chunk] = []
    for chunk_id in candidates:
        graph_chunk = graph["chunks"][chunk_id]
        score = candidates[chunk_id] + score_text(query_tokens, graph_chunk["text"])
        if score > 0:
            ranked.append(graph_chunk_to_chunk(graph_chunk, score))
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:limit]


def build_graph_context(question: Dict[str, Any], graph: Dict[str, Any], doc_ids: List[str], max_chars: int) -> Tuple[str, Dict[str, List[Chunk]]]:
    stem = str(question.get("question", "")).strip()
    options = question.get("options", {})
    grouped: Dict[str, List[Chunk]] = {
        "GLOBAL": rank_graph_chunks(graph, doc_ids, build_question_text(question), 5)
    }
    if isinstance(options, dict):
        for letter in ("A", "B", "C", "D"):
            if letter not in options:
                continue
            query = "\n".join(
                [
                    stem,
                    f"{letter}. {options[letter]}",
                    f"type={question.get('type', '')}",
                    f"domain={question.get('domain', '')}",
                ]
            )
            grouped[letter] = rank_graph_chunks(graph, doc_ids, query, 4)

    parts: List[str] = []
    used = 0
    for section in ("GLOBAL", "A", "B", "C", "D"):
        chunks = grouped.get(section, [])
        if not chunks:
            continue
        block = "## {0}\n{1}".format(section, "\n\n".join(format_chunk(chunk) for chunk in chunks))
        if used + len(block) > max_chars and parts:
            continue
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts), grouped


def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else {}
    return {}


def call_json(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        response_format={"type": "json_object"},
    )
    parsed = safe_json_loads(response.choices[0].message.content or "{}")
    usage = {
        "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
    }
    return parsed, usage


def research_evidence(client: OpenAI, model: str, question: Dict[str, Any], graph_context: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    question_text = build_question_text(question)
    domain = str(question.get("domain", "")).strip()
    guidance = DOMAIN_GUIDANCE.get(domain, "Use only supplied evidence.")
    messages = [
        {
            "role": "system",
            "content": (
                "You are the Researcher agent in a graph RAG system. "
                "Extract concise evidence for each option from the grouped graph context. "
                "Do not decide the final answer."
            ),
        },
        {
            "role": "user",
            "content": f"""
Question:
{question_text}

Domain guidance:
{guidance}

Grouped graph context:
{graph_context}

Return strict JSON:
{{
  "option_evidence": {{
    "A": ["short quote or calculation evidence"],
    "B": ["short quote or calculation evidence"],
    "C": ["short quote or calculation evidence"],
    "D": ["short quote or calculation evidence"]
  }},
  "missing_or_conflicting": {{
    "A": "brief note",
    "B": "brief note",
    "C": "brief note",
    "D": "brief note"
  }}
}}
""".strip(),
        },
    ]
    return call_json(client, model, messages)


def adjudicate(
    client: OpenAI,
    model: str,
    question: Dict[str, Any],
    research: Dict[str, Any],
) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
    question_text = build_question_text(question)
    answer_format = str(question.get("answer_format", "mcq")).strip().lower()
    messages = [
        {
            "role": "system",
            "content": (
                "You are the Auditor and Adjudicator agents in a graph RAG system. "
                "Audit the research evidence, judge every option independently, "
                "then output the final answer."
            ),
        },
        {
            "role": "user",
            "content": f"""
Question:
{question_text}

Researcher evidence:
{json.dumps(research, ensure_ascii=False)}

Rules:
1. judgments must contain A/B/C/D keys when options exist.
2. Each judgment must contain: verdict, evidence, reasoning.
3. verdict must be a JSON boolean true or false.
4. If evidence is missing, conflicting, only approximate, or only partially supports an option, verdict must be false.
5. Verify direction words exactly: increase/decrease, higher/lower, earlier/later, ordinary/special, must/may.
6. If answer_format is mcq or tf, answer with one uppercase letter only.
7. If answer_format is multi, answer with sorted uppercase letters only.
8. answer must exactly equal the true verdict letters.
9. Return strict JSON with keys: judgments, answer, reason, evidence_doc_ids.
""".strip(),
        },
    ]
    parsed, usage = call_json(client, model, messages)
    judgment_answer = normalize_answer_from_judgments(parsed.get("judgments"), answer_format)
    answer = judgment_answer or normalize_answer(str(parsed.get("answer", "")), answer_format)
    return answer, usage, parsed


def write_answer_csv(path: Path, rows: List[Dict[str, Any]], summary: Dict[str, int]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        writer.writerow(["summary", "", summary["prompt_tokens"], summary["completion_tokens"], summary["total_tokens"]])
        for row in rows:
            writer.writerow([row["qid"], row["answer"], row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve AFAC2026 with hierarchical graph RAG and multi-agent reasoning.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--doc-meta", required=True)
    parser.add_argument("--graph-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-output", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    settings = load_settings()
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    graph = load_graph(Path(args.graph_index))
    catalog = build_doc_catalog(Path(args.doc_meta), graph)
    questions = normalize_question_items(read_json_any(args.questions))
    if args.limit > 0:
        questions = questions[: args.limit]

    rows: List[Dict[str, Any]] = []
    evidence_rows: List[Dict[str, Any]] = []
    summary = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for question in tqdm(questions, desc="solve_graph"):
        qid = str(question.get("qid", "")).strip()
        doc_ids = retrieve_docs(question, catalog, settings.top_k_docs)
        graph_context, grouped_chunks = build_graph_context(question, graph, doc_ids, settings.max_context_chars)
        research, research_usage = research_evidence(client, settings.model, question, graph_context)
        answer, adjudication_usage, adjudication = adjudicate(client, settings.model, question, research)
        usage = {
            "prompt_tokens": research_usage["prompt_tokens"] + adjudication_usage["prompt_tokens"],
            "completion_tokens": research_usage["completion_tokens"] + adjudication_usage["completion_tokens"],
            "total_tokens": research_usage["total_tokens"] + adjudication_usage["total_tokens"],
        }
        for key in summary:
            summary[key] += usage[key]

        rows.append({"qid": qid, "answer": answer, **usage})
        evidence_rows.append(
            {
                "qid": qid,
                "answer": answer,
                "selected_doc_ids": doc_ids,
                "graph_chunks": {
                    section: [
                        {
                            "doc_id": chunk.doc_id,
                            "chunk_id": chunk.chunk_id,
                            "score": chunk.score,
                            "text_preview": chunk.text[:300],
                        }
                        for chunk in chunks
                    ]
                    for section, chunks in grouped_chunks.items()
                },
                "researcher_output": research,
                "adjudicator_output": adjudication,
            }
        )

    write_answer_csv(Path(args.output), rows, summary)
    print(f"Saved answer csv to: {args.output}")
    if args.evidence_output:
        evidence_path = Path(args.evidence_output)
        ensure_dir(evidence_path.parent)
        safe_json_dump(evidence_path, evidence_rows)
        print(f"Saved evidence json to: {evidence_path}")


if __name__ == "__main__":
    main()
