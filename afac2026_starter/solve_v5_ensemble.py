from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

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
    add_usage,
    answer_protocol,
    build_document_indexes,
    call_json,
    domain_instructions,
    parse_answer,
    select_evidence,
    valid_citation_ratio,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:  # type: ignore
        return iterable


def load_answer_rows(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    rows: Dict[str, Dict[str, Any]] = {}
    summary = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
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
            rows[row["qid"]] = row
            for key in summary:
                summary[key] += row[key]
    return rows, summary


def disagreement_kind(v2_answer: str, graph_answer: str, v4_answer: str) -> str:
    if v2_answer == graph_answer == v4_answer:
        return "all_agree"
    if v4_answer == graph_answer:
        return "v4_eq_graph"
    if v4_answer == v2_answer:
        return "v4_eq_v2"
    if v2_answer == graph_answer:
        return "v2_eq_graph"
    return "all_different"


def should_arbitrate(kind: str) -> bool:
    return kind in {"v2_eq_graph", "all_different"}


def load_trace_map(path: Path) -> Dict[str, Dict[str, Any]]:
    raw = read_json_any(path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected trace list: {path}")
    return {
        str(item.get("qid", "")): item
        for item in raw
        if isinstance(item, dict) and str(item.get("qid", ""))
    }


def collect_graph_chunk_ids(trace: Dict[str, Any], limit: int = 8) -> List[Tuple[str, str]]:
    grouped = trace.get("graph_chunks")
    if not isinstance(grouped, dict):
        return []
    collected: List[Tuple[str, str]] = []
    seen: Set[str] = set()

    # Option evidence is more useful for arbitration than another global top-k list.
    for round_index in range(2):
        for section in ("A", "B", "C", "D", "GLOBAL"):
            chunks = grouped.get(section)
            if not isinstance(chunks, list) or round_index >= len(chunks):
                continue
            item = chunks[round_index]
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id", ""))
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            collected.append((section, chunk_id))
            if len(collected) >= limit:
                return collected
    return collected


def format_graph_supplement(
    graph: Dict[str, Any],
    trace: Dict[str, Any],
    max_chars: int = 6500,
) -> Tuple[str, Set[str]]:
    graph_chunks = graph.get("chunks")
    if not isinstance(graph_chunks, dict):
        return "", set()
    parts: List[str] = []
    valid_ids: Set[str] = set()
    used = 0
    for section, chunk_id in collect_graph_chunk_ids(trace):
        item = graph_chunks.get(chunk_id)
        if not isinstance(item, dict):
            continue
        block = (
            f"[graph_chunk_id={chunk_id}] [doc_id={item.get('doc_id', '')}] "
            f"[retrieved_for={section}]\n{item.get('text', '')}\n"
        )
        if used + len(block) > max_chars:
            continue
        parts.append(block)
        valid_ids.add(chunk_id)
        used += len(block)
    return "\n".join(parts), valid_ids


def candidate_labels(answers: Sequence[str]) -> Dict[str, str]:
    unique = sorted(set(answers))
    return {
        f"CANDIDATE_{index + 1}": answer
        for index, answer in enumerate(unique)
    }


def arbitration_messages(
    question: Dict[str, Any],
    page_context: str,
    graph_context: str,
    candidates: Dict[str, str],
) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the final evidence arbitrator for financial long-document QA. Candidate answers are "
                "untrusted hypotheses. Decide independently from the supplied source excerpts. Preserve document "
                "identity, years, units, inequality direction, legal conditions, and exceptions. Return strict JSON only."
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

Untrusted candidate answers:
{json.dumps(candidates, ensure_ascii=False)}

Page-aware evidence:
{page_context}

Supplemental graph-retrieved evidence:
{graph_context or '(none)'}

Instructions:
1. Ignore candidate popularity. Two candidates may share the same retrieval mistake.
2. Test every option against original evidence. For comparisons, show both values and evaluate every >, <, or = sign.
3. For claims using both/all, verify every named document. Missing evidence is not a contradiction.
4. For financial reports distinguish year, unit, proposed policy, and completed payment. For rules retain actor, trigger, deadline, modal verb, and exception.
5. Citations must exactly match supplied chunk_id or graph_chunk_id values.

Return strict JSON:
{{
  "judgments": {{
    "A": {{"verdict": true, "confidence": 0.0, "citations": ["chunk_id"], "reasoning": "brief"}},
    "B": {{"verdict": false, "confidence": 0.0, "citations": ["chunk_id"], "reasoning": "brief"}},
    "C": {{"verdict": false, "confidence": 0.0, "citations": ["chunk_id"], "reasoning": "brief"}},
    "D": {{"verdict": false, "confidence": 0.0, "citations": ["chunk_id"], "reasoning": "brief"}}
  }},
  "proposition_verdict": null,
  "answer": "A",
  "decision_confidence": 0.0,
  "candidate_assessment": {{"CANDIDATE_1": "supported or rejected with reason"}},
  "reason": "brief final reason"
}}
""".strip(),
        },
    ]


def decision_confidence(parsed: Dict[str, Any]) -> float:
    try:
        return float(parsed.get("decision_confidence", 0))
    except (TypeError, ValueError):
        return 0.0


def policy_accepts(
    policy: str,
    evidence_accepted: bool,
    arbitrated_answer: str,
    graph_answer: str,
    v4_answer: str,
) -> bool:
    if not evidence_accepted:
        return False
    if policy == "graph-confirmed":
        return arbitrated_answer == graph_answer and graph_answer != v4_answer
    return True


def recompute_summary(rows: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    return {
        key: sum(int(row.get(key, 0)) for row in rows.values())
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def ordered_rows(
    questions: Sequence[Dict[str, Any]],
    rows: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [rows[str(question.get("qid", ""))] for question in questions]


def main() -> None:
    parser = argparse.ArgumentParser(description="AFAC2026 v5 graph-evidence arbitration.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--doc-meta", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--graph-index", required=True)
    parser.add_argument("--graph-evidence", required=True)
    parser.add_argument("--v2-answer", required=True)
    parser.add_argument("--graph-answer", required=True)
    parser.add_argument("--v4-answer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-output", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--accept-policy",
        choices=("evidence", "graph-confirmed"),
        default="graph-confirmed",
    )
    args = parser.parse_args()

    settings = load_settings()
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    questions = normalize_question_items(read_json_any(args.questions))
    indexes = build_document_indexes(Path(args.doc_meta), Path(args.processed_dir))
    graph = read_json_any(args.graph_index)
    graph_traces = load_trace_map(Path(args.graph_evidence))
    v2_rows, _ = load_answer_rows(Path(args.v2_answer))
    graph_rows, _ = load_answer_rows(Path(args.graph_answer))
    v4_rows, _ = load_answer_rows(Path(args.v4_answer))

    risky_questions: List[Dict[str, Any]] = []
    kinds: Dict[str, str] = {}
    for question in questions:
        qid = str(question.get("qid", ""))
        kind = disagreement_kind(
            v2_rows[qid]["answer"],
            graph_rows[qid]["answer"],
            v4_rows[qid]["answer"],
        )
        kinds[qid] = kind
        if should_arbitrate(kind):
            risky_questions.append(question)
    if args.limit > 0:
        risky_questions = risky_questions[:args.limit]

    output_path = Path(args.output)
    evidence_path = Path(args.evidence_output) if args.evidence_output else None
    traces: List[Dict[str, Any]] = []
    completed: Set[str] = set()
    if args.resume and evidence_path and evidence_path.exists():
        restored = read_json_any(evidence_path)
        if isinstance(restored, list):
            traces = [item for item in restored if isinstance(item, dict)]
            for trace in traces:
                qid = str(trace.get("qid", ""))
                if not qid or qid not in v4_rows:
                    continue
                completed.add(qid)
                evidence_accepted = bool(trace.get("evidence_accepted", trace.get("accepted", False)))
                accepted = policy_accepts(
                    args.accept_policy,
                    evidence_accepted,
                    str(trace.get("arbitrated_answer", "")),
                    str(trace.get("graph_answer", "")),
                    str(trace.get("v4_answer", v4_rows[qid]["answer"])),
                )
                final_answer = (
                    str(trace.get("arbitrated_answer", ""))
                    if accepted
                    else str(trace.get("v4_answer", v4_rows[qid]["answer"]))
                )
                trace["evidence_accepted"] = evidence_accepted
                trace["accepted"] = accepted
                trace["accept_policy"] = args.accept_policy
                trace["final_answer"] = final_answer
                v4_rows[qid]["answer"] = final_answer
                usage = trace.get("usage") if isinstance(trace.get("usage"), dict) else {}
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    v4_rows[qid][key] += int(usage.get(key, 0) or 0)

    pending = [question for question in risky_questions if str(question.get("qid", "")) not in completed]
    for question in tqdm(pending, desc="solve_v5_ensemble"):
        qid = str(question.get("qid", ""))
        answer_format = str(question.get("answer_format", "multi")).lower()
        pack = select_evidence(question, indexes, 18000)
        graph_context, graph_chunk_ids = format_graph_supplement(graph, graph_traces.get(qid, {}))
        candidates = candidate_labels(
            [v2_rows[qid]["answer"], graph_rows[qid]["answer"], v4_rows[qid]["answer"]]
        )
        parsed, usage = call_json(
            client,
            settings.model,
            arbitration_messages(question, pack.context, graph_context, candidates),
        )
        arbitrated_answer, issues = parse_answer(parsed, answer_format)
        valid_ids = {chunk.chunk_id for chunk in pack.chunks} | graph_chunk_ids
        citation_ratio = valid_citation_ratio(parsed, valid_ids)
        confidence = decision_confidence(parsed)
        evidence_accepted = not issues and confidence >= 0.85 and citation_ratio >= 1.0
        previous_answer = v4_rows[qid]["answer"]
        accepted = policy_accepts(
            args.accept_policy,
            evidence_accepted,
            arbitrated_answer,
            graph_rows[qid]["answer"],
            previous_answer,
        )
        final_answer = arbitrated_answer if accepted else previous_answer
        v4_rows[qid]["answer"] = final_answer
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            v4_rows[qid][key] += usage[key]

        traces.append(
            {
                "qid": qid,
                "kind": kinds[qid],
                "v2_answer": v2_rows[qid]["answer"],
                "graph_answer": graph_rows[qid]["answer"],
                "v4_answer": previous_answer,
                "arbitrated_answer": arbitrated_answer,
                "final_answer": final_answer,
                "evidence_accepted": evidence_accepted,
                "accepted": accepted,
                "accept_policy": args.accept_policy,
                "decision_confidence": confidence,
                "citation_ratio": citation_ratio,
                "issues": issues,
                "candidates": candidates,
                "page_retrieval": pack.diagnostics,
                "graph_chunk_ids": sorted(graph_chunk_ids),
                "arbitrator_output": parsed,
                "usage": usage,
            }
        )
        summary = recompute_summary(v4_rows)
        write_answer_csv(output_path, ordered_rows(questions, v4_rows), summary)
        if evidence_path:
            ensure_dir(evidence_path.parent)
            safe_json_dump(evidence_path, traces)

    summary = recompute_summary(v4_rows)
    write_answer_csv(output_path, ordered_rows(questions, v4_rows), summary)
    if evidence_path:
        ensure_dir(evidence_path.parent)
        safe_json_dump(evidence_path, traces)
    print(f"Arbitrated questions: {len(traces)}/{len(risky_questions)}")
    print(f"Accepted arbitration decisions: {sum(1 for trace in traces if trace['accepted'])}")
    print(f"Changed from v4: {sum(1 for trace in traces if trace['final_answer'] != trace['v4_answer'])}")
    print(f"Total tokens including v4: {summary['total_tokens']}")
    print(f"Saved answer csv to: {output_path}")


if __name__ == "__main__":
    main()
