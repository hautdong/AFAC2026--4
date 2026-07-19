from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .common import clean_text, ensure_dir, read_json_any, safe_json_dump, tokenize


FINANCE_TERMS = [
    "发行人",
    "发行规模",
    "发行金额",
    "募集资金",
    "信用等级",
    "主体评级",
    "债项评级",
    "票面利率",
    "受托管理人",
    "营业收入",
    "净利润",
    "归属于上市公司股东",
    "经营活动产生的现金流量净额",
    "研发投入",
    "现金分红",
    "资产负债率",
    "受益所有人",
    "客户尽职调查",
    "股东会",
    "股东大会",
    "特别决议",
    "普通决议",
    "董事候选人",
    "担保事项",
    "变更募集资金用途",
    "保险责任",
    "身故保险金",
    "现金价值",
    "退保",
    "保费",
    "给付比例",
    "免责",
    "行业趋势",
    "市场规模",
    "同比",
    "环比",
]

ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千万\d]+条")
NUMBER_RE = re.compile(
    r"(?:\d{4}年\d{1,2}月\d{1,2}日|\d{4}年|\d+(?:\.\d+)?%|\d+(?:\.\d+)?(?:亿元|万元|元|年|月|日|个工作日|个自然日|倍|股|份))"
)


def split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 180) -> List[str]:
    text = clean_text(text)
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def extract_terms(text: str) -> List[str]:
    terms = [term for term in FINANCE_TERMS if term in text]
    terms.extend(ARTICLE_RE.findall(text))
    terms.extend(NUMBER_RE.findall(text))
    return sorted(set(terms))


def add_to_index(index: Dict[str, List[str]], keys: Iterable[str], chunk_id: str) -> None:
    for key in keys:
        if not key:
            continue
        bucket = index.setdefault(key, [])
        if chunk_id not in bucket:
            bucket.append(chunk_id)


def build_graph(processed_dir: Path) -> Dict[str, Any]:
    docs: Dict[str, Any] = {}
    chunks: Dict[str, Any] = {}
    doc_chunks: Dict[str, List[str]] = {}
    term_chunks: Dict[str, List[str]] = {}
    token_chunks: Dict[str, List[str]] = {}

    for path in sorted(processed_dir.glob("*.json")):
        doc = read_json_any(path)
        doc_id = str(doc.get("doc_id", "")).strip()
        if not doc_id:
            continue
        title = str(doc.get("title", doc_id))
        text = str(doc.get("text", ""))
        docs[doc_id] = {"doc_id": doc_id, "title": title, "source_path": doc.get("source_path", "")}
        doc_chunks[doc_id] = []

        for index, chunk_text in enumerate(split_into_chunks(text)):
            chunk_id = f"{doc_id}::gchunk::{index:04d}"
            terms = extract_terms(chunk_text)
            chunks[chunk_id] = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "title": title,
                "text": chunk_text,
                "terms": terms,
            }
            doc_chunks[doc_id].append(chunk_id)
            add_to_index(term_chunks, terms, chunk_id)
            add_to_index(token_chunks, tokenize(chunk_text), chunk_id)

    graph = {
        "schema": "afac_hierarchical_graph_v1",
        "docs": docs,
        "chunks": chunks,
        "doc_chunks": doc_chunks,
        "term_chunks": term_chunks,
        "token_chunks": token_chunks,
    }
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight hierarchical graph index.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    graph = build_graph(Path(args.processed_dir))
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    safe_json_dump(output_path, graph)
    print(f"Graph docs: {len(graph['docs'])}")
    print(f"Graph chunks: {len(graph['chunks'])}")
    print(f"Graph saved to: {output_path}")


if __name__ == "__main__":
    main()
