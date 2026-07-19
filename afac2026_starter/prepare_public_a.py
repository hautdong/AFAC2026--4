from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from .common import ensure_dir, normalize_question_items, read_json_any, safe_json_dump


QUESTION_FILES = [
    "financial_contracts_questions.json",
    "financial_reports_questions.json",
    "insurance_questions.json",
    "regulatory_questions.json",
    "research_questions.json",
]


def load_all_questions(question_dir: Path) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for name in QUESTION_FILES:
        path = question_dir / name
        items = normalize_question_items(read_json_any(path))
        merged.extend(items)
    return merged


def build_file_index(raw_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in raw_root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".pdf", ".txt", ".html", ".htm"}:
            continue
        index[path.stem] = path
    return index


def build_documents(questions: List[Dict[str, Any]], file_index: Dict[str, Path], raw_root: Path) -> List[Dict[str, str]]:
    doc_ids = []
    seen = set()
    for question in questions:
        for doc_id in question.get("doc_ids", []):
            if doc_id not in seen:
                seen.add(doc_id)
                doc_ids.append(doc_id)

    documents: List[Dict[str, str]] = []
    for doc_id in doc_ids:
        if doc_id not in file_index:
            raise FileNotFoundError("No source file found for doc_id={0}".format(doc_id))
        source_path = file_index[doc_id]
        rel_path = source_path.relative_to(raw_root).as_posix()
        documents.append(
            {
                "doc_id": doc_id,
                "path": rel_path,
                "title": doc_id,
            }
        )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official public_dataset_a for AFAC2026 starter.")
    parser.add_argument("--dataset-root", required=True, help="Path to public_dataset_a")
    parser.add_argument("--out-dir", required=True, help="Directory to write generated metadata and merged questions")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    raw_root = dataset_root / "raw"
    question_dir = dataset_root / "questions" / "group_a"
    out_dir = ensure_dir(args.out_dir)
    metadata_dir = ensure_dir(out_dir / "metadata")
    output_question_dir = ensure_dir(out_dir / "questions")

    questions = load_all_questions(question_dir)
    file_index = build_file_index(raw_root)
    documents = build_documents(questions, file_index, raw_root)

    safe_json_dump(metadata_dir / "documents.json", documents)
    safe_json_dump(output_question_dir / "public_a.json", questions)

    print("Prepared questions: {0}".format(len(questions)))
    print("Prepared documents: {0}".format(len(documents)))
    print("Metadata saved to: {0}".format(metadata_dir / "documents.json"))
    print("Questions saved to: {0}".format(output_question_dir / "public_a.json"))


if __name__ == "__main__":
    main()
