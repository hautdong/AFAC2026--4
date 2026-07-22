from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .common import ensure_dir, safe_json_dump


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".html", ".htm"}


def read_question_file(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    raw = json.loads(text)
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        for key in ("questions", "data", "items"):
            if isinstance(raw.get(key), list):
                return [item for item in raw[key] if isinstance(item, dict)]
    raise ValueError(f"Unsupported B question structure: {path}")


def infer_answer_format(question: Dict[str, Any]) -> str:
    type_name = str(question.get("type", ""))
    if type_name == "单选题":
        return "mcq"
    if type_name == "多选题":
        return "multi"
    if type_name == "判断题":
        return "tf"
    return "open"


def load_questions(question_dir: Path) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    for path in sorted(question_dir.glob("*")):
        if path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        for raw in read_question_file(path):
            item = dict(raw)
            item["answer_format"] = infer_answer_format(item)
            questions.append(item)
    return questions


def build_documents(raw_root: Path) -> List[Dict[str, str]]:
    documents: List[Dict[str, str]] = []
    seen: Dict[str, Path] = {}
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        doc_id = path.stem
        if doc_id in seen:
            raise ValueError(f"Duplicate doc_id '{doc_id}': {seen[doc_id]} and {path}")
        seen[doc_id] = path
        relative = path.relative_to(raw_root)
        documents.append(
            {
                "doc_id": doc_id,
                "path": relative.as_posix(),
                "title": doc_id,
                "domain": relative.parts[0] if len(relative.parts) > 1 else "",
            }
        )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AFAC2026 B-board questions and full document catalog.")
    parser.add_argument("--question-dir", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    metadata_dir = ensure_dir(out_dir / "metadata")
    question_output_dir = ensure_dir(out_dir / "questions")
    questions = load_questions(Path(args.question_dir))
    documents = build_documents(Path(args.raw_root))
    safe_json_dump(question_output_dir / "public_b.json", questions)
    safe_json_dump(metadata_dir / "documents.json", documents)
    print(f"Prepared B questions: {len(questions)}")
    print(f"Prepared full document catalog: {len(documents)}")


if __name__ == "__main__":
    main()
