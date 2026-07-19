from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union


LETTER_PATTERN = re.compile(r"[ABCD]")


@dataclass
class Settings:
    api_key: str
    model: str
    base_url: str
    max_context_chars: int = 24000
    top_k_docs: int = 3
    top_k_chunks: int = 14


def load_settings() -> Settings:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    model = os.getenv("QWEN_MODEL", "qwen-plus").strip()
    base_url = os.getenv(
        "QWEN_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).strip()
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is empty. Please fill it in .env first.")
    return Settings(api_key=api_key, model=model, base_url=base_url)


def ensure_dir(path: Union[str, Path]) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json_any(path: Union[str, Path]) -> Any:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    text = text.strip()
    if not text:
        raise ValueError(f"Empty file: {path}")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def normalize_doc_items(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("documents", "docs", "data", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError("Unsupported document metadata format.")


def normalize_question_items(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("questions", "data", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError("Unsupported question file format.")


def safe_json_dump(path: Union[str, Path], data: Any) -> None:
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clean_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    text = clean_text(text).lower()
    ascii_tokens = re.findall(r"[a-z0-9_./%-]+", text)
    cn_tokens: List[str] = []
    for block in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        cn_tokens.extend(block[i : i + 2] for i in range(len(block) - 1))
        cn_tokens.append(block)
    return ascii_tokens + cn_tokens


def score_text(query_tokens: Iterable[str], text: str) -> int:
    haystack = clean_text(text).lower()
    score = 0
    for token in query_tokens:
        if len(token) <= 1:
            continue
        if token in haystack:
            score += min(3, haystack.count(token))
    return score


def normalize_answer(raw_answer: str, answer_format: str) -> str:
    letters = LETTER_PATTERN.findall(raw_answer.upper())
    if answer_format in {"mcq", "tf"}:
        return letters[0] if letters else "A"
    unique_letters = sorted(set(letters))
    return "".join(unique_letters) if unique_letters else "A"


def normalize_answer_from_judgments(judgments: Any, answer_format: str) -> str:
    if not isinstance(judgments, dict):
        return ""

    true_letters: List[str] = []
    for letter in ("A", "B", "C", "D"):
        item = judgments.get(letter)
        if not isinstance(item, dict):
            continue
        verdict = item.get("verdict")
        if isinstance(verdict, bool):
            is_true = verdict
        else:
            normalized = str(verdict).strip().lower()
            is_true = normalized in {"true", "yes", "correct", "supported", "1"}
        if is_true:
            true_letters.append(letter)

    if not true_letters:
        return ""
    if answer_format in {"mcq", "tf"}:
        return true_letters[0]
    return "".join(true_letters)


def build_question_text(question: Dict[str, Any]) -> str:
    options = question.get("options", {})
    option_lines: List[str] = []
    if isinstance(options, dict):
        for key in sorted(options):
            option_lines.append(f"{key}. {options[key]}")
    return "\n".join(
        [
            str(question.get("question", "")).strip(),
            *option_lines,
            f"answer_format={question.get('answer_format', '')}",
            f"type={question.get('type', '')}",
            f"domain={question.get('domain', '')}",
        ]
    ).strip()
