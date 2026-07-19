from __future__ import annotations

import argparse
import re
from pathlib import Path

import fitz

from .common import clean_text, ensure_dir, normalize_doc_items, read_json_any, safe_json_dump

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


def resolve_doc_path(raw_path: str, docs_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return docs_root / path


def extract_pdf(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)
    pages = []
    full_text_parts = []
    for page_index, page in enumerate(doc, start=1):
        text = clean_text(page.get_text("text"))
        pages.append({"page": page_index, "text": text})
        full_text_parts.append(f"[PAGE {page_index}]\n{text}")
    return {"pages": pages, "text": "\n\n".join(full_text_parts)}


def extract_text_file(text_path: Path) -> dict:
    raw = text_path.read_text(encoding="utf-8")
    text = clean_text(raw)
    return {"pages": [{"page": 1, "text": text}], "text": text}


def extract_html(html_path: Path) -> dict:
    raw = html_path.read_text(encoding="utf-8")
    text = re.sub(r"<script.*?</script>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = clean_text(text)
    return {"pages": [{"page": 1, "text": text}], "text": text}


def extract_document(source_path: Path) -> dict:
    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(source_path)
    if suffix == ".txt":
        return extract_text_file(source_path)
    if suffix in {".html", ".htm"}:
        return extract_html(source_path)
    raise ValueError("Unsupported file type: {0}".format(source_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PDF text for AFAC2026 starter.")
    parser.add_argument("--doc-meta", required=True, help="Path to documents.json")
    parser.add_argument("--docs-root", required=True, help="Root folder of PDFs")
    parser.add_argument("--out-dir", required=True, help="Output folder for processed docs")
    args = parser.parse_args()

    docs_root = Path(args.docs_root)
    out_dir = ensure_dir(args.out_dir)
    doc_items = normalize_doc_items(read_json_any(args.doc_meta))

    processed = 0
    for item in tqdm(doc_items, desc="preprocess"):
        doc_id = str(item.get("doc_id", "")).strip()
        raw_path = str(item.get("path", "")).strip()
        title = str(item.get("title", doc_id)).strip()
        if not doc_id or not raw_path:
            continue

        pdf_path = resolve_doc_path(raw_path, docs_root)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found for doc_id={doc_id}: {pdf_path}")

        extracted = extract_document(pdf_path)
        safe_json_dump(
            out_dir / f"{doc_id}.json",
            {
                "doc_id": doc_id,
                "title": title,
                "source_path": str(pdf_path),
                "pages": extracted["pages"],
                "text": extracted["text"],
            },
        )
        processed += 1

    print(f"Processed {processed} documents into: {out_dir}")


if __name__ == "__main__":
    main()
