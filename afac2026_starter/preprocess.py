from __future__ import annotations

import argparse
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fitz

from .common import clean_layout_text, ensure_dir, normalize_doc_items, read_json_any, safe_json_dump

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


SECTION_PATTERNS = (
    re.compile(r"^第[一二三四五六七八九十百千\d]+[编章节部分条]"),
    re.compile(r"^[一二三四五六七八九十百]+、"),
    re.compile(r"^[（(][一二三四五六七八九十\d]+[）)]"),
    re.compile(r"^\d+(?:\.\d+)*[、.．]\s*\S+"),
)
CLAUSE_PATTERN = re.compile(r"第[一二三四五六七八九十百千零〇\d]+条")
TABLE_HINTS = ("单位：", "项目", "本期金额", "上期金额", "同比", "年度", "占比", "比例", "金额")
TITLE_HINTS = ("年度报告", "募集说明书", "保险条款", "管理办法", "深度研究", "研究报告", "招股说明书")
TITLE_NOISE = ("请务必阅读", "免责声明", "目录", "证券代码", "第 页", "of ")


def resolve_doc_path(raw_path: str, docs_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return docs_root / path


def extract_sections(text: str) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if len(line) > 100:
            continue
        pattern_index = next((i for i, pattern in enumerate(SECTION_PATTERNS) if pattern.search(line)), None)
        if pattern_index is None:
            continue
        sections.append(
            {
                "line": line_number,
                "level": pattern_index + 1,
                "text": line,
            }
        )
    return sections


def extract_clause_numbers(text: str) -> List[str]:
    return list(dict.fromkeys(CLAUSE_PATTERN.findall(text)))


def block_items(page: fitz.Page, textpage: Optional[fitz.TextPage] = None) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    raw_blocks = page.get_text("blocks", textpage=textpage, sort=True)
    for raw in raw_blocks:
        if len(raw) < 7 or int(raw[6]) != 0:
            continue
        text = clean_layout_text(str(raw[4]))
        if not text:
            continue
        blocks.append(
            {
                "bbox": [round(float(value), 2) for value in raw[:4]],
                "block_no": int(raw[5]),
                "text": text,
            }
        )
    return blocks


def is_table_candidate(text: str) -> bool:
    digit_count = len(re.findall(r"\d", text))
    hint_count = sum(hint in text for hint in TABLE_HINTS)
    has_table_label = re.search(r"(?:^|\n)(?:表|图表)\s*\d+", text) is not None
    return digit_count >= 25 and ("单位：" in text or hint_count >= 3 or has_table_label)


def extract_tables(page: fitz.Page) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    tables: List[Dict[str, Any]] = []
    try:
        finder = page.find_tables()
        for index, table in enumerate(finder.tables, start=1):
            rows: List[List[str]] = []
            for raw_row in table.extract():
                row = [clean_layout_text(str(cell or "")).replace("\n", " ") for cell in raw_row]
                if any(row):
                    rows.append(row)
            if not rows:
                continue
            tables.append(
                {
                    "table": index,
                    "bbox": [round(float(value), 2) for value in table.bbox],
                    "rows": rows,
                    "text": "\n".join(" | ".join(cell for cell in row) for row in rows),
                }
            )
    except Exception as exc:  # PyMuPDF table detection can fail on malformed vector pages.
        return [], f"{type(exc).__name__}: {exc}"
    return tables, None


def make_page_item(
    page_number: int,
    text: str,
    blocks: Optional[List[Dict[str, Any]]] = None,
    tables: Optional[List[Dict[str, Any]]] = None,
    image_count: int = 0,
    extraction_method: str = "text",
    table_error: Optional[str] = None,
) -> Dict[str, Any]:
    clean = clean_layout_text(text)
    block_values = blocks or []
    table_values = tables or []
    table_text = "\n".join(
        f"[TABLE {item['table']}]\n{item['text']}" for item in table_values if item.get("text")
    )
    search_text = clean
    if table_text:
        search_text = f"{clean}\n{table_text}" if clean else table_text
    quality: Dict[str, Any] = {
        "chars": len(clean),
        "lines": len(clean.splitlines()) if clean else 0,
        "blocks": len(block_values),
        "tables": len(table_values),
        "images": image_count,
        "needs_ocr": len(clean) < 40 and image_count > 0,
        "extraction_method": extraction_method,
    }
    if table_error:
        quality["table_error"] = table_error
    return {
        "page": page_number,
        "text": clean,
        "search_text": search_text,
        "blocks": block_values,
        "sections": extract_sections(clean),
        "clause_numbers": extract_clause_numbers(clean),
        "tables": table_values,
        "quality": quality,
    }


def extract_pdf(pdf_path: Path, extract_table_data: bool = False, use_ocr: bool = False) -> Dict[str, Any]:
    pages: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []
    ocr_available = shutil.which("tesseract") is not None
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            image_count = len(page.get_images(full=True))
            textpage: Optional[fitz.TextPage] = None
            method = "pymupdf_text"
            raw_text = page.get_text("text", sort=True)
            if use_ocr and ocr_available and len(clean_layout_text(raw_text)) < 40 and image_count > 0:
                try:
                    textpage = page.get_textpage_ocr(language="chi_sim+eng", dpi=200, full=True)
                    raw_text = page.get_text("text", textpage=textpage, sort=True)
                    method = "pymupdf_ocr"
                except Exception:
                    method = "pymupdf_text_ocr_failed"
            text = clean_layout_text(raw_text)
            blocks = block_items(page, textpage=textpage)
            tables: List[Dict[str, Any]] = []
            table_error: Optional[str] = None
            if extract_table_data and is_table_candidate(text):
                tables, table_error = extract_tables(page)
            item = make_page_item(
                page_index,
                text,
                blocks=blocks,
                tables=tables,
                image_count=image_count,
                extraction_method=method,
                table_error=table_error,
            )
            pages.append(item)
            full_text_parts.append(f"[PAGE {page_index}]\n{item['search_text']}")
    return {
        "pages": pages,
        "text": "\n\n".join(full_text_parts),
        "ocr_requested": use_ocr,
        "ocr_available": ocr_available,
    }


def extract_text_file(text_path: Path) -> Dict[str, Any]:
    text = clean_layout_text(text_path.read_text(encoding="utf-8"))
    page = make_page_item(1, text, extraction_method="plain_text")
    return {"pages": [page], "text": text, "ocr_requested": False, "ocr_available": False}


class LayoutHTMLParser(HTMLParser):
    BREAK_TAGS = {"br", "p", "div", "li", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6"}
    CELL_TAGS = {"td", "th"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            self.ignored_depth += 1
        elif not self.ignored_depth and tag in self.BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style"} and self.ignored_depth:
            self.ignored_depth -= 1
        elif not self.ignored_depth and tag in self.BREAK_TAGS:
            self.parts.append("\n")
        elif not self.ignored_depth and tag in self.CELL_TAGS:
            self.parts.append(" | ")

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)


def extract_html(html_path: Path) -> Dict[str, Any]:
    parser = LayoutHTMLParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    text = clean_layout_text("".join(parser.parts))
    page = make_page_item(1, text, extraction_method="html_parser")
    return {"pages": [page], "text": text, "ocr_requested": False, "ocr_available": False}


def extract_document(source_path: Path, extract_table_data: bool = False, use_ocr: bool = False) -> Dict[str, Any]:
    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(source_path, extract_table_data=extract_table_data, use_ocr=use_ocr)
    if suffix == ".txt":
        return extract_text_file(source_path)
    if suffix in {".html", ".htm"}:
        return extract_html(source_path)
    raise ValueError("Unsupported file type: {0}".format(source_path))


def derive_title(pages: Sequence[Dict[str, Any]], fallback: str) -> str:
    candidates: List[Tuple[int, str]] = []
    for page_order, page in enumerate(pages[:5]):
        for line_order, line in enumerate(str(page.get("text", "")).splitlines()[:30]):
            compact = line.strip()
            if not 4 <= len(compact) <= 120 or compact == fallback:
                continue
            if len(re.findall(r"[\u4e00-\u9fff]", compact)) < 2:
                continue
            if any(noise in compact for noise in TITLE_NOISE):
                continue
            if re.search(r"\.{4,}|…{2,}", compact):
                continue
            if re.match(r"^\d+(?:\.\d+)*[、.．]\s*", compact):
                continue
            score = (5 - page_order) * 100 + max(0, 30 - line_order) * 2
            score += 80 if any(hint in compact for hint in TITLE_HINTS) else 0
            score += min(len(compact), 80)
            candidates.append((score, compact))
    return max(candidates, default=(0, fallback), key=lambda item: item[0])[1]


def document_quality(pages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    qualities = [page.get("quality", {}) for page in pages]
    return {
        "page_count": len(pages),
        "text_chars": sum(int(item.get("chars", 0)) for item in qualities),
        "empty_pages": sum(int(item.get("chars", 0)) == 0 for item in qualities),
        "short_pages": sum(0 < int(item.get("chars", 0)) < 100 for item in qualities),
        "ocr_candidate_pages": [
            int(page.get("page", 0))
            for page in pages
            if page.get("quality", {}).get("needs_ocr")
        ],
        "table_count": sum(int(item.get("tables", 0)) for item in qualities),
        "section_count": sum(len(page.get("sections", [])) for page in pages),
        "clause_anchor_count": sum(len(page.get("clause_numbers", [])) for page in pages),
        "table_error_pages": [
            int(page.get("page", 0))
            for page in pages
            if page.get("quality", {}).get("table_error")
        ],
    }


def build_quality_report(documents: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    details = [
        {
            "doc_id": item["doc_id"],
            "title": item["title"],
            "source_path": item["source_path"],
            **item["quality"],
        }
        for item in documents
    ]
    return {
        "summary": {
            "document_count": len(details),
            "page_count": sum(item["page_count"] for item in details),
            "empty_pages": sum(item["empty_pages"] for item in details),
            "short_pages": sum(item["short_pages"] for item in details),
            "ocr_candidate_pages": sum(len(item["ocr_candidate_pages"]) for item in details),
            "table_count": sum(item["table_count"] for item in details),
            "section_count": sum(item["section_count"] for item in details),
            "clause_anchor_count": sum(item["clause_anchor_count"] for item in details),
        },
        "documents": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract layout-aware document text for AFAC2026.")
    parser.add_argument("--doc-meta", required=True, help="Path to documents.json")
    parser.add_argument("--docs-root", required=True, help="Root folder of source documents")
    parser.add_argument("--out-dir", required=True, help="Output folder for processed documents")
    parser.add_argument("--extract-tables", action="store_true", help="Detect and preserve table rows on candidate pages")
    parser.add_argument("--ocr", action="store_true", help="OCR image-only pages when Tesseract is installed")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing per-document JSON files")
    parser.add_argument("--quality-report", default="", help="Optional JSON path for extraction quality metrics")
    args = parser.parse_args()

    docs_root = Path(args.docs_root)
    out_dir = ensure_dir(args.out_dir)
    doc_items = normalize_doc_items(read_json_any(args.doc_meta))
    processed_documents: List[Dict[str, Any]] = []

    for item in tqdm(doc_items, desc="preprocess"):
        doc_id = str(item.get("doc_id", "")).strip()
        raw_path = str(item.get("path", "")).strip()
        metadata_title = str(item.get("title", doc_id)).strip()
        if not doc_id or not raw_path:
            continue

        source_path = resolve_doc_path(raw_path, docs_root)
        if not source_path.exists():
            raise FileNotFoundError(f"Source document not found for doc_id={doc_id}: {source_path}")

        output_path = out_dir / f"{doc_id}.json"
        if args.skip_existing and output_path.exists():
            existing = read_json_any(output_path)
            if isinstance(existing, dict) and existing.get("doc_id") == doc_id:
                processed_documents.append(existing)
                continue

        extracted = extract_document(
            source_path,
            extract_table_data=args.extract_tables,
            use_ocr=args.ocr,
        )
        derived_title = derive_title(extracted["pages"], doc_id)
        title = metadata_title if metadata_title and metadata_title != doc_id else derived_title
        quality = document_quality(extracted["pages"])
        processed = {
            "schema_version": 2,
            "doc_id": doc_id,
            "title": title,
            "metadata_title": metadata_title,
            "source_path": str(source_path),
            "source_type": source_path.suffix.lower(),
            "pages": extracted["pages"],
            "text": extracted["text"],
            "quality": quality,
            "ocr_requested": extracted.get("ocr_requested", False),
            "ocr_available": extracted.get("ocr_available", False),
        }
        safe_json_dump(output_path, processed)
        processed_documents.append(processed)

    report = build_quality_report(processed_documents)
    report_path = Path(args.quality_report) if args.quality_report else out_dir / "quality_report.json"
    ensure_dir(report_path.parent)
    safe_json_dump(report_path, report)
    print(f"Processed {len(processed_documents)} documents into: {out_dir}")
    print(f"Quality report saved to: {report_path}")
    if args.ocr and not shutil.which("tesseract"):
        print("Warning: Tesseract is not installed; OCR candidate pages were marked but not recognized.")


if __name__ == "__main__":
    main()
