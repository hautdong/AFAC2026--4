import tempfile
import unittest
from pathlib import Path

from afac2026_starter.common import clean_layout_text
from afac2026_starter.preprocess import (
    derive_title,
    document_quality,
    extract_html,
    make_page_item,
)
from afac2026_starter.solve_v4 import split_page_text


class PreprocessTests(unittest.TestCase):
    def test_layout_cleaner_preserves_lines(self):
        text = clean_layout_text("第一章  总则\n\n项目\t2025 年\t2024 年\n营业收入  100  90")
        self.assertEqual(3, len(text.splitlines()))
        self.assertIn("项目 2025 年 2024 年", text)

    def test_page_item_keeps_sections_clauses_and_tables(self):
        page = make_page_item(
            3,
            "第一章 总则\n第十二条 金融机构应当核验信息",
            tables=[
                {
                    "table": 1,
                    "bbox": [0, 0, 10, 10],
                    "rows": [["项目", "2025年"], ["收入", "100"]],
                    "text": "项目 | 2025年\n收入 | 100",
                }
            ],
            image_count=1,
        )
        self.assertEqual("第一章 总则", page["sections"][0]["text"])
        self.assertIn("第十二条", page["clause_numbers"])
        self.assertIn("[TABLE 1]", page["search_text"])
        self.assertEqual(1, page["quality"]["tables"])

    def test_quality_marks_image_only_page_for_ocr(self):
        pages = [
            make_page_item(1, "", image_count=1),
            make_page_item(2, "第二页有可检索文本", image_count=0),
        ]
        quality = document_quality(pages)
        self.assertEqual(1, quality["empty_pages"])
        self.assertEqual([1], quality["ocr_candidate_pages"])

    def test_derived_title_prefers_document_title_line(self):
        pages = [
            make_page_item(
                1,
                "证券代码：000001\n美的集团股份有限公司2025年年度报告全文\n第一章 重要提示",
            )
        ]
        self.assertEqual("美的集团股份有限公司2025年年度报告全文", derive_title(pages, "doc01"))

    def test_html_parser_preserves_block_and_table_boundaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.html"
            path.write_text(
                "<h1>管理办法</h1><p>第一条 总则</p><table><tr><td>项目</td><td>金额</td></tr></table>",
                encoding="utf-8",
            )
            extracted = extract_html(path)
        lines = extracted["pages"][0]["text"].splitlines()
        self.assertGreaterEqual(len(lines), 3)
        self.assertIn("第一条", extracted["pages"][0]["clause_numbers"])

    def test_chunker_does_not_flatten_table_rows(self):
        text = "\n".join(f"项目{i} | {i * 100} | {i * 90}" for i in range(80))
        chunks = split_page_text(text, target_size=260, overlap=40)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all("\n" in chunk for chunk in chunks))
        self.assertIn("项目0 | 0 | 0", chunks[0])


if __name__ == "__main__":
    unittest.main()
