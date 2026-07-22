import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from afac2026_starter.solve_b import (
    load_template,
    normalize_open_answers,
    rank_candidate_documents,
    write_b_csv,
)
from afac2026_starter.solve_v4 import DocumentIndex, PageChunk, answer_protocol


def document(doc_id, title, text):
    chunk = PageChunk(
        doc_id=doc_id,
        title=title,
        doc_order=0,
        page=1,
        chunk_order=0,
        chunk_id=f"{doc_id}::p1::c0",
        text=text,
        tokens=Counter(),
    )
    from afac2026_starter.common import tokenize

    chunk.tokens = Counter(tokenize(text))
    return DocumentIndex(doc_id=doc_id, title=title, pages=[], chunks=[chunk])


class SolveBTests(unittest.TestCase):
    def test_multi_protocol_does_not_require_exhaustive_restating(self):
        protocol = answer_protocol({"answer_format": "multi"})
        self.assertIn("option's own claims are true", protocol)
        self.assertIn("专项资管计划等", protocol)

    def test_exact_regulation_title_ranks_first(self):
        indexes = {
            "target": document("target", "银行卡清算机构管理办法", "申请受理后九十日内作出决定"),
            "other": document("other", "证券公司分类评价规定", "证券公司评价计分"),
        }
        question = {
            "question": "根据《银行卡清算机构管理办法》，申请受理后多久作出决定？",
            "options": {},
            "domain": "regulatory",
            "type": "计算题",
            "answer_format": "open",
        }
        ranked = rank_candidate_documents(question, indexes, list(indexes), limit=2)
        self.assertEqual("target", ranked[0].doc_id)

    def test_open_answer_normalization_preserves_ordering(self):
        answers = normalize_open_answers(["美的 ＞ 宁德时代", " 2.35 "], 2)
        self.assertEqual(["美的>宁德时代", "2.35"], answers)
        self.assertEqual([], normalize_open_answers(["1.00"], 2))

    def test_template_and_writer_use_b_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            template = Path(temp_dir) / "submit.csv"
            template.write_text(
                "qid,answer_1,answer_2,answer_3,answer_4,prompt_tokens,completion_tokens,total_tokens\n"
                "summary,,,,,0,0,0\nq1,999999.99,999999.99,,,0,0,0\n",
                encoding="utf-8-sig",
            )
            order, patterns = load_template(template)
            output = Path(temp_dir) / "answer.csv"
            write_b_csv(
                output,
                [
                    {
                        "qid": "q1",
                        "answer_1": "1.00",
                        "answer_2": "2.00",
                        "answer_3": "",
                        "answer_4": "",
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    }
                ],
                {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                order,
            )
            with output.open("r", encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
        self.assertEqual(["999999.99", "999999.99"], patterns["q1"])
        self.assertEqual("summary", rows[0]["qid"])
        self.assertEqual("1.00", rows[1]["answer_1"])


if __name__ == "__main__":
    unittest.main()
