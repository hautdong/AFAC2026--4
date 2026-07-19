import unittest

from afac2026_starter.solve_v4 import (
    DocumentIndex,
    PageChunk,
    extract_anchors,
    parse_answer,
    select_evidence,
)


def make_chunk(doc_id, page, text, order):
    from collections import Counter
    from afac2026_starter.common import tokenize

    return PageChunk(
        doc_id=doc_id,
        title=doc_id,
        doc_order=0,
        page=page,
        chunk_order=order,
        chunk_id=f"{doc_id}::p{page}::c{order}",
        text=text,
        tokens=Counter(tokenize(text)),
    )


class SolveV4Tests(unittest.TestCase):
    def test_extracts_financial_and_article_anchors(self):
        anchors = extract_anchors("根据第四十二条，2025 年收入增长 12.5%，金额达到 30 亿元")
        compact = {item.replace(" ", "") for item in anchors}
        self.assertIn("第四十二条", compact)
        self.assertIn("2025年", compact)
        self.assertIn("12.5%", compact)
        self.assertIn("30亿元", compact)

    def test_tf_uses_proposition_verdict(self):
        answer, issues = parse_answer(
            {"proposition_verdict": False, "answer": "B", "judgments": {}},
            "tf",
        )
        self.assertEqual("B", answer)
        self.assertEqual([], issues)

    def test_mcq_flags_multiple_true_options(self):
        answer, issues = parse_answer(
            {
                "answer": "B",
                "judgments": {
                    "A": {"verdict": True},
                    "B": {"verdict": True},
                },
            },
            "mcq",
        )
        self.assertEqual("B", answer)
        self.assertIn("mcq_true_count_2", issues)

    def test_retrieval_keeps_evidence_from_every_document(self):
        indexes = {
            "d1": DocumentIndex(
                "d1",
                "d1",
                [],
                [make_chunk("d1", 1, "2024年营业收入100亿元", 0)],
            ),
            "d2": DocumentIndex(
                "d2",
                "d2",
                [],
                [make_chunk("d2", 1, "2025年营业收入120亿元", 0)],
            ),
        }
        question = {
            "qid": "q1",
            "question": "比较两年营业收入",
            "options": {"A": "2025年高于2024年", "B": "2025年低于2024年"},
            "answer_format": "mcq",
            "doc_ids": ["d1", "d2"],
        }
        pack = select_evidence(question, indexes, 10000)
        self.assertEqual({"d1", "d2"}, {chunk.doc_id for chunk in pack.chunks})
        self.assertIn("DOCUMENT 1", pack.context)
        self.assertIn("DOCUMENT 2", pack.context)
        self.assertIn("d1::p1::c0", {chunk.chunk_id for chunk in pack.chunks})
        self.assertIn("d2::p1::c0", {chunk.chunk_id for chunk in pack.chunks})

    def test_retrieval_carries_continuation_chunk(self):
        first = make_chunk("d1", 1, "利润分配预案和现金分红政策", 0)
        second = make_chunk("d1", 2, "按归属于股东净利润的50%实施现金分红", 1)
        indexes = {"d1": DocumentIndex("d1", "d1", [], [first, second])}
        question = {
            "qid": "q2",
            "question": "公司是否实施现金分红政策",
            "options": {"A": "按净利润50%现金分红", "B": "未分红"},
            "answer_format": "mcq",
            "doc_ids": ["d1"],
        }
        pack = select_evidence(question, indexes, 10000)
        self.assertIn("d1::p2::c1", {chunk.chunk_id for chunk in pack.chunks})


if __name__ == "__main__":
    unittest.main()
