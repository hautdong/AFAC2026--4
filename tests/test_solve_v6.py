import unittest
from collections import Counter

from afac2026_starter.common import tokenize
from afac2026_starter.solve_v4 import EvidencePack, PageChunk
from afac2026_starter.solve_v6 import (
    apply_deterministic_checks,
    compact_review_pack,
    literal_evidence_hints,
    question_profile,
    retrieval_settings,
    semantic_guardrails,
)


def chunk(doc_id, page, text, labels, score):
    return PageChunk(
        doc_id=doc_id,
        title=doc_id,
        doc_order=1 if doc_id == "d1" else 2,
        page=page,
        chunk_order=page,
        chunk_id=f"{doc_id}::p{page}",
        text=text,
        tokens=Counter(tokenize(text)),
        score=score,
        matched_for=set(labels),
    )


class SolveV6Tests(unittest.TestCase):
    def test_question_profile_extracts_numeric_scope(self):
        profile = question_profile(
            {
                "question": "2025年本期金额是否超过10亿元？",
                "options": {"A": "超过10亿元", "B": "不超过10亿元"},
                "doc_ids": ["d1", "d2"],
                "answer_format": "tf",
            }
        )
        self.assertIn("2025年", profile["numbers_dates_units"])
        self.assertIn("10亿元", profile["numbers_dates_units"])
        self.assertIn("本期", profile["scope_checks"])
        self.assertTrue(profile["requires_cross_document_check"])

    def test_accuracy_mode_keeps_full_evidence_depth(self):
        settings = retrieval_settings({"domain": "financial_reports"})
        self.assertTrue(settings["include_early_summary"])
        self.assertEqual(3, settings["continuation_limit"])
        self.assertEqual(24000, settings["max_chars"])

    def test_compact_review_keeps_both_documents(self):
        chunks = [
            chunk("d1", 1, "文档一首页", {"DOC_HEADER"}, 5),
            chunk("d1", 2, "选项A事实", {"A"}, 10),
            chunk("d2", 1, "文档二首页", {"DOC_HEADER"}, 5),
            chunk("d2", 2, "选项A对比事实", {"A", "REVIEW_1"}, 12),
        ]
        pack = EvidencePack(chunks=chunks, context="", diagnostics={})
        question = {
            "doc_ids": ["d1", "d2"],
            "options": {"A": "两份文档均满足", "B": "不满足"},
        }
        compact = compact_review_pack(question, pack, {"judgments": {}}, 5000)
        self.assertEqual({"d1", "d2"}, {item.doc_id for item in compact.chunks})
        self.assertIn("DOCUMENT 1", compact.context)
        self.assertIn("DOCUMENT 2", compact.context)

    def test_literal_hint_preserves_listing_field(self):
        source = chunk(
            "d1",
            1,
            "证券简称：海峡股份 股票代码：002320 上市地点：深圳证券交易所",
            {"B"},
            10,
        )
        pack = EvidencePack(chunks=[source], context=source.text, diagnostics={})
        question = {
            "options": {"B": "文档明确标注了证券上市地点为深圳证券交易所"},
        }
        hints = literal_evidence_hints(question, pack)
        self.assertEqual("d1::p1", hints["B"][0]["chunk_id"])
        self.assertIn("上市地点", hints["B"][0]["matched_anchors"])

    def test_dividend_policy_guardrail(self):
        guards = semantic_guardrails(
            {
                "question": "是否实施现金分红政策",
                "options": {"A": "按净利润50%实施现金分红"},
            }
        )
        self.assertTrue(any("50% cash-dividend policy" in guard for guard in guards))

    def test_explicit_listing_field_overrides_model_narrowing(self):
        source = chunk(
            "text07",
            1,
            "证券简称：海峡股份 上市地点：深圳证券交易所",
            {"B"},
            10,
        )
        pack = EvidencePack(chunks=[source], context=source.text, diagnostics={})
        question = {
            "answer_format": "multi",
            "options": {
                "A": "发行人名称正确",
                "B": "文档中明确标注了证券上市地点为深圳证券交易所",
            },
        }
        parsed = {
            "answer": "A",
            "judgments": {
                "A": {"verdict": True},
                "B": {"verdict": False},
            },
        }
        checked = apply_deterministic_checks(question, pack, parsed)
        self.assertTrue(checked["judgments"]["B"]["verdict"])
        self.assertEqual("AB", checked["answer"])

    def test_source_characterized_trend_does_not_require_every_year(self):
        source = chunk(
            "report",
            12,
            "居民可支配收入增长动能减弱，收入增速放缓，2023-2025 年从6.33%降至4.99%",
            {"D"},
            10,
        )
        pack = EvidencePack(chunks=[source], context=source.text, diagnostics={})
        question = {
            "answer_format": "multi",
            "options": {"D": "2023年至2025年居民收入增速呈现持续放缓趋势"},
        }
        parsed = {"answer": "A", "judgments": {"D": {"verdict": False}}}
        checked = apply_deterministic_checks(question, pack, parsed)
        self.assertTrue(checked["judgments"]["D"]["verdict"])
        self.assertEqual("D", checked["answer"])

    def test_simple_numeric_field_does_not_gain_unstated_scope(self):
        source = chunk(
            "text10",
            33,
            "标的公司控股股东力诺投资的资产负债率为43.24%",
            {"D"},
            10,
        )
        pack = EvidencePack(chunks=[source], context=source.text, diagnostics={})
        question = {
            "answer_format": "multi",
            "domain": "financial_contracts",
            "doc_ids": ["text01", "text10"],
            "options": {"D": "第二份文档中标的公司控股股东的资产负债率为43.24%"},
        }
        parsed = {"answer": "A", "judgments": {"D": {"verdict": False}}}
        checked = apply_deterministic_checks(question, pack, parsed)
        self.assertTrue(checked["judgments"]["D"]["verdict"])
        self.assertEqual("D", checked["answer"])


if __name__ == "__main__":
    unittest.main()
