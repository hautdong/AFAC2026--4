import unittest
from collections import Counter

from afac2026_starter.common import tokenize
from afac2026_starter.solve_v4 import EvidencePack, PageChunk
from afac2026_starter.solve_v6 import (
    augment_structured_metric_evidence,
    apply_deterministic_checks,
    compact_review_pack,
    expanded_retrieval_question,
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

    def test_financial_growth_query_gets_report_synonyms(self):
        expanded = expanded_retrieval_question(
            {
                "domain": "financial_reports",
                "options": {"A": "2025年营业总收入增长率高于2024年"},
            }
        )
        self.assertIn("同比增长", expanded["options"]["A"])
        self.assertIn("主营业务分析", expanded["options"]["A"])

    def test_revenue_growth_comparison_keeps_explicit_rate_chunk(self):
        report_chunk = chunk("r2024", 20, "2024年营业总收入4091亿元，同比增长9.5%", set(), 1)
        indexes = {
            "r2024": type("Doc", (), {"chunks": [report_chunk]})(),
        }
        pack = EvidencePack(chunks=[], context="", diagnostics={})
        question = {
            "domain": "financial_reports",
            "doc_ids": ["r2024"],
            "question": "营业总收入增长率比较",
            "options": {"A": "2024年营业总收入增长率"},
        }
        augmented = augment_structured_metric_evidence(question, indexes, pack, 5000)
        self.assertIn("r2024::p20", {item.chunk_id for item in augmented.chunks})
        self.assertIn("9.5%", augmented.context)

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

    def test_overdue_interest_does_not_inherit_liquidated_damages_base(self):
        source = chunk(
            "text03",
            175,
            "逾期利息具体计算方式为本金×票面利率；违约金具体计算方式为延迟支付的本金和利息×票面利率",
            {"A"},
            10,
        )
        pack = EvidencePack(chunks=[source], context=source.text, diagnostics={})
        question = {
            "answer_format": "multi",
            "domain": "financial_contracts",
            "options": {"A": "文档约定的违约利息计算基数包含本金和利息", "B": "其他"},
        }
        parsed = {
            "answer": "AB",
            "judgments": {"A": {"verdict": True}, "B": {"verdict": True}},
        }
        checked = apply_deterministic_checks(question, pack, parsed)
        self.assertFalse(checked["judgments"]["A"]["verdict"])
        self.assertEqual("B", checked["answer"])

    def test_pending_dividend_plan_still_has_stated_terms(self):
        source = chunk(
            "report2025",
            59,
            "2025年度利润分配预案尚需批准，向全体股东每10股派发现金分红69.57元",
            {"B"},
            10,
        )
        pack = EvidencePack(chunks=[source], context=source.text, diagnostics={})
        question = {
            "answer_format": "multi",
            "domain": "financial_reports",
            "options": {"B": "2025年度现金分红方案为每10股派发现金分红69.57元"},
        }
        checked = apply_deterministic_checks(
            question,
            pack,
            {"answer": "", "judgments": {"B": {"verdict": False}}},
        )
        self.assertTrue(checked["judgments"]["B"]["verdict"])

    def test_shareholder_return_comparison_survives_unit_conversion_error(self):
        source = chunk(
            "annual_midea_2025_report",
            49,
            "2025年度公司现金分红与股份回购之总金额超过当年度公司归母净利润",
            {"D"},
            10,
        )
        pack = EvidencePack(chunks=[source], context=source.text, diagnostics={})
        question = {
            "answer_format": "multi",
            "domain": "financial_reports",
            "options": {"D": "2025年度公司现金分红与股份回购之总金额超过了当年归母净利润"},
        }
        checked = apply_deterministic_checks(
            question,
            pack,
            {"answer": "", "judgments": {"D": {"verdict": False}}},
        )
        self.assertTrue(checked["judgments"]["D"]["verdict"])
        self.assertEqual("D", checked["answer"])

    def test_bus_accident_option_implies_transport_coverage(self):
        source = chunk(
            "8",
            2,
            "持有效客票乘坐合法从事客运的营运交通工具，包括公共汽车，遭受意外伤害导致伤残给付保险金",
            set(),
            10,
        )
        pack = EvidencePack(chunks=[source], context=source.text, diagnostics={})
        question = {
            "answer_format": "multi",
            "domain": "insurance",
            "options": {"C": "众安营运交通意外险：乘坐公交车发生车祸导致伤残"},
        }
        checked = apply_deterministic_checks(
            question,
            pack,
            {"answer": "", "judgments": {"C": {"verdict": False}}},
        )
        self.assertTrue(checked["judgments"]["C"]["verdict"])
        self.assertEqual("C", checked["answer"])

    def test_generic_leukemia_recurrence_does_not_prove_first_recurrence(self):
        source = chunk(
            "3",
            1,
            "保险责任限于急性白血病首次复发并在医院接受治疗",
            set(),
            10,
        )
        pack = EvidencePack(chunks=[source], context=source.text, diagnostics={})
        question = {
            "answer_format": "multi",
            "domain": "insurance",
            "options": {"B": "众安白血病医疗险：白血病复发住院"},
        }
        checked = apply_deterministic_checks(
            question,
            pack,
            {"answer": "B", "judgments": {"B": {"verdict": True}}},
        )
        self.assertFalse(checked["judgments"]["B"]["verdict"])
        self.assertEqual("", checked["answer"])

    def test_regulatory_effective_dates_follow_option_title_order(self):
        first = chunk(
            "doc_due",
            1,
            "《客户尽职调查办法》自2026年1月1日起施行",
            set(),
            10,
        )
        second = chunk(
            "doc_beneficial",
            1,
            "《受益所有人识别办法》自2026年1月20日起施行",
            set(),
            10,
        )
        pack = EvidencePack(chunks=[first, second], context="", diagnostics={})
        question = {
            "answer_format": "multi",
            "domain": "regulatory",
            "options": {
                "D": "《客户尽职调查办法》的施行日期早于《受益所有人识别办法》",
            },
        }
        checked = apply_deterministic_checks(
            question,
            pack,
            {"answer": "", "judgments": {"D": {"verdict": False}}},
        )
        self.assertTrue(checked["judgments"]["D"]["verdict"])
        self.assertEqual("D", checked["answer"])

    def test_regulatory_classification_effective_date_literal(self):
        source = chunk(
            "csrc",
            12,
            "本规定自2025 年8 月22 日起施行",
            set(),
            10,
        )
        pack = EvidencePack(chunks=[source], context=source.text, diagnostics={})
        question = {
            "answer_format": "multi",
            "domain": "regulatory",
            "options": {"D": "相关分类监管规定的施行时间为 2025 年 8 月 22 日"},
        }
        checked = apply_deterministic_checks(
            question,
            pack,
            {"answer": "", "judgments": {"D": {"verdict": False}}},
        )
        self.assertTrue(checked["judgments"]["D"]["verdict"])

    def test_research_tf_comparison_keeps_proposition_direction(self):
        first = chunk("pack2_text01", 4, "韩国寿险银保渠道近20年的复合增速达到12%", set(), 10)
        second = chunk("pack2_text19", 1, "全球RFID标签出货量复合增速达14.1%", set(), 10)
        pack = EvidencePack(chunks=[first, second], context="", diagnostics={})
        question = {
            "answer_format": "tf",
            "domain": "research",
            "question": "韩国寿险银保渠道近20年的复合增速低于远望谷在新兴赛道的RFID标签出货量复合增速。",
            "options": {"A": "正确", "B": "错误"},
        }
        checked = apply_deterministic_checks(
            question,
            pack,
            {"answer": "B", "judgments": {"A": {"verdict": False}, "B": {"verdict": True}}},
        )
        self.assertTrue(checked["judgments"]["A"]["verdict"])
        self.assertFalse(checked["judgments"]["B"]["verdict"])
        self.assertTrue(checked["proposition_verdict"])
        self.assertEqual("A", checked["answer"])

    def test_research_literal_metrics_are_not_overly_narrowed(self):
        source = chunk(
            "report",
            1,
            "韩国银保渠道保费贡献超过50%，近20年复合增速达到12%；2025年金融信创市场规模预计接近2500亿元；"
            "2008年至2025Q1-3客户资金杠杆从1.56倍提升至4.09倍；2023-2025年居民可支配收入增速从6.33%降至4.99%",
            set(),
            10,
        )
        pack = EvidencePack(chunks=[source], context=source.text, diagnostics={})
        question = {
            "answer_format": "multi",
            "domain": "research",
            "options": {
                "A": "韩国寿险银保渠道保费贡献率在过去20年复合增速达到12%",
                "B": "2025年金融信创市场规模预计接近2500亿元",
                "C": "2008年至2025Q1-3我国上市券商客户资金杠杆从1.56倍提升至4.09倍",
                "D": "2023年至2025年间居民可支配收入增速从6.33%下降至4.99%",
            },
        }
        checked = apply_deterministic_checks(
            question,
            pack,
            {"answer": "", "judgments": {letter: {"verdict": False} for letter in "ABCD"}},
        )
        self.assertEqual("ABCD", checked["answer"])

    def test_research_scope_and_ict_forecast_are_literal(self):
        source = chunk(
            "report",
            1,
            "IDC预测2029年中国ICT市场规模接近8894.3亿美元；韩国银保渠道保费贡献超过50%，支撑了韩国人身险保费的快速增长",
            set(),
            10,
        )
        pack = EvidencePack(chunks=[source], context=source.text, diagnostics={})
        question = {
            "answer_format": "multi",
            "domain": "research",
            "options": {
                "A": "2029年中国ICT市场规模预计接近8894.3亿美元",
                "B": "韩国寿险银保渠道保费贡献超过50%，支撑了韩国寿险管理体系",
            },
        }
        checked = apply_deterministic_checks(
            question,
            pack,
            {"answer": "", "judgments": {"A": {"verdict": False}, "B": {"verdict": True}}},
        )
        self.assertTrue(checked["judgments"]["A"]["verdict"])
        self.assertFalse(checked["judgments"]["B"]["verdict"])
        self.assertEqual("A", checked["answer"])

    def test_consecutive_duration_must_reach_report_year(self):
        source = chunk(
            "annual_midea_2024_report",
            44,
            "自2019年起公司连续四年推出回购计划",
            {"A"},
            10,
        )
        source.title = "美的集团2024年年度报告"
        pack = EvidencePack(chunks=[source], context=source.text, diagnostics={})
        question = {
            "question": "美的集团自2019年起连续实施了股份回购方案。",
            "answer_format": "tf",
            "domain": "financial_reports",
            "options": {"A": "正确", "B": "错误"},
        }
        checked = apply_deterministic_checks(
            question,
            pack,
            {"answer": "A", "judgments": {"A": {"verdict": True}, "B": {"verdict": False}}},
        )
        self.assertFalse(checked["judgments"]["A"]["verdict"])
        self.assertTrue(checked["judgments"]["B"]["verdict"])
        self.assertEqual("B", checked["answer"])

    def test_default_description_or_related_clause_is_disjunction(self):
        first = chunk("text02", 192, "违约情形及认定：以下情形构成本期债券项下的违约", {"D"}, 10)
        second = chunk("text14", 180, "二、违约责任及解决措施：以下事件构成违约事件", {"D"}, 10)
        second.doc_order = 2
        pack = EvidencePack(chunks=[first, second], context="", diagnostics={})
        question = {
            "answer_format": "multi",
            "domain": "financial_contracts",
            "doc_ids": ["text02", "text14"],
            "options": {"D": "两份文件均提到了具体的违约情形描述或相关条款"},
        }
        checked = apply_deterministic_checks(
            question,
            pack,
            {"answer": "", "judgments": {"D": {"verdict": False}}},
        )
        self.assertTrue(checked["judgments"]["D"]["verdict"])
        self.assertEqual("D", checked["answer"])


if __name__ == "__main__":
    unittest.main()
