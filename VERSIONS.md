# Version Log

## v4-score-70.7675

- Date: 2026-07-20
- Public A score: 70.7675
- Total tokens: 1,736,811
- Submit file: `results/v4_candidate/answer.csv`

Changes:

- Added page-aware, document-coverage retrieval with mandatory document headers and early summary evidence.
- Added numeric, percentage, date, article-number, and financial-scope anchors.
- Added continuation chunks so tables and clauses spanning extracted pages are not truncated.
- Added structured evidence memory with chunk citations and confidence values.
- Added domain checks for financial reports, insurance formulas/rankings, contracts, regulations, and research reports.
- Added conditional second-pass verification only for low-confidence, numeric, unsupported, or inconsistent answers.
- Added true/false proposition handling, answer consistency validation, API retries, incremental checkpoints, and resume support.

## v2-score-61.3618

- Date: 2026-07-19
- Public A score: 61.3618
- Estimated accuracy: about 67/100
- Total tokens: 1,402,540
- Submit file: `results/v2_submit/answer.csv`

Changes:

- Added option-specific evidence retrieval.
- Grouped evidence as global and per-option context.
- Added domain guidance for contracts, reports, insurance, regulatory, and research questions.
- Used per-option `judgments` to derive the final answer letters.

## v1-score-48

- Date: 2026-07-19
- Public A score: about 48
- Estimated accuracy: about 50/100
- Total tokens: 577,024
- Submit file: `results/answer.csv`

Changes:

- Initial runnable baseline.
- Keyword chunk retrieval.
- Single-pass Qwen answer generation.
