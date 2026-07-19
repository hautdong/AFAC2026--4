# Version Log

## v5-score-69.3513

- Date: 2026-07-20
- Public A score: 69.3513
- Total tokens: 2,035,582
- Submit file: `results/v5_graph_confirmed/answer.csv`

Changes:

- Compared v2, graph RAG, and v4 answers to isolate 19 unresolved questions.
- Re-arbitrated those questions using page-aware evidence plus graph-retrieved source chunks.
- Required valid citations, high confidence, and exact agreement with the graph answer before overriding v4.
- Changed only `fc_a_004` from `AD` to `A`; retained v4 for the other 99 questions.

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
