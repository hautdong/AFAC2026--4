# Version Log

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
