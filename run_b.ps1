$ErrorActionPreference = "Stop"

python -m afac2026_starter.prepare_public_b `
  --question-dir upload_b\question_b `
  --raw-root public_dataset_a\raw `
  --out-dir data_b

python -m afac2026_starter.preprocess `
  --doc-meta data_b\metadata\documents.json `
  --docs-root public_dataset_a\raw `
  --out-dir cache\processed_docs_b `
  --skip-existing

python -m afac2026_starter.solve_b `
  --questions data_b\questions\public_b.json `
  --doc-meta data_b\metadata\documents.json `
  --processed-dir cache\processed_docs_b `
  --template upload_b\submit.csv `
  --output results\b_final\answer.csv `
  --evidence-output results\b_final\evidence.json `
  --resume
