$ErrorActionPreference = "Stop"

python -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "已创建 .env，请先填入 DASHSCOPE_API_KEY 后再重新运行。" -ForegroundColor Yellow
    exit 1
}

python -m afac2026_starter.prepare_public_a `
    --dataset-root "public_dataset_a" `
    --out-dir "data"

python -m afac2026_starter.preprocess `
    --doc-meta "data\metadata\documents.json" `
    --docs-root "public_dataset_a\raw" `
    --out-dir "cache\processed_docs"

python -m afac2026_starter.solve `
    --questions "data\questions\public_a.json" `
    --doc-meta "data\metadata\documents.json" `
    --processed-dir "cache\processed_docs" `
    --output "results\answer.csv" `
    --evidence-output "results\evidence.json"
