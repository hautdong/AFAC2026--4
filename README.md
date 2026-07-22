# AFAC2026 赛题四最小可运行 Starter

这个 starter 不是冲榜方案，它的目标只有两个：

1. 让你先把比赛流程真正跑通。
2. 让你看懂这类题最小闭环到底长什么样。

它做的事情很简单：

1. 把 PDF 提取成文本。
2. A 榜优先使用题目自带的 `doc_ids` 找文档。
3. B 榜用纯关键词规则做候选文档检索。
4. 从候选文档里切块，挑相关片段。
5. 调用 Qwen API 回答，并输出 `answer.csv`。

## 1. 你先要知道的事

根据你给我的赛题说明，这个比赛的核心限制是：

1. 正式答题阶段只能调用 Qwen 系列模型 API。
2. 不能用 embedding 模型做检索和推理。
3. 最终提交的是 `answer.csv`，并且要带 token 统计。

另外，按天池比赛页面当前公开信息，比赛时间是 `2026-06-06` 到 `2026-07-21`，A 榜评测到 `2026-07-21 20:00`，B 榜评测从 `2026-07-22 00:00` 到 `2026-07-24 17:00`。如果你现在是先学会流程，这个 starter 也一样能本地跑通。  
官方页面：https://tianchi.aliyun.com/competition/entrance/532486  
赛题介绍页：https://tianchi.aliyun.com/competition/entrance/532486/information

## 2. 你需要准备什么

### 账号

1. 天池账号
2. 阿里云百炼账号
3. 百炼 API Key

### 本地环境

1. Python 3.8+
2. Windows PowerShell
3. 可选：conda 环境

你刚刚告诉我你有 `conda` 环境 `huang_3.8`，这个 starter 我已经按 Python 3.8 兼容来处理了。

## 3. 数据怎么放

你现在已经是官方原始数据结构，这个 starter 已经支持直接使用，不需要你手工造 `documents.json`。

把数据放成这样就可以：

```text
afac2026_starter/
  public_dataset_a/
    questions/
      group_a/
        financial_contracts_questions.json
        financial_reports_questions.json
        insurance_questions.json
        regulatory_questions.json
        research_questions.json
    raw/
      financial_contracts/
      financial_reports/
      insurance/
      regulatory/
      research/
```

运行时程序会自动：

1. 合并五个题目文件
2. 扫描 `raw/` 下真实文件
3. 自动生成 `data/metadata/documents.json`
4. 自动生成 `data/questions/public_a.json`

也就是说，只要 `public_dataset_a/` 放对，你就不用再操心元数据格式。

## 4. 先配置 API

复制配置文件：

```powershell
cd E:\lunwen_jiangjie\afac2026_starter
Copy-Item .env.example .env
```

然后编辑 `.env`：

```env
DASHSCOPE_API_KEY=你的百炼APIKey
QWEN_MODEL=qwen3.6-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

说明：

1. `QWEN_MODEL` 可以改成你有权限调用的 Qwen 模型。
2. 如果你用的是百炼新的工作空间专属域名，也可以把 `QWEN_BASE_URL` 换成专属地址。
3. 这个 starter 默认用的是兼容 OpenAI 的调用方式。

百炼官方文档：

1. OpenAI 兼容调用说明：https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope
2. 文本模型 API 总览：https://help.aliyun.com/zh/model-studio/qwen-api-reference/

## 5. 零基础最短跑通命令

### 第一步：进入目录

```powershell
cd E:\lunwen_jiangjie\afac2026_starter
```

### 第二步：激活你的 conda 环境

```powershell
conda activate huang_3_8
```

### 第三步：一键跑 A 榜数据

```powershell
.\run_a.ps1
```

这一步会自动做 4 件事：

1. 安装依赖
2. 从 `public_dataset_a` 自动生成题目和文档元数据
3. 提取文本到 `cache\processed_docs`
4. 生成结果到 `results\answer.csv`

### 更推荐你先跑一个 3 题小测试

```powershell
.\run_smoke_test.ps1
```

它会只跑前 3 题，并输出：

1. `results\answer_smoke.csv`
2. `results\evidence_smoke.json`

这一步特别适合先检查：

1. API Key 是否有效
2. 依赖有没有装齐
3. 数据路径是不是正确
4. Qwen 调用能不能成功返回

## 6. 如果你想手动一条条执行

### 先激活 conda 环境

```powershell
conda activate huang_3_8
```

### 先把官方数据包转成 starter 需要的元数据

```powershell
python -m afac2026_starter.prepare_public_a `
  --dataset-root public_dataset_a `
  --out-dir data
```

### 安装依赖

```powershell
pip install -r requirements.txt
```

### 预处理 PDF

```powershell
python -m afac2026_starter.preprocess `
  --doc-meta data\metadata\documents.json `
  --docs-root public_dataset_a\raw `
  --out-dir cache\processed_docs
```

### 生成答案

```powershell
python -m afac2026_starter.solve `
  --questions data\questions\public_a.json `
  --doc-meta data\metadata\documents.json `
  --processed-dir cache\processed_docs `
  --output results\answer.csv `
  --evidence-output results\evidence.json
```

## 7. 输出文件长什么样

程序会生成：

1. `results\answer.csv`
2. `results\evidence.json`

其中 `answer.csv` 第一行是总 token 统计，后面每一行对应一道题。

## 8. 这个 starter 现在能做什么，不能做什么

### 能做什么

1. 跑通数据预处理
2. 跑通 Qwen API 调用
3. 输出符合比赛要求的基础 `answer.csv`
4. 让你开始调试 A 榜

### 不能做什么

1. 现在准确率不会高
2. B 榜检索只是最简单关键词规则
3. 没有做复杂记忆压缩
4. 没有做更强的多轮证据校验

所以你可以把它理解为：这是“能交卷”的第一版，不是“能冲榜”的最终版。

## 9. 下一步怎么优化

你先别急着追求高分，建议按这个顺序升级：

1. 先确认 `answer.csv` 能稳定生成。
2. 再看 `evidence.json`，确认检索片段对不对。
3. 然后把 prompt 改成按选项逐项判断。
4. 再做领域规则，比如保险、监管、财报分别写不同提示词。
5. 最后再做 token 优化和动态记忆压缩。

## 10. 常见报错

### 1) `401 Unauthorized`

一般是 `DASHSCOPE_API_KEY` 错了，或者没写进 `.env`。

### 2) `404 Not Found`

一般是 `QWEN_BASE_URL` 或模型名不对。

### 3) 找不到文档

一般是 `public_dataset_a\raw` 目录不完整，或者你不是用官方原始目录结构。

### 4) 输出答案为空

一般是模型没按要求输出字母。这个 starter 已经做了基础兜底，但你仍然要检查 `evidence.json`。

## 11. 你最应该怎么用它

如果你是 0 基础，我建议你就做这三件事：

1. 先把数据按目录放好。
2. 把 `.env` 配好。
3. 执行 `.\run_a.ps1`。

只要这一步通了，你就已经超过“完全不会开始”的阶段了。

## 12. B 榜运行方式

B 榜题目放在 `upload_b\question_b`，提交模板为 `upload_b\submit.csv`。当前 B 管线会在 573 份原始材料中先做分领域词法召回，再由 Qwen 选择 1-4 份文档，最后执行证据检索、逐项判断或计算和独立复核，全程不使用 embedding 模型。

确认 `.env` 使用可调用的 `qwen3.6-plus` 后运行：

```powershell
conda activate huang_3_8
cd E:\lunwen_jiangjie\afac2026_starter
.\run_b.ps1
```

脚本支持断点续跑，结果位于：

```text
results\b_final\answer.csv
results\b_final\evidence.json
```

中断后重新执行 `run_b.ps1` 即可跳过已经完成的题目。只重跑某一道题可使用：

```powershell
python -m afac2026_starter.solve_b `
  --questions data_b\questions\public_b.json `
  --doc-meta data_b\metadata\documents.json `
  --processed-dir cache\processed_docs_b `
  --template upload_b\submit.csv `
  --output results\b_check\answer.csv `
  --evidence-output results\b_check\evidence.json `
  --qid fc_b_002
```
