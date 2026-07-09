# 金融长文本 Agent

本仓库保留金融长文本 Agent 的核心代码、运行脚本和依赖清单。数据集、索引、日志和最终输出不纳入版本库；需要运行时在仓库根目录放置 `public_dataset_upload/`，脚本会自行生成 `processed_data/`、`logs/`、`answer.csv` 和 `evidence.json`。

## 目录

```text
agent/
script/
requirements.txt
README.md
```

`answer.csv` 与 `evidence.json` 会在每道题完成后刷新，完整推理结束后再做最终导出和校验。`processed_data/` 会由 `prepare_data.py` 和 `build_index.py` 生成，包含统一 Page IR、结构化 evidence blocks 和多路检索索引。

## 安装依赖

```bash
cd /Users/dxin/Desktop/learn-claude-code/financial_long_context_agent4
python3 -m pip install -r requirements.txt
```

## 一键运行

```bash
python3 script/run_all.py
```

默认支持断点续跑：如果 `logs/question_results.jsonl` 已存在，脚本会跳过已完成题目，只继续调用剩余题目。若要清空已有结果并从头重跑，使用：

```bash
python3 script/run_all.py --fresh
```

也可以从 Agent 入口运行：

```bash
python3 agent/agent.py
```

脚本默认路径全部绑定到本目录，即使从上级仓库目录执行，也会读写 `financial_long_context_agent4/` 内部文件。

## 分步运行

```bash
python3 script/prepare_data.py
python3 script/build_index.py
python3 script/evaluate_candidate_recall.py
python3 script/answer_questions.py
python3 script/export_answer.py
python3 script/validate_outputs.py
```

`prepare_data.py` 会输出 `documents.jsonl`、`page_ir.jsonl`、`evidence_blocks.jsonl` 和 `parse_summary.json`。`build_index.py` 会基于 evidence blocks 生成 `fulltext_bm25.pkl`、`field_index.pkl`、`article_index.json`、`table_index.jsonl`、`number_index.jsonl` 和 `index_summary.json`。

`evaluate_candidate_recall.py` 会隐藏公开 A 组题目的 `doc_ids`，模拟 B 榜无参考文档输入，统计候选文档是否覆盖真实文档。结果输出到 `logs/candidate_recall_summary.json` 和 `logs/candidate_recall_details.csv`。

模型调用集中在 `agent/qwen_client.py`，主链路只依赖 Qwen 返回的 JSON 答案和 token usage。
