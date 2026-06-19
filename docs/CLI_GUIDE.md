# 命令行操作指南

> 反映**当前工作流**：per-dataset 专用 encoder + 5-seed 双层 ensemble + LLM 知识蒸馏 (KD)。
> 权威结果见 [EXPERIMENTS.md](EXPERIMENTS.md)，总览见 [README.md](../README.md)。
> （旧的 `train_all_languages.py` / `config_rethink_v3` 单模型流程已弃用。）

## 目录

1. [环境准备](#1-环境准备)
2. [多 seed 训练](#2-多-seed-训练)
3. [双层 ensemble 推理](#3-双层-ensemble-推理)
4. [评测](#4-评测)
5. [vote / topk 阈值扫描](#5-vote--topk-阈值扫描)
6. [知识蒸馏 (KD) 流程](#6-知识蒸馏-kd-流程)
7. [生成提交包](#7-生成提交包)
8. [常用参数说明](#8-常用参数说明)

---

## 1. 环境准备

```bash
bash scripts/setup_env.sh
mamba activate absa
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

每个数据集的 encoder / 数据 / 超参由 `configs/pipeline_<dataset>_v*.yaml` 指定（见结果表）。

---

## 2. 多 seed 训练

**脚本**: `scripts/train_multi_seed.py` — 对一组 seed 依次训练，每 seed 内 `save_top_k=3` 权重平均。

```bash
python scripts/train_multi_seed.py \
    --config configs/pipeline_eng_restaurant_v16.yaml \
    --seeds 42 123 456 789 1024 \
    --save_prefix eng_restaurant_seed \
    --train_file data/subtask_3/eng/eng_restaurant_train_alltasks.jsonl
```

| 参数 | 说明 |
|------|------|
| `--config` | pipeline config（指定 encoder/超参/数据路径） |
| `--seeds` | 随机种子列表（默认 5 个）|
| `--save_prefix` | checkpoint 目录前缀 → `checkpoints/<prefix><seed>` |
| `--train_file` | 训练数据（覆盖 config 中路径，KD 时指向 distill 集）|
| `--pretrained_checkpoint(s)` | （可选）迁移/继续训练的起点权重 |

输出：`checkpoints/<prefix>{42,123,456,789,1024}/`（每个含 top-k 权重）。

---

## 3. 双层 ensemble 推理

**脚本**: `scripts/inference.py --multi_seed_ensemble` — 跨 5 seed 投票 + span top-k 过滤。

```bash
SD=""; for s in 42 123 456 789 1024; do SD="$SD checkpoints/eng_restaurant_seed${s}"; done
python scripts/inference.py --multi_seed_ensemble \
    --seed_dirs $SD \
    --config configs/pipeline_eng_restaurant_v16.yaml \
    --test_file data/subtask_3/eng/eng_restaurant_dev_task3.jsonl \
    --output predictions/eng_restaurant_pred.jsonl \
    --vote_threshold 3 --span_topk_threshold 0.0 --batch_size 16
```

| 参数 | 说明 |
|------|------|
| `--multi_seed_ensemble` | 启用多 seed 双层 ensemble |
| `--seed_dirs` | 5 个 seed checkpoint 目录 |
| `--vote_threshold` | 跨 seed 投票阈值（quad 至少被 N 个 seed 预测才保留，常用 3 或 4）|
| `--span_topk_threshold` | span top-k 概率阈值（0.0 = 不过滤）|
| `--test_file` / `--output` | 输入文本 / 预测输出 |

> 单模型推理仍可用 `--checkpoint <path>`；旧 `--ensemble --checkpoint_dir`（单 seed 内 top-k）也保留。

---

## 4. 评测

**脚本**: `scripts/official_eval.py`（官方评测）

```bash
python scripts/official_eval.py \
    -p predictions/eng_restaurant_pred.jsonl \
    -g data/subtask_3/eng/eng_restaurant_dev_task3.jsonl \
    -t 3
```

| 参数 | 说明 |
|------|------|
| `-p` / `--pred_data_path` | 预测文件 |
| `-g` / `--gold_data_path` | gold 文件 |
| `-t` / `--task` | 任务编号（固定 3）|
| `--do_norm` | 文本归一化（可选）|

输出含 `cF1`（主指标）。

---

## 5. vote / topk 阈值扫描

确定 ensemble 最优 operating point（多数数据集 vote=3 或 4 最佳）：

```bash
SD=""; for s in 42 123 456 789 1024; do SD="$SD checkpoints/<prefix>seed${s}"; done
for vote in 2 3 4; do for topk in 0.0 0.10 0.20; do
  python scripts/inference.py --multi_seed_ensemble --seed_dirs $SD \
    --config <cfg> --test_file <dev> --output /tmp/v${vote}_t${topk}.jsonl \
    --vote_threshold $vote --span_topk_threshold $topk --batch_size 16
  cf1=$(python scripts/official_eval.py -p /tmp/v${vote}_t${topk}.jsonl -g <dev> -t 3 | grep -oE "cF1: [0-9.]+")
  echo "vote=$vote topk=$topk -> $cf1"
done; done
```

---

## 6. 知识蒸馏 (KD) 流程

LLM 对 in-domain 无标注文本打 silver 标签 → 小模型训练 → 推理不用 LLM。

```bash
# (A) silver-labeling — 云端 DeepSeek
export DEEPSEEK_API_KEY=sk-...
python scripts/llm_silver_label.py \
    --input <pool>.jsonl --train_file <dataset>_train_alltasks.jsonl \
    --output <dataset>_silver.jsonl --model deepseek-v4-pro --domain restaurant --n_shots 6

# (A') silver-labeling — 本地 ollama（如 qwen3.6，自动 think:false）
OLLAMA_URL=http://localhost:11433 python scripts/llm_silver_label_ollama.py \
    --input <pool>.jsonl --train_file <dataset>_train_alltasks.jsonl \
    --output <dataset>_silver_qwen.jsonl --model "qwen3.6:latest" --domain restaurant --workers 2

# (B) 无 in-domain 语料 → 本地 LLM 原生生成池（dataset ∈ eng_restaurant/jpn_hotel/zho_restaurant/zho_laptop）
OLLAMA_URL=http://localhost:11433 python scripts/gen_native_ollama.py \
    --dataset zho_restaurant --n 4000 --output <pool>.jsonl --model "qwen3.6:latest"

# (C) distill 训练 = gold + 校验后 silver → 回到 §2 train_multi_seed（--train_file 指向 distill 集）→ §3 推理
```

一键全流程示例：`scripts/run_v23_silver_rus.sh`（DeepSeek）、`scripts/run_v23_qwen_kd_multi.sh`（本地 qwen 多数据集）、`scripts/run_v23_qwen_zho_laptop.sh`（含 silver 封顶）。

KD 生效判据见 [EXPERIMENTS.md §5.29](EXPERIMENTS.md)：silver 须提供 gold 缺失的信息（gold 弱/类别稀疏，或 silver:gold<1 保持少数派）。

---

## 7. 生成提交包

```bash
# 用各数据集最佳 config + vote/topk 跑 test 集，输出 submission/final/<dataset>_test.jsonl
bash scripts/run_test_submission.sh
# 打包（8 个 *_test.jsonl，排除 *_backup.jsonl）→ submission/submission.zip
```

切换某数据集时，旧版本重命名为 `<dataset>_test_v<XX>_backup.jsonl` 保留。

---

## 8. 常用参数说明

### 当前 8 数据集 config（`configs/pipeline_*`）

| 数据集 | config | 方法 |
|---|---|---|
| eng_laptop | `pipeline_eng_laptop_v23_silver.yaml` | KD (DeepSeek) |
| eng_restaurant | `pipeline_eng_restaurant_v16.yaml` | baseline |
| jpn_hotel | `pipeline_jpn_v16.yaml` | baseline |
| rus_restaurant | `pipeline_rus_v23.yaml` | KD (DeepSeek) |
| ukr_restaurant | `pipeline_ukr_v23.yaml` | KD (DeepSeek) |
| tat_restaurant | `pipeline_tat_v23.yaml` | KD (DeepSeek) |
| zho_restaurant | `pipeline_zho_restaurant_v23_qwen.yaml` | KD (qwen3.6) |
| zho_laptop | `pipeline_zho_laptop_v16.yaml` | baseline |

### 典型 vote/topk

| 数据集 | vote | topk |
|---|---|---|
| jpn / zho_laptop / rus | 4 | 0.10 |
| eng_laptop / ukr / zho_restaurant | 4 / 3 | 0.0 |
| tat | 3 | 0.20 |

最优阈值随数据集不同，建议用 §5 扫描确认。
