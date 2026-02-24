# 命令行操作指南

## 目录

1. [环境准备](#1-环境准备)
2. [单数据集训练](#2-单数据集训练)
3. [全语言批量训练](#3-全语言批量训练)
4. [推理（生成预测）](#4-推理生成预测)
5. [评测](#5-评测)
6. [批量评测](#6-批量评测)
7. [Loss 权重搜索](#7-loss-权重搜索)
8. [常用参数说明](#8-常用参数说明)

---

## 1. 环境准备

```bash
pip install -r requirements.txt
```

所需环境：Python 3.10+, PyTorch 2.0+, transformers, tqdm, pyyaml

---

## 2. 单数据集训练

**脚本**: `scripts/train.py`

```bash
# 基本用法（使用 V3 配置）
python scripts/train.py --config configs/config_rethink_v3.yaml

# 指定数据文件（覆盖 config 中的路径）
python scripts/train.py --config configs/config_rethink_v3.yaml \
    --train_file data/subtask_3/eng/eng_restaurant_train_alltasks.jsonl \
    --val_file data/subtask_3/eng/eng_restaurant_dev_task3.jsonl

# 指定 GPU
python scripts/train.py --config configs/config_rethink_v3.yaml --device cuda:1
```

**参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `configs/config.yaml` | 配置文件路径 |
| `--train_file` | config 中的值 | 训练数据路径（覆盖 config） |
| `--val_file` | config 中的值 | 验证数据路径（覆盖 config） |
| `--device` | auto | 设备（cuda/cpu） |

**输出**:
- 权重: `{save_dir}/checkpoint_best.pt`, `checkpoint_top2.pt`, `checkpoint_top3.pt`
- 日志: `training.log`

---

## 3. 全语言批量训练

**脚本**: `scripts/train_all_languages.py`

支持的 8 个数据集:
`eng_restaurant`, `eng_laptop`, `jpn_hotel`, `rus_restaurant`, `tat_restaurant`, `ukr_restaurant`, `zho_restaurant`, `zho_laptop`

```bash
# 训练全部 8 个数据集
python scripts/train_all_languages.py --base_config configs/config_rethink_v3.yaml

# 只训练指定数据集
python scripts/train_all_languages.py --base_config configs/config_rethink_v3.yaml \
    --datasets eng_restaurant jpn_hotel

# 快速测试（减少 epoch）
python scripts/train_all_languages.py --base_config configs/config_rethink_v3.yaml \
    --datasets eng_restaurant --num_epochs 3
```

**参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base_config` | `configs/config_rethink_v3.yaml` | 基础配置文件 |
| `--datasets` | 全部 8 个 | 指定要训练的数据集 |
| `--num_epochs` | config 中的值 | 覆盖训练轮数 |

**输出**:
- 权重: `checkpoints/all_lang/{dataset_name}/checkpoint_best.pt`
- 汇总: `checkpoints/all_lang/results.json`

**说明**: 脚本自动处理每个数据集的标签集和编码器选择（英文用 `deberta-v3-base`，其他语言用 `mdeberta-v3-base`）。

---

## 4. 推理（生成预测）

**脚本**: `scripts/inference.py`

```bash
# 单模型推理
python scripts/inference.py \
    --checkpoint checkpoints/all_lang/eng_restaurant/checkpoint_best.pt \
    --config configs/config_rethink_v3.yaml \
    --test_file data/subtask_3/eng/eng_restaurant_dev_task3.jsonl \
    --output predictions/eng_restaurant_pred.jsonl

# Ensemble 推理（加权平均 top-k 权重）
python scripts/inference.py \
    --ensemble \
    --checkpoint_dir checkpoints/all_lang/eng_restaurant \
    --config configs/config_rethink_v3.yaml \
    --test_file data/subtask_3/eng/eng_restaurant_dev_task3.jsonl \
    --output predictions/eng_restaurant_ensemble.jsonl

# 调整 batch_size（默认 32）
python scripts/inference.py \
    --checkpoint checkpoints/all_lang/eng_restaurant/checkpoint_best.pt \
    --config configs/config_rethink_v3.yaml \
    --test_file data/subtask_3/eng/eng_restaurant_dev_task3.jsonl \
    --output predictions/eng_restaurant_pred.jsonl \
    --batch_size 64
```

**参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | 无 | 模型权重路径（单模型模式） |
| `--ensemble` | false | 启用 ensemble 模式 |
| `--checkpoint_dir` | 无 | 权重目录（ensemble 模式必填） |
| `--config` | `configs/config.yaml` | 配置文件路径 |
| `--test_file` | **必填** | 测试数据路径 |
| `--output` | **必填** | 预测输出路径 |
| `--batch_size` | 32 | 推理 batch size |
| `--device` | auto | 设备 |

**说明**: 推理时自动从 checkpoint 加载 `entity_labels` 和 `attribute_labels`，无需担心标签不匹配。如果 checkpoint 包含 EMA 权重，会自动使用 EMA 权重。

---

## 5. 评测

**脚本**: `scripts/official_eval.py`（官方评测脚本）

```bash
python scripts/official_eval.py \
    -p predictions/eng_restaurant_pred.jsonl \
    -g data/subtask_3/eng/eng_restaurant_dev_task3.jsonl \
    -t 3
```

**参数**:

| 参数 | 说明 |
|------|------|
| `-p` / `--pred_data_path` | 预测文件路径 |
| `-g` / `--gold_data_path` | 标注文件路径 |
| `-t` / `--task` | 任务编号（固定为 3） |
| `--do_norm` | 启用文本归一化（可选） |

**输出示例**:
```
--- Evaluation Results ---
True Positives (cTP): 42.35
cPrecision: 0.6521
cRecall: 0.6089
cF1: 0.6298
```

---

## 6. 批量评测

**脚本**: `scripts/batch_eval.py`

对所有已训练的数据集依次执行推理 + 评测，输出汇总表。

```bash
# 评测全部数据集
python scripts/batch_eval.py --base_config configs/config_rethink_v3.yaml

# 评测指定数据集
python scripts/batch_eval.py --base_config configs/config_rethink_v3.yaml \
    --datasets eng_restaurant jpn_hotel

# 使用 ensemble
python scripts/batch_eval.py --base_config configs/config_rethink_v3.yaml --ensemble

# 指定权重和输出目录
python scripts/batch_eval.py --base_config configs/config_rethink_v3.yaml \
    --checkpoint_dir checkpoints/all_lang \
    --output_dir predictions
```

**参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base_config` | `configs/config_rethink_v3.yaml` | 配置文件 |
| `--checkpoint_dir` | `checkpoints/all_lang` | 权重根目录 |
| `--output_dir` | `predictions` | 预测输出目录 |
| `--datasets` | 全部 | 指定数据集 |
| `--ensemble` | false | 使用 ensemble 推理 |

**输出**: `predictions/eval_results.json`

---

## 7. Loss 权重搜索

**脚本**: `scripts/sweep_loss_weights.py`

自动搜索不同 loss 权重组合，找到最优配置。

```bash
# 运行所有预设权重组合
python scripts/sweep_loss_weights.py --config configs/config_rethink_v3.yaml

# 只运行指定组合
python scripts/sweep_loss_weights.py --config configs/config_rethink_v3.yaml \
    --groups baseline span-focus va-boost
```

**预设权重组合**: `baseline`, `span-focus`, `va-boost`, `span+va` 等

---

## 8. 常用参数说明

### 配置文件层级

```
configs/config_rethink_v3.yaml    ← 推荐使用（V3 最优配置）
configs/config_rethink_v2.yaml    ← V2（无 loss weights）
configs/config_rethink.yaml       ← V1（基础 rethink）
configs/config_baseline.yaml      ← 无 rethink
```

### 完整工作流

```bash
# Step 1: 训练全部数据集
python scripts/train_all_languages.py --base_config configs/config_rethink_v3.yaml

# Step 2: 批量推理 + 评测
python scripts/batch_eval.py --base_config configs/config_rethink_v3.yaml

# Step 3: (可选) ensemble 评测
python scripts/batch_eval.py --base_config configs/config_rethink_v3.yaml --ensemble
```

### 快速验证（开发调试用）

```bash
# 单数据集、少 epoch 快速验证
python scripts/train_all_languages.py --base_config configs/config_rethink_v3.yaml \
    --datasets eng_restaurant --num_epochs 3

# 推理 + 评测
python scripts/inference.py \
    --checkpoint checkpoints/all_lang/eng_restaurant/checkpoint_best.pt \
    --config configs/config_rethink_v3.yaml \
    --test_file data/subtask_3/eng/eng_restaurant_dev_task3.jsonl \
    --output predictions/eng_restaurant_pred.jsonl

python scripts/official_eval.py \
    -p predictions/eng_restaurant_pred.jsonl \
    -g data/subtask_3/eng/eng_restaurant_dev_task3.jsonl \
    -t 3
```
