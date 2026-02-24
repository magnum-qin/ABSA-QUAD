# DimABSA-QUAD

SemEval-2026 Task 3: Dimensional Aspect Sentiment Quad Prediction

## Overview

从文本中提取情感四元组 `(Aspect, Category, Opinion, Valence#Arousal)`

```
Input:  "The thai food was delicious but delivery was terrible."
Output: [(thai food, FOOD#QUALITY, delicious, 7.5#6.2),
         (delivery, SERVICE#GENERAL, terrible, 2.1#7.0)]
```

## Results

| Model | cF1 | Exact F1 | VA RMSE | Best Epoch |
|-------|-----|----------|---------|------------|
| Baseline | 0.2935 | 0.3252 | 1.3200 | 7 |
| + Rethink Module | 0.4064 | 0.4490 | 1.3125 | 18 |
| + Span Fix + FGM + EMA | 0.3836 | 0.4195 | 1.2208 | 11 |
| + Hungarian Matching | 0.4348 | 0.4748 | 1.1406 | 16 |
| + Loss Weight Sweep (span=2.0) | **0.6278** | - | - | - |

**Improvement over Baseline: +33.43% cF1**

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
python scripts/train.py --config configs/config_rethink_v3.yaml

# Train all 8 language/domain datasets
python scripts/train_all_languages.py --base_config configs/config_rethink_v3.yaml

# Batch inference + evaluation (all datasets)
python scripts/batch_eval.py --base_config configs/config_rethink_v3.yaml

# Single dataset inference
python scripts/inference.py \
    --checkpoint checkpoints/all_lang/eng_restaurant/checkpoint_best.pt \
    --config configs/config_rethink_v3.yaml \
    --test_file data/subtask_3/eng/eng_restaurant_dev_task3.jsonl \
    --output predictions/eng_restaurant_pred.jsonl

# Official evaluation
python scripts/official_eval.py \
    -p predictions/eng_restaurant_pred.jsonl \
    -g data/subtask_3/eng/eng_restaurant_dev_task3.jsonl \
    -t 3
```

## Architecture

```
Input → DeBERTa-v3 → [Span BIO, Pair Classifier, V-A Regressor]
                              ↓
                    Feature-level Rethink
                              ↓
                    Refined Predictions
```

**Key Components:**
- **Span Extractor**: BIO tagging for aspect/opinion extraction (with position-aware duplicate handling)
- **Pair Classifier**: Entity + Attribute classification with label smoothing (0.1)
- **V-A Regressor**: Continuous valence-arousal prediction (1-9)
- **Feature-level Rethink**: Iterative refinement mechanism (configurable iterations)
- **Hungarian Matching**: Optimal pair alignment during training (+5.12% cF1 vs greedy)
- **FGM**: Fast Gradient Method adversarial training, warmup-guarded to avoid early instability
- **EMA**: Exponential Moving Average of model weights (decay=0.999)
- **Loss Weights**: Per-task loss weighting (bio/entity/attribute/va/consistency) via config
- **Top-K Checkpointing**: Min-heap based top-k checkpoint management for robust model selection

## Documentation

- [Model Architecture](docs/MODEL.md) - 详细架构说明
- [Experiments](docs/EXPERIMENTS.md) - 实验记录与结果分析
- [CLI Guide](docs/CLI_GUIDE.md) - 命令行操作指南（训练、推理、评测）
- [AI Guide](README_AI.md) - AI 快速理解指南（架构、已实现功能、创新方向）

## Project Structure

```
├── configs/              # 训练配置
│   ├── config_baseline.yaml
│   ├── config_rethink.yaml
│   ├── config_rethink_v2.yaml
│   └── config_rethink_v3.yaml
├── src/
│   ├── models/
│   │   ├── dimabsa_model.py    # 主模型
│   │   ├── va_module.py        # V-A回归模块
│   │   └── feature_rethink.py  # Rethink模块
│   ├── data/
│   │   └── dataset.py          # 数据处理
│   └── evaluation/
│       └── metrics.py          # 评估指标
├── scripts/
│   ├── train.py
│   ├── train_all_languages.py   # 全语言批量训练
│   ├── inference.py             # 支持 EMA + ensemble
│   ├── batch_eval.py            # 批量推理+评测
│   ├── official_eval.py         # 官方评测脚本
│   └── sweep_loss_weights.py    # loss权重自动搜索
├── docs/
│   ├── MODEL.md
│   └── EXPERIMENTS.md
└── checkpoints/
```

## Data Format

**Input (JSONL):**
```json
{"ID": "R001", "Text": "The food was great."}
```

**Output (JSONL):**
```json
{
  "ID": "R001",
  "Quadruplet": [
    {"Aspect": "food", "Category": "FOOD#QUALITY", "Opinion": "great", "VA": "7.50#6.00"}
  ]
}
```

## Evaluation

### Step 1: Train (All Languages)

```bash
# Train all 8 datasets (eng/rus/tat/ukr/zho_restaurant, eng/zho_laptop, jpn_hotel)
python scripts/train_all_languages.py --base_config configs/config_rethink_v3.yaml

# Train specific datasets only
python scripts/train_all_languages.py --base_config configs/config_rethink_v3.yaml \
    --datasets eng_restaurant jpn_hotel

# Quick test (fewer epochs)
python scripts/train_all_languages.py --base_config configs/config_rethink_v3.yaml \
    --datasets eng_restaurant --num_epochs 3
```

Checkpoints are saved to `checkpoints/all_lang/{dataset_name}/`.

### Step 2: Generate Predictions

```bash
# Single dataset
python scripts/inference.py \
    --config configs/config_rethink_v3.yaml \
    --checkpoint checkpoints/all_lang/eng_restaurant/checkpoint_best.pt \
    --test_file data/subtask_3/eng/eng_restaurant_dev_task3.jsonl \
    --output predictions/eng_restaurant_pred.jsonl

# Ensemble (weight-average top-k checkpoints)
python scripts/inference.py \
    --ensemble --checkpoint_dir checkpoints/all_lang/eng_restaurant \
    --config configs/config_rethink_v3.yaml \
    --test_file data/subtask_3/eng/eng_restaurant_dev_task3.jsonl \
    --output predictions/eng_restaurant_ensemble.jsonl
```

### Step 3: Official Evaluation

Uses the [official DimABSA2026 evaluation script](https://github.com/DimABSA/DimABSA2026/tree/main/evaluation_script):

```bash
python scripts/official_eval.py \
    -p predictions/eng_restaurant_pred.jsonl \
    -g data/subtask_3/eng/eng_restaurant_dev_task3.jsonl \
    -t 3
```

Output:
```
--- Evaluation Results ---
True Positives (cTP): ...
cPrecision: ...
cRecall: ...
cF1: ...
```

### Batch Evaluation (All Languages)

```bash
# All datasets
python scripts/batch_eval.py --base_config configs/config_rethink_v3.yaml

# Specific datasets
python scripts/batch_eval.py --base_config configs/config_rethink_v3.yaml \
    --datasets eng_restaurant jpn_hotel

# With ensemble
python scripts/batch_eval.py --base_config configs/config_rethink_v3.yaml --ensemble
```

Results are saved to `predictions/eval_results.json`.

**Note:** Inference automatically loads `entity_labels` and `attribute_labels` from the checkpoint's saved config, so label order mismatch is handled automatically.

## Configuration

| Parameter | Default (V3) | Description |
|-----------|---------|-------------|
| encoder_name | microsoft/deberta-v3-base | Pretrained encoder |
| use_rethink | true | Enable Feature-level Rethink |
| rethink_iterations | 2 | Number of iterative rethink passes |
| batch_size | 8 | Training batch size |
| encoder_lr | 1.5e-5 | Encoder learning rate |
| head_lr | 7.5e-5 | Task head learning rate |
| num_epochs | 30 | Max training epochs |
| patience | 8 | Early stopping patience |
| fgm_epsilon | 0.1 | Adversarial perturbation strength |
| ema_decay | 0.999 | EMA decay for evaluation |
| save_top_k | 3 | Number of best checkpoints to retain |
| loss_weights.span | 2.0 | BIO span loss weight |
| loss_weights.entity | 1.0 | Entity classification loss weight |
| loss_weights.attribute | 1.0 | Attribute classification loss weight |
| loss_weights.va | 1.0 | V-A regression loss weight |
| loss_weights.consistency | 0.5 | Rethink consistency loss weight |

## References

- [SemEval-2026 Task 3](https://github.com/DimABSA/DimABSA2026)
- [DeBERTa-v3](https://github.com/microsoft/DeBERTa)

## License

MIT
