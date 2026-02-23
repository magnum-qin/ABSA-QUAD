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

# Inference (single checkpoint, auto-loads EMA weights)
python scripts/inference.py \
    --checkpoint checkpoints/rethink_v3/checkpoint_best.pt \
    --config configs/config_rethink_v3.yaml \
    --test_file data/subtask_3/eng/eng_restaurant_test_task3.jsonl \
    --output predictions.jsonl

# Inference (ensemble: weight-average top-k checkpoints)
python scripts/inference.py \
    --ensemble --checkpoint_dir checkpoints/rethink_v3 \
    --config configs/config_rethink_v3.yaml \
    --test_file data/subtask_3/eng/eng_restaurant_test_task3.jsonl \
    --output predictions_ensemble.jsonl

# Loss weights sweep (hyperparameter search)
python scripts/sweep_loss_weights.py --config configs/config_rethink_v3.yaml
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
│   ├── inference.py             # 支持 EMA + ensemble
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

```python
from src.evaluation.metrics import evaluate_predictions
metrics = evaluate_predictions('predictions.jsonl', 'gold.jsonl')
# {'continuous_f1': 0.43, 'exact_f1': 0.47, 'va_rmse': 1.14}
```

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
| loss_weights.bio | 1.0 | BIO span loss weight |
| loss_weights.entity | 1.0 | Entity classification loss weight |
| loss_weights.attribute | 1.0 | Attribute classification loss weight |
| loss_weights.va | 1.0 | V-A regression loss weight |
| loss_weights.consistency | 0.5 | Rethink consistency loss weight |

## References

- [SemEval-2026 Task 3](https://github.com/DimABSA/DimABSA2026)
- [DeBERTa-v3](https://github.com/microsoft/DeBERTa)

## License

MIT
