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

| Model | cF1 | Exact F1 | VA RMSE |
|-------|-----|----------|---------|
| Baseline | 0.2935 | 0.3252 | 1.3200 |
| + Rethink | 0.4064 | 0.4490 | 1.3125 |
| + Hungarian Matching | **0.4348** | **0.4748** | **1.1406** |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
python scripts/train.py --config configs/config_rethink_v3.yaml

# Inference
python scripts/inference.py \
    --checkpoint checkpoints/rethink_v3/checkpoint_best.pt \
    --test_file data/subtask_3/eng/eng_restaurant_test_task3.jsonl \
    --output predictions.jsonl
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
- **Span Extractor**: BIO tagging for aspect/opinion extraction
- **Pair Classifier**: Entity + Attribute classification
- **V-A Regressor**: Continuous valence-arousal prediction (1-9)
- **Feature-level Rethink**: Two-stage refinement mechanism
- **Hungarian Matching**: Optimal pair alignment during training

## Documentation

- [Model Architecture](docs/MODEL.md) - 详细架构说明
- [Experiments](docs/EXPERIMENTS.md) - 实验记录与结果分析

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
│   └── inference.py
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

| Parameter | Default | Description |
|-----------|---------|-------------|
| encoder_name | deberta-v3-base | Pretrained encoder |
| use_rethink | true | Enable Rethink module |
| fgm_epsilon | 0.1 | Adversarial perturbation |
| batch_size | 8 | Training batch size |
| learning_rate | 1.5e-5 | Encoder learning rate |

## References

- [SemEval-2026 Task 3](https://github.com/DimABSA/DimABSA2026)
- [DeBERTa-v3](https://github.com/microsoft/DeBERTa)

## License

MIT
