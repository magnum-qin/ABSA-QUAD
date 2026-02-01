# DimASQP: Dimensional Aspect Sentiment Quad Prediction

🏆 **SemEval-2026 Task 3 Implementation**

A hybrid deep learning framework for Dimensional Aspect-Based Sentiment Analysis with Feature-level Rethink mechanism.

## 📋 Overview

This project implements a state-of-the-art model for **DimASQP** (Dimensional Aspect Sentiment Quad Prediction) task, which extracts quadruplets containing:
- **Aspect**: Opinion target (e.g., "thai food")
- **Category**: Entity#Attribute format (e.g., "FOOD#QUALITY")
- **Opinion**: Sentiment expression (e.g., "delicious")
- **V-A Score**: Valence-Arousal in [1,9] range (e.g., "7.50#6.25")

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Text                                │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              DeBERTa-v3 / BERT Encoder                      │
│                  Hidden States [B, L, 768]                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   ┌─────────┐      ┌─────────┐      ┌─────────┐
   │  Span   │      │Category │      │   V-A   │
   │  BIO    │      │  Class  │      │ Regress │
   └────┬────┘      └────┬────┘      └────┬────┘
        │                │                │
        └────────────────┼────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature-level Rethink Module                   │
│   • Label Embeddings from Stage 1 predictions               │
│   • Gated fusion with original hidden states                │
│   • Self-attention refinement                               │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
              Stage 2: Refined Predictions
                          │
                          ▼
            (Aspect, Category, Opinion, V#A)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/dimabsa-project.git
cd dimabsa-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

Download the official dataset from [SemEval-2026 Task 3](https://github.com/DimABSA/DimABSA2026):

```bash
mkdir -p data
# Place train.jsonl, dev.jsonl, test.jsonl in data/
```

### 3. Training

```bash
# Train with default config
python scripts/train.py --config configs/config.yaml

# Train with custom settings
python scripts/train.py \
    --config configs/config.yaml \
    --train_file data/train.jsonl \
    --val_file data/dev.jsonl \
    --device cuda
```

### 4. Inference

```bash
# Generate predictions
python scripts/inference.py \
    --checkpoint checkpoints/checkpoint_best.pt \
    --test_file data/test.jsonl \
    --output predictions.jsonl
```

### 5. Evaluation

```bash
# Evaluate predictions
python -c "
from src.evaluation.metrics import evaluate_predictions
metrics = evaluate_predictions('predictions.jsonl', 'data/dev.jsonl')
print(metrics)
"
```

## 📁 Project Structure

```
dimabsa-project/
├── configs/
│   └── config.yaml          # Model and training configuration
├── src/
│   ├── models/
│   │   ├── dimabsa_model.py # Main model architecture
│   │   ├── va_module.py     # V-A regression module
│   │   └── feature_rethink.py # Feature-level Rethink module
│   ├── data/
│   │   └── dataset.py       # Data loading and preprocessing
│   ├── evaluation/
│   │   └── metrics.py       # Continuous F1 and other metrics
│   ├── training/
│   └── utils/
├── scripts/
│   ├── train.py             # Training script
│   └── inference.py         # Inference script
├── notebooks/
├── requirements.txt
└── README.md
```

## 📊 Key Features

### 1. V-A Continuous Regression
- Predicts Valence and Arousal in [1, 9] range
- Uses Huber Loss for robust regression
- Separate heads with shared feature extraction

### 2. Feature-level Rethink
- Converts Stage 1 predictions to learnable embeddings
- Fuses with original hidden states via gating mechanism
- Stage 2 predictions are refined and more accurate

### 3. Continuous F1 Metric
- Official SemEval-2026 evaluation metric
- Penalizes V-A prediction errors proportionally
- Combines categorical matching with regression quality

## 📈 Expected Results

| Metric | Restaurant | Laptop |
|--------|------------|--------|
| cF1    | ~40%       | ~35%   |
| Exact F1 | ~45%     | ~40%   |
| VA RMSE | ~1.2      | ~1.3   |

## 🔧 Configuration

Key configuration options in `configs/config.yaml`:

```yaml
model:
  encoder_name: "microsoft/deberta-v3-base"
  use_rethink: true
  va_min: 1.0
  va_max: 9.0

training:
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 20
```

## 📝 Data Format

### Input (JSONL)
```json
{
    "ID": "R001",
    "Text": "average to good thai food, but terrible delivery."
}
```

### Output (JSONL)
```json
{
    "ID": "R001",
    "Quadruplet": [
        {
            "Aspect": "thai food",
            "Category": "FOOD#QUALITY",
            "Opinion": "average to good",
            "VA": "6.75#6.38"
        },
        {
            "Aspect": "delivery",
            "Category": "SERVICE#GENERAL",
            "Opinion": "terrible",
            "VA": "2.88#6.62"
        }
    ]
}
```

## 🎯 V-A Score Interpretation

| Valence | Meaning | Example |
|---------|---------|---------|
| 1.0 | Extremely negative | "horrible", "terrible" |
| 5.0 | Neutral | "okay", "average" |
| 9.0 | Extremely positive | "amazing", "perfect" |

| Arousal | Meaning | Example |
|---------|---------|---------|
| 1.0 | Very calm/low intensity | "fine", "acceptable" |
| 5.0 | Medium intensity | "good", "nice" |
| 9.0 | Very excited/high intensity | "incredible!", "absolutely love" |

## 📚 References

1. SemEval-2026 Task 3: [DimABSA GitHub](https://github.com/DimABSA/DimABSA2026)
2. SIGHAN-2024 dimABSA: First Dimensional ABSA shared task
3. DeBERTa-v3: [Microsoft Research](https://github.com/microsoft/DeBERTa)

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- Your Name
- Your Institution

## 🙏 Acknowledgments

- SemEval-2026 Task 3 Organizers
- SIGHAN-2024 dimABSA Team
- Hugging Face Transformers Team
