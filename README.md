# DimABSA-QUAD

SemEval-2026 Task 3: Dimensional Aspect Sentiment Quad Prediction

从文本中提取情感四元组 `(Aspect, Category, Opinion, Valence#Arousal)`，覆盖 8 个语言/领域数据集。

```
Input:  "The thai food was delicious but delivery was terrible."
Output: [(thai food, FOOD#QUALITY, delicious, 7.5#6.2),
         (delivery, SERVICE#GENERAL, terrible, 2.1#7.0)]
```

## 方法概览

每个数据集使用**专用预训练 encoder + 5-seed 双层 ensemble**，部分数据稀缺的数据集额外用 **LLM 知识蒸馏 (KD)** 突破。

- **专用 encoder**：按语言/领域选最优 encoder（见下表），而非单模型跑全部。
- **双层 ensemble**：每个 seed 内 `save_top_k=3` 权重平均 → 跨 5 seed `vote_threshold` + `span_topk_threshold` 投票。
- **LLM 知识蒸馏 (V23)**：LLM（云端 DeepSeek 或本地 qwen3.6）对 in-domain 无标注文本打 silver 标签 → 小模型在 gold+silver 上训练 → **推理时完全不用 LLM**。详见 [§5.28–5.29](docs/EXPERIMENTS.md)。

## 当前结果（dev cF1，8 数据集提交配置）

| 数据集 | encoder | 方法 | dev cF1 | test SOTA 对比 |
|---|---|---|---|---|
| eng_laptop | deberta-v3-base | **KD (DeepSeek)** | **0.4141** | ✅ 反超 nchellwig 0.4006 |
| eng_restaurant | deberta-v3-large | V16 baseline | 0.6554 | ✅ 领先 |
| jpn_hotel | ku-nlp/deberta-v2-large-japanese | V16 baseline | 0.4466 | ✅ 反超 PALI 0.4252 |
| rus_restaurant | xlm-roberta-large | **KD (DeepSeek)** | **0.5753** | ✅ 反超 PAI 0.5599 |
| ukr_restaurant | mdeberta-v3-base | **KD (DeepSeek)** | **0.5560** | ✅ 反超 PAI 0.5437 |
| tat_restaurant | xlm-roberta-large | **KD (DeepSeek)** | **0.5196** | ✅ 反超 Takoyaki 0.4736 |
| zho_restaurant | chinese-roberta-wwm-ext-large | **KD (qwen3.6)** | **0.5263** | ✅ 领先（基线 0.4745）|
| zho_laptop | chinese-roberta-wwm-ext-large | V16 baseline | 0.4487 | ❌ 距 NYCU 0.4824 −3.4pp |

**5/8 数据集采用 KD**（4 个 DeepSeek + 1 个本地 qwen3.6）。注意 dev≠test，最终以 test 集为准。

### KD 何时有效（核心发现）

KD 有效 ⟺ **silver 能提供 gold 缺失的信息**：
- gold 弱 / 类别空间极稀疏 → silver 补充稀疏类先验（eng_laptop +8.9pp、rus/ukr/tat）；
- 或 silver 保持少数派（silver:gold < 1）→ 只加 seed 多样性不淹没 gold（zho_restaurant +5.2pp）。

反例：强 gold（eng_restaurant 0.655）被 2× 噪声 silver 碾压 → −25pp；强 native encoder（jpn）→ 净负面。完整消融见 [docs/EXPERIMENTS.md §5.29.1–5.29.3](docs/EXPERIMENTS.md)。

## Setup

```bash
# 安装 Miniforge（如未安装）：https://github.com/conda-forge/miniforge
bash scripts/setup_env.sh
mamba activate absa
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
```

## Quick Start

当前工作流以 **`train_multi_seed.py` + per-dataset pipeline config + 多 seed ensemble 推理**为准。

```bash
# 1) 5-seed 训练（pipeline config 指定 encoder / 数据 / 超参）
python scripts/train_multi_seed.py \
    --config configs/pipeline_<dataset>_v16.yaml \
    --seeds 42 123 456 789 1024 \
    --save_prefix <dataset>_seed \
    --train_file data/subtask_3/<lang>/<dataset>_train_alltasks.jsonl

# 2) 多 seed 双层 ensemble 推理（vote + topk）
python scripts/inference.py --multi_seed_ensemble \
    --seed_dirs checkpoints/<dataset>_seed42 ... checkpoints/<dataset>_seed1024 \
    --config configs/pipeline_<dataset>_v16.yaml \
    --test_file data/subtask_3/<lang>/<dataset>_dev_task3.jsonl \
    --output predictions/<dataset>_pred.jsonl \
    --vote_threshold 3 --span_topk_threshold 0.0 --batch_size 16

# 3) 官方评测
python scripts/official_eval.py -p predictions/<dataset>_pred.jsonl \
    -g data/subtask_3/<lang>/<dataset>_dev_task3.jsonl -t 3
```

### 知识蒸馏 (KD) 流程

```bash
# A) silver-labeling（云端 DeepSeek）
export DEEPSEEK_API_KEY=sk-...
python scripts/llm_silver_label.py --input <pool>.jsonl \
    --train_file <dataset>_train_alltasks.jsonl --output <dataset>_silver.jsonl \
    --model deepseek-v4-pro --domain restaurant --n_shots 6

# A') silver-labeling（本地 ollama，如 qwen3.6）— 自动关闭推理 channel (think:false)
OLLAMA_URL=http://localhost:11433 python scripts/llm_silver_label_ollama.py \
    --input <pool>.jsonl --train_file <dataset>_train_alltasks.jsonl \
    --output <dataset>_silver_qwen.jsonl --model "qwen3.6:latest" --domain restaurant

# B) 无 in-domain 语料时，本地 LLM 原生生成池（中/日/英等）
OLLAMA_URL=http://localhost:11433 python scripts/gen_native_ollama.py \
    --dataset zho_restaurant --n 4000 --output <pool>.jsonl --model "qwen3.6:latest"

# C) distill 训练 = gold + 校验后的 silver → 同上 train_multi_seed + ensemble
```

完整可复现脚本：`scripts/run_v23_<dataset>_deepseek.sh`（DeepSeek 臂）、`scripts/run_v23_<dataset>_qwen.sh` / `scripts/run_v23_multi_qwen.sh`（本地 qwen 臂）。命名规则见 [docs/NAMING.md](docs/NAMING.md)。

## Architecture

```
Input → 专用 Encoder → [Span BIO, Pair Classifier, V-A Regressor]
                              ↓
                    Feature-level Rethink (2-stage)
                              ↓
                    Refined Predictions
```

- **Span Extractor**：BIO 标注抽取 aspect/opinion（位置感知去重）
- **Pair Classifier**：Entity + Attribute 分类（label smoothing 0.1）
- **V-A Regressor**：连续 valence-arousal 回归 (1–9)
- **Feature-level Rethink**：迭代精化（可配置轮数）
- **Hungarian Matching**：训练时最优配对对齐
- **FGM / EMA**：对抗训练（warmup 保护）+ 权重 EMA (decay 0.999)
- **双层 Ensemble**：seed 内 top-k 权重平均 + 跨 seed vote/topk 投票

详见 [docs/MODEL.md](docs/MODEL.md)。

## Project Structure

```
├── configs/                     # pipeline_<dataset>_v*.yaml（per-dataset encoder/数据/超参）
├── src/
│   ├── models/
│   │   ├── dimabsa_model.py      # 主模型
│   │   ├── va_module.py          # V-A 回归
│   │   ├── feature_rethink.py    # Rethink 模块
│   │   └── crf.py                # （可选）CRF
│   ├── data/dataset.py           # 数据处理（Quadruplet → BIO/pair/VA）
│   └── evaluation/metrics.py     # 评估指标
├── scripts/
│   ├── train_multi_seed.py       # 多 seed 训练（当前主入口）
│   ├── inference.py              # 推理（--multi_seed_ensemble 双层投票）
│   ├── official_eval.py          # 官方评测
│   ├── llm_silver_label.py       # KD silver-labeling（DeepSeek 云端）
│   ├── llm_silver_label_ollama.py# KD silver-labeling（本地 ollama，think:false）
│   ├── gen_native_ollama.py      # 本地 LLM 原生生成 in-domain 池（中/日/英…）
│   ├── build_v23_*.py            # 组装 distill 训练集
│   └── run_v23_<dataset>_<teacher>.sh  # 各数据集 KD 全流程（命名见 docs/NAMING.md）
├── data/subtask_3/<lang>/        # 8 数据集 train/dev/test + silver/pool
├── submission/final/             # 各数据集 test 预测（*_backup.jsonl = 切换前版本）
├── submission/submission.zip     # 正式提交包
├── docs/EXPERIMENTS.md           # 完整实验记录（V4→V23，含 KD 消融）
└── checkpoints/                  # 模型权重（save_top_k=3 × 5 seed）
```

## Data Format

**Input (JSONL):** `{"ID": "R001", "Text": "The food was great."}`

**Output (JSONL):**
```json
{"ID": "R001", "Quadruplet": [
  {"Aspect": "food", "Category": "FOOD#QUALITY", "Opinion": "great", "VA": "7.50#6.00"}
]}
```

## Documentation

- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — 完整实验记录与结果分析（含 KD 消融 §5.28–5.29，**最权威最新**）
- [docs/MODEL.md](docs/MODEL.md) — 模型架构详解
- [docs/CLI_GUIDE.md](docs/CLI_GUIDE.md) — 命令行操作指南
- [README_AI.md](README_AI.md) — AI 快速理解指南

## Key Configuration

| Parameter | 默认 | 说明 |
|-----------|------|------|
| encoder_name | per-dataset | 专用 encoder（见结果表）|
| rethink.enabled / num_iterations | true / 2 | Feature-level Rethink |
| batch_size | 4–8 | 训练 batch（large encoder 用 4）|
| encoder_lr / head_lr | 1.5e-5 / 7.5e-5 | 分层学习率 |
| num_epochs / patience | 30 / 8 | 最大 epoch / 早停 |
| fgm.epsilon / ema.decay | 0.1 / 0.999 | 对抗扰动 / EMA |
| save_top_k | 3 | seed 内保留的最优 checkpoint 数 |
| loss_weights.span | 2.0 | BIO span loss 权重 |
| vote_threshold / span_topk_threshold | 推理时扫 | 跨 seed 投票 / span top-k 阈值 |

## References

- [SemEval-2026 Task 3 (DimABSA2026)](https://github.com/DimABSA/DimABSA2026)
- 官方评测脚本：[evaluation_script](https://github.com/DimABSA/DimABSA2026/tree/main/evaluation_script)

## License

MIT
