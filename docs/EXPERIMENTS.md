# 实验记录与结果分析

本文档记录 DimABSA-QUAD 项目的实验配置、结果对比和消融研究。

## 目录

1. [实验环境](#1-实验环境)
2. [数据集](#2-数据集)
3. [评估指标](#3-评估指标)
4. [配置版本演进](#4-配置版本演进) (Baseline → V1 → V2 → V3 → V3-CRF(失败) → V4 Ensemble → V5 增强Ensemble)
5. [V5 ~ V10 规划与结果](#5-v4-ensemble--v5-规划) (含 V6 迁移、V7 多源、V8 全合成、V9-V10 paraphrase、COMET 过滤)
6. [主要结果](#6-主要结果)
7. [消融实验](#7-消融实验)
8. [错误分析](#8-错误分析)

### V6 → V12 一览（jpn_hotel dev cF1）

| 版本 | 方法 | 训练样本 | cF1 | Δ vs V6 |
|------|------|---------|------|---------|
| V6 | zho transfer + MeCab + vote=2 | 1600 | 0.3312 | baseline |
| V7 | multi-source transfer (eng+zho) | 1600 | 0.3279 | −0.33% |
| V8 | +199 全合成（文本+标签）| 1799 | 0.3202 | −1.10% |
| V9 | +2699 paraphrase（tongyi）| 4299 | 0.3227 | −0.85% |
| V10 | +2957 paraphrase（DeepSeek 全量）| 4557 | 0.3299 | −0.13% |
| V11 | V10 + COMET≥0.92（轻度过滤）| 4302 | 0.3317 | +0.05% |
| **V12** ⭐⭐ | **V10 + COMET≥0.95（中度过滤）** | **3466** | **0.3523** | **+2.11%** ✨ |

**核心结论**：
1. **数据 quality > quantity**：V12 训练集比 V10 少 24%，cF1 反而高 +2.24%
2. **更激进的过滤=更好的 ensemble**：V12 单 seed 平均 0.2959（比 V11 低 1pp），但 ensemble 0.3523（比 V11 高 +2.06pp）。COMET 过滤强化了 seed 间多样性，ensemble 投票时误差互消更显著
3. **COMET 在更高阈值下才显出威力**：0.92 仅切 8.6%（V11 +0.05%），0.95 切 36.9%（V12 +2.11%）—— 阈值"剂量-效应"非线性

---

## 1. 实验环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 3090 / 4090 (24GB) |
| Python | 3.10+ |
| PyTorch | 2.0+ |
| Transformers | 4.30+ |
| 训练时间 | ~30-40 分钟/30 epochs |

---

## 2. 数据集

### 2.1 SemEval-2026 Task 3 数据

| 数据集 | 样本数 | 四元组数 | 平均四元组/样本 |
|--------|--------|----------|-----------------|
| Restaurant Train | 2,284 | ~4,500 | ~2.0 |
| Restaurant Dev | 200 | ~400 | ~2.0 |
| Restaurant Test | 1,000 | - | - |
| Laptop Train | - | - | - |
| Laptop Dev | - | - | - |
| Laptop Test | 1,000 | - | - |

### 2.2 多语言数据

| 语言 | 领域 | Entity 数 | Attribute 数 | Encoder |
|------|------|-----------|-------------|---------|
| English (eng) | Restaurant | 6+NULL | 5+NULL | deberta-v3-base |
| English (eng) | Laptop | 23+NULL | 9+NULL | deberta-v3-base |
| Japanese (jpn) | Hotel | 7+NULL | 8+NULL | mdeberta-v3-base |
| Russian (rus) | Restaurant | 6+NULL | 5+NULL | mdeberta-v3-base |
| Tatar (tat) | Restaurant | 6+NULL | 5+NULL | mdeberta-v3-base |
| Ukrainian (ukr) | Restaurant | 6+NULL | 5+NULL | mdeberta-v3-base |
| Chinese (zho) | Restaurant | 6+NULL | 5+NULL | mdeberta-v3-base |
| Chinese (zho) | Laptop | 22+NULL | 8+NULL | mdeberta-v3-base |

训练命令: `python scripts/train_all_languages.py --base_config configs/config_rethink_v3.yaml`

评测命令: `python scripts/batch_eval.py --base_config configs/config_rethink_v3.yaml`

---

## 3. 评估指标

### 3.1 主指标: Continuous F1 (cF1)

SemEval-2026 官方主指标，结合分类准确性和 V-A 回归质量：

```
cTP = 1 - dist(VA_pred, VA_gold) / D_max

其中:
- dist = sqrt((V_p - V_g)² + (A_p - A_g)²)  # 欧氏距离
- D_max = sqrt(2 × (9-1)²) ≈ 11.31         # 最大可能距离

cRecall = Σ(cTP) / (TP_categorical + FN_categorical)
cPrecision = Σ(cTP) / (TP_categorical + FP_categorical)
cF1 = 2 × cRecall × cPrecision / (cRecall + cPrecision)
```

**匹配规则**: 两个四元组"分类匹配"当且仅当：
- Aspect 文本相同 (忽略大小写)
- Category 相同
- Opinion 文本相同 (忽略大小写)

### 3.2 辅助指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| Exact F1 | 忽略 V-A 的精确匹配 F1 | 标准 F1 公式 |
| VA RMSE | V-A 预测的均方根误差 | sqrt(mean((V_p-V_g)² + (A_p-A_g)²)) |
| V RMSE | Valence 单独的 RMSE | sqrt(mean((V_p-V_g)²)) |
| A RMSE | Arousal 单独的 RMSE | sqrt(mean((A_p-A_g)²)) |

---

## 4. 配置版本演进

### 4.1 Baseline

**配置文件**: `configs/config_baseline.yaml`

```yaml
model:
  rethink:
    enabled: false        # 不使用 Rethink
training:
  learning_rate: 2e-5
  num_epochs: 20
  fgm:
    enabled: false        # 不使用对抗训练
```

### 4.2 Rethink V1

**配置文件**: `configs/config_rethink.yaml`

```yaml
model:
  rethink:
    enabled: true         # 启用 Rethink
    num_iterations: 2
training:
  learning_rate: 2e-5
  num_epochs: 20
```

**改进**: 启用 Feature-level Rethink 模块

### 4.3 Rethink V2

**配置文件**: `configs/config_rethink_v2.yaml`

```yaml
model:
  rethink:
    enabled: true
training:
  learning_rate: 1e-5     # 降低学习率
  head_lr: 5e-5
  num_epochs: 30          # 增加训练轮数
  patience: 8             # 增加耐心值
  fgm:
    enabled: true         # 启用 FGM
    epsilon: 0.5          # 较强扰动
```

**改进**: 降低学习率 + 强 FGM 对抗训练

### 4.4 Rethink V3 (Current Best)

**配置文件**: `configs/config_rethink_v3.yaml`

```yaml
model:
  rethink:
    enabled: true
    num_iterations: 2
training:
  encoder_lr: 1.5e-5
  head_lr: 7.5e-5
  num_epochs: 30
  fgm:
    enabled: true
    epsilon: 0.1          # 温和扰动
  ema:
    enabled: true
    decay: 0.999          # EMA 平滑
  loss_weights:
    span: 2.0             # Span 损失加权 (sweep 得到)
    pair: 1.0
    entity: 1.0
    attribute: 1.0
    va: 1.0
    consistency: 0.5
  save_top_k: 3           # 保留最佳 3 个 checkpoint
```

**V3 完整改进列表**:
1. 温和 FGM (epsilon=0.1) + warmup guard
2. EMA (decay=0.999) 稳定评估
3. Label Smoothing (0.1) 防过拟合
4. Span 位置修复 (字符级位置追踪)
5. Hungarian 匹配 (最优 pred↔gold 对齐)
6. 损失权重 sweep (span=2.0, +4.25% cF1)
7. Top-K checkpointing + Weight Averaging Ensemble
8. 多语言支持 (deberta-v3-base / mdeberta-v3-base)
9. 标签参数化 (从 config 传入，支持不同领域标签集)

### 4.5 Rethink V4 (CRF + NULL Anchor — 已回退)

**配置文件**: `configs/config_rethink_v4_ablation.yaml`

```yaml
model:
  use_crf: true             # 新增 Linear-Chain CRF 解码
  rethink:
    enabled: true
    num_iterations: 2
training:
  encoder_lr: 1.5e-5
  head_lr: 7.5e-5
  num_epochs: 30
  loss_weights:
    span: 2.0
    pair: 1.0
    entity: 1.0
    attribute: 1.0
    va: 1.5               # 提高 VA 权重
    consistency: 0.5
```

**V3-CRF 改动**（实验性，已回退）:
1. 新增 `src/models/crf.py`，Linear-Chain CRF 替代 Softmax BIO 解码
2. CRF 硬约束转移矩阵（禁止 O→I 等非法转移）
3. NULL Anchor token 处理隐式 aspect/opinion
4. 数据预处理支持 NULL span 的 BIO 标注

**V3-CRF 实验结果**: 所有优化均未超越 V3 基线

| # | 实验 | 配置 | cF1 | vs V3 | 结论 |
|---|------|------|-----|-------|------|
| 核心 | CRF + NULL Anchor | use_crf=true | 0.6249 | -0.3% | 失败，CRF 额外参数增加过拟合 |
| #1 | CRF Loss Clamp | clamp(max=20.0) | 0.6177 | -1.0% | 失败，抑制有效梯度信号 |
| #3 | Loss Weight Sweep | 6 组 (span/va) | 最高 0.6271 | -0.1% | 确认 span=2.0, va=1.5 最优 |
| #7 | VA Smooth L1 | F.huber_loss | — | — | 已有，无需改动 |
| #9 | DeBERTa-v3-large | 304M params | 0.5539 | -7.4% | 失败，2284 条数据量不足 |
| #11 | Pair Threshold | 0.05~0.50 sweep | 均 0.6264 | 0% | 确认 0.3 最优，输出高度极化 |
| #12 | Multi-Seed Ensemble | 5-seed 投票 | **0.6415** | **+1.4%** | ✅ 成功，cP=0.689, cR=0.600 |

**Loss Weight Sweep 详细结果**:

| Group | span | va | cF1 |
|-------|------|----|-----|
| **s2.0-v1.5 (原始)** | 2.0 | 1.5 | **0.6312** |
| s2.5-v2.0 | 2.5 | 2.0 | 0.6271 |
| s2.5-v1.5 | 2.5 | 1.5 | 0.6262 |
| s2.0-v2.0 | 2.0 | 2.0 | 0.6150 |
| s3.0-v1.5 | 3.0 | 1.5 | 0.6143 |
| s2.0-v1.0 | 2.0 | 1.0 | 0.6068 |

**V3-CRF 核心结论**:
- CRF 在小数据集（2284 条）上无效，BIO 标注本身简单，CRF 引入的转移矩阵参数增加过拟合
- DeBERTa-v3-large 严重过拟合，304M 参数需要远超 2284 条的数据量
- 超参已充分搜索，V3 配置已是当前数据量下的最优
- **最终决策：回退到 V3 架构**，在推理策略（Ensemble）和数据层面寻求突破

---

## 5. V4 Ensemble & V5 规划

### 5.1 V4: Multi-Seed Ensemble（✅ 已完成）

- 5 个不同随机种子（42, 123, 456, 789, 1024）训练同一 V3 架构
- 推理时多数投票（≥3/5 模型一致才保留）+ VA 均值
- **实际 cF1：0.6415**（cPrecision=0.6894，cRecall=0.5998），**+1.4% vs V3 单模型**

**各 seed 单模型结果：**

| Seed | 最佳 cF1 | Best Epoch |
|------|---------|-----------|
| 42   | 0.6232  | 30        |
| 123  | 0.6292  | 21        |
| 456  | 0.6234  | 30        |
| 789  | 0.6286  | 23        |
| 1024 | 0.6319  | 29        |
| **平均** | **0.6273** | — |
| **Ensemble (vote≥3)** | **0.6415** | — |

```bash
# 训练
python scripts/train_multi_seed.py --config configs/config_rethink_v3.yaml

# 推理
python scripts/inference.py --multi_seed_ensemble \
  --seed_dirs checkpoints/ensemble_seed42 checkpoints/ensemble_seed123 \
    checkpoints/ensemble_seed456 checkpoints/ensemble_seed789 \
    checkpoints/ensemble_seed1024 \
  --config configs/config_rethink_v3.yaml \
  --test_file data/subtask_3/eng/eng_restaurant_dev_task3.jsonl \
  --output predictions/ensemble_pred.jsonl
```

### 5.2 V5: 回译增强 + Multi-Seed Ensemble（✅ 已完成）

基于 SIGHAN 2024 冠军方案（HITSZ-HLT）和 CCIIPLab 第 2 名方案分析，V5 聚焦：**数据 > 表征 > 融合**。

#### Phase 1: 数据增强（✅ 已完成）

**D1 回译增强**（英→de/fr/zh→英）：
- 工具：HuggingFace MarianMT（Helsinki-NLP/opus-mt）
- 脚本：`scripts/augment_data.py`
- span 重定位：精确匹配 + fuzzy 匹配（SequenceMatcher ratio ≥ 0.8）

| 语言 | 保留率 | 新增样本 |
|------|--------|---------|
| DE (德语) | 82.0% | 1,874 |
| FR (法语) | 76.0% | 1,735 |
| ZH (中文) | 53.1% | 1,213 |
| **合计** | — | **4,822** |
| **原始 + 增强** | — | **7,106 (3.11x)** |

**V5 各 seed 训练结果**（增强数据 7106 条）：

| Seed | 最佳 cF1 | Best Epoch |
|------|---------|-----------|
| 42   | 0.6474  | 28        |
| 123  | 0.6417  | 24        |
| 456  | 0.6410  | 15        |
| 789  | 0.6356  | 26        |
| 1024 | 0.6332  | 21        |
| **平均** | **0.6398** | — |
| **Ensemble (vote≥3)** | **0.6530** | — |
| **双层 Ensemble (vote≥3)** | **0.6554** | — |

**推理策略消融（增强数据 5-seed）：**

| 策略 | Vote | cF1 | cPrecision | cRecall |
|------|------|-----|-----------|---------|
| Standard | 2 | 0.6488 | 0.6536 | 0.6440 |
| Standard | 3 | 0.6530 | 0.7049 | 0.6082 |
| Standard | 4 | 0.6488 | 0.7569 | 0.5677 |
| Dual-layer | 2 | 0.6401 | 0.6386 | 0.6417 |
| **Dual-layer** | **3** | **0.6554** | **0.7164** | **0.6040** |
| Dual-layer | 4 | 0.6494 | 0.7549 | 0.5698 |

双层 Ensemble = 每个 seed 内 top-3 checkpoint 权重平均 + 跨 seed 多数投票，cF1 **0.6554**，**+4.4% vs V3 单模型**

**4 语言回译增强**（已生成，待重训）：

| 语言 | 保留率 | 新增样本 |
|------|--------|---------|
| DE (德语) | 82.0% | 1,874 |
| FR (法语) | 76.0% | 1,735 |
| ZH (中文) | 53.1% | 1,213 |
| RU (俄语) | 62.8% | 1,434 |
| **合计** | — | **6,256** |
| **原始 + 增强** | — | **8,540 (3.74x)** |

**V5.1: 多 Pivot 增强（非英语语言）：**

针对非英语语言增强倍率低的问题，新增同语系 pivot 语言对：
- 俄语 (ru): 新增 `uk` pivot（乌克兰语），使用 `opus-mt-ru-uk` / `opus-mt-uk-ru`
- 乌克兰语 (uk): 新增 `ru` pivot（俄语），使用 `opus-mt-uk-ru` / `opus-mt-ru-uk`

预期效果：
- ukr_restaurant: 1.3x → ~2.5x（乌俄互译质量极高，保留率预期 >80%）
- rus_restaurant: ~2.0x → ~3.0x（增加多样性）

```bash
# 3语言增强数据生成
python scripts/augment_data.py \
  --input data/subtask_3/eng/eng_restaurant_train_alltasks.jsonl \
  --output data/subtask_3/eng/eng_restaurant_train_augmented.jsonl \
  --langs de fr zh

# 4语言增强数据生成（含RU）
python scripts/augment_data.py \
  --input data/subtask_3/eng/eng_restaurant_train_alltasks.jsonl \
  --output data/subtask_3/eng/eng_restaurant_train_aug_full.jsonl \
  --langs de fr zh ru

# 多seed训练
python scripts/train_multi_seed.py \
  --config configs/config_rethink_v3.yaml \
  --train_file data/subtask_3/eng/eng_restaurant_train_augmented.jsonl \
  --save_prefix aug_ensemble_seed

# 双层 Ensemble 推理（当前最佳）
python scripts/inference.py --multi_seed_ensemble --dual_layer \
  --seed_dirs checkpoints/aug_ensemble_seed42 checkpoints/aug_ensemble_seed123 \
    checkpoints/aug_ensemble_seed456 checkpoints/aug_ensemble_seed789 \
    checkpoints/aug_ensemble_seed1024 \
  --config configs/config_rethink_v3.yaml \
  --test_file data/subtask_3/eng/eng_restaurant_dev_task3.jsonl \
  --output predictions/dual_layer_v3.jsonl
```

#### 待完成优化

| 方案 | 方法 | 预期提升 |
|------|------|----------|
| 4语言增强重训 | 用 8540 样本（含 RU）重训 5-seed | +0.5~1% |
| D2 同义词替换 | 替换 aspect/opinion 词（WordNet/nlpaug） | +1~2% |
| D3 LLM 伪标注 | GPT/Qwen 对未标注评论生成伪标签 | +2~4% |
| D4 跨语言数据 | 利用中文/日文等数据做多语言联合训练 | +1~2% |

#### Phase 2: 表征增强

| 方案 | 方法 | 参考 | 预期提升 |
|------|------|------|----------|
| R1 VA 对比学习 | InfoNCE loss 校准 VA 空间 | CCIIPLab (SIGHAN 第 2 名) | +1~2% |
| R2 领域自适应预训练 | 餐厅评论语料 MLM 继续预训练 | HITSZ-HLT | +1~2% |
| R3 Biaffine Attention | 替代拼接配对方式 | STBA | +0.5~1% |

#### Phase 3: LLM 融合

| 方案 | 方法 | 参考 | 预期提升 |
|------|------|------|----------|
| F1 BERT + LLM 混合 | LLM 生成 quad 候选，与 BERT 结果融合 | HITSZ-HLT (SIGHAN 冠军) | +2~4% |
| ~~F2 双层 Ensemble~~ | ~~每个 seed 内 top-k 权重平均 + seed 间投票~~ | — | ✅ 已实现 +0.24% |

### 5.3 V5 全语言增强 Ensemble 结果（✅ 已完成）

对所有 8 个语言/领域执行完整 pipeline：回译增强 → 5-seed 训练 → Dual-layer Ensemble 推理。

**增强数据统计：**

| 语言/领域 | 原始样本 | 增强后样本 | 倍率 | Encoder | 回译方向 |
|-----------|---------|-----------|------|---------|---------|
| eng_restaurant | 2,284 | 7,106 | 3.11x | deberta-v3-base | en→de/fr/zh/ru→en |
| eng_laptop | 4,076 | 15,049 | 3.69x | deberta-v3-base | en→de/fr/zh/ru→en |
| jpn_hotel | 1,600 | ~3,000 | ~1.9x | mdeberta-v3-base | ja→en→ja |
| rus_restaurant | 1,240 | ~2,500 | ~2.0x | mdeberta-v3-base | ru→en→ru |
| tat_restaurant | 1,240 | ~2,500 | ~2.0x | mdeberta-v3-base | tat→en→tat (via trk) |
| ukr_restaurant | 1,240 | 1,615 | 1.30x | mdeberta-v3-base | uk→en→uk |
| zho_laptop | 3,490 | ~7,000 | ~2.0x | mdeberta-v3-base | zh→en→zh |
| zho_restaurant | 6,050 | ~12,000 | ~2.0x | mdeberta-v3-base | zh→en→zh |

**全语言官方评测结果（V5 Dual-layer Ensemble, vote≥3）：**

| 语言/领域 | cPrecision | cRecall | **cF1** | vs V3 基线 | 提升 |
|-----------|-----------|---------|---------|-----------|------|
| eng_restaurant | 0.7164 | 0.6040 | **0.6554** | 0.6162 | +3.92% |
| eng_laptop | 0.3798 | 0.2924 | **0.3304** | 0.2591 | +7.13% |
| jpn_hotel | 0.2896 | 0.2459 | **0.2660** | 0.2489 | +1.71% |
| rus_restaurant | 0.3898 | 0.3401 | **0.3633** | 0.3305 | +3.28% |
| tat_restaurant | 0.4695 | 0.3452 | **0.3979** | 0.3786 | +1.93% |
| ukr_restaurant | 0.4564 | 0.3535 | **0.3984** | 0.3079 | +9.05% |
| zho_laptop | 0.4131 | 0.3119 | **0.3554** | 0.3190 | +3.64% |
| zho_restaurant | 0.4816 | 0.4677 | **0.4745** | 0.4366 | +3.79% |
| **平均** | | | **0.4052** | 0.3621 | **+4.31%** |

### 5.4 vs 官方 Baseline 对比（论文 arXiv:2601.23022）

论文仅提供 LLM baseline（无 BERT-based baseline）。我们的 DeBERTa + 增强 + Ensemble 方案超越所有官方 baseline：

| 语言/领域 | **我们 V5** | Llama-3.3 70B | GPT-OSS 120B | GPT-5 mini (0-shot) | Kimi K2 (0-shot) |
|-----------|-----------|---------------|-------------|-------------------|-----------------|
| eng_restaurant | **0.6554** | 0.5048 | 0.5013 | 0.4036 | 0.3740 |
| eng_laptop | **0.3304** | 0.2483 | 0.2411 | 0.2304 | 0.2805 |
| jpn_hotel | 0.2660 | 0.3577 | **0.4151** | 0.0907 | 0.1309 |
| rus_restaurant | 0.3633 | **0.4118** | 0.3683 | 0.2508 | 0.2656 |
| tat_restaurant | **0.3979** | 0.3702 | 0.3094 | 0.1974 | 0.2310 |
| ukr_restaurant | 0.3984 | **0.4070** | 0.3663 | 0.2465 | 0.2974 |
| zho_restaurant | **0.4745** | 0.4391 | 0.4249 | 0.2481 | 0.2975 |
| zho_laptop | **0.3554** | 0.3506 | 0.3551 | 0.1356 | 0.1569 |
| **平均** | **0.4052** | 0.3862 | 0.3727 | 0.2254 | 0.2542 |

### 5.5 V6: 跨语言迁移学习 + MeCab 后处理（✅ 已完成）

**V6 核心改进：**
1. **跨语言迁移学习**：用 zho_restaurant（最大数据集，6050 条）的 best seed 作为 source，迁移 encoder 权重到其他 5 个非英语目标语言
2. **MeCab 日语后处理**：用形态素分析修正 span 边界（去除尾部助词/助动词等功能词）
3. **Tokenizer Bug 修复**：multi-seed ensemble 推理时 tokenizer 使用了英文 deberta 而非多语言 mdeberta，导致日语等语言 span 碎片化
4. **EMA 保存修复**：checkpoint 保存时正确 apply EMA shadow 权重，确保推理使用的是 EMA 平滑后的参数
5. **Source 选择优化**：使用 best seed (seed1024, cF1=0.4464) 而非固定 seed42 作为 transfer source

**V6 各语言单 seed 训练结果：**

| 语言/领域 | seed42 | seed123 | seed456 | seed789 | seed1024 | 平均 |
|-----------|--------|---------|---------|---------|----------|------|
| jpn_hotel | 0.2733 | 0.2789 | 0.2847 | 0.2808 | 0.2904 | 0.2816 |
| rus_restaurant | 0.4154 | 0.4273 | 0.4480 | 0.4585 | 0.4439 | 0.4386 |
| tat_restaurant | 0.4435 | 0.4140 | 0.4132 | 0.4488 | 0.4685 | 0.4376 |
| ukr_restaurant | 0.4500 | 0.4285 | 0.4406 | 0.4626 | 0.4815 | 0.4526 |
| zho_laptop | 0.3209 | 0.3255 | — | — | — | ~0.32 |

**V6 全语言 Ensemble 评测结果（vote≥3, jpn 用 vote≥4 + MeCab）：**

| 语言/领域 | V5 cF1 | **V6 cF1** | Δ | Llama-3.3 70B | GPT-OSS 120B | 胜负 |
|-----------|--------|-----------|---|---------------|-------------|------|
| eng_restaurant | 0.6554 | **0.6554** | — | 0.5048 | 0.5013 | **胜** |
| eng_laptop | 0.3304 | **0.3304** | — | 0.2483 | 0.2411 | **胜** |
| jpn_hotel | 0.2660 | **0.3312** | +6.5% | 0.3577 | 0.4151 | 负（差距缩小） |
| rus_restaurant | 0.3633 | **0.4520** | +8.9% | 0.4118 | 0.3683 | **胜** ✨ |
| tat_restaurant | 0.3979 | **0.4538** | +5.6% | 0.3702 | 0.3094 | **胜** |
| ukr_restaurant | 0.3984 | **0.4674** | +6.9% | 0.4070 | 0.3663 | **胜** ✨ |
| zho_restaurant | 0.4745 | **0.4745** | — | 0.4391 | 0.4249 | **胜** |
| zho_laptop | 0.3554 | **0.3526** | -0.3% | 0.3506 | 0.3551 | **胜** |
| **平均** | 0.4052 | **0.4397** | **+3.5%** | 0.3862 | 0.3727 | **胜** |

**V6 分析：**
- 平均 cF1 **0.4397**，超越最强 baseline Llama-3.3 70B (0.3862) **+5.4%**
- **7/8 语言超越所有 baseline**（V5 仅 5/8），新增突破 rus_restaurant (+4.0% vs Llama) 和 ukr_restaurant (+6.0% vs Llama)
- jpn_hotel 从 0.2660 提升至 **0.3312**（+6.5%），与 Llama-3.3 差距从 9.2% 缩小至 2.7%
- jpn_hotel 仍是唯一落后的语言，主因：44 个 category（其他语言仅 12-14 个）、hotel 领域与 source（restaurant）不匹配
- 迁移学习对低资源语言效果最佳：ukr +6.9%, rus +8.9%
- MeCab 后处理为 jpn 额外贡献 +3.2% cF1

**日语 vote threshold 消融（含 MeCab 后处理）：**

| Vote | 无 MeCab | + MeCab | 提升 |
|------|---------|---------|------|
| ≥1 | 0.2530 | 0.2875 | +3.5% |
| **≥2** | 0.2910 | **0.3323** | **+4.1%** |
| ≥3 | 0.2955 | 0.3225 | +2.7% |
| ≥4 | 0.2990 | 0.3312 | +3.2% |
| ≥5 | 0.2823 | 0.3141 | +3.2% |

### 5.6 目标里程碑

| 阶段 | 实际/预期 cF1 | 对比 V3 基线 | 状态 |
|------|-------------|-------------|------|
| V3 单模型 (eng_rest) | 0.6278 | 基线 | ✅ |
| V4 Ensemble (eng_rest) | 0.6415 | +2.2% | ✅ |
| V5 增强 Ensemble (eng_rest) | 0.6554 | +4.4% | ✅ |
| V5 全语言 Ensemble (8 lang avg) | 0.4052 | 超越所有官方 baseline | ✅ |
| **V6 迁移学习 + MeCab (8 lang avg)** | **0.4397** | **7/8 超越所有 baseline** | **✅ 当前最佳** |
| V7 多源迁移（jpn）| 0.3279 | jpn 单语言 −0.33% | ❌ 失败 |
| V8 LLM 全合成（jpn）| 0.3202 | jpn 单语言 −1.10% | ❌ 失败 |
| V9 paraphrase tongyi（jpn）| 0.3227 | jpn 单语言 −0.85% | ❌ 失败 |
| V10 paraphrase DeepSeek（jpn）| 0.3299 | jpn 单语言 −0.13% | ⚠ 接近持平 |
| V11 V10 + COMET≥0.92（jpn）| 0.3317 | jpn 单语言 +0.05% | ✅ 反超 V6 |
| **V12 V10 + COMET≥0.95（jpn）** | **0.3523** | **jpn 单语言 +2.11%** | **✅✅ 当前最佳** |

---

### 5.7 V7：多源迁移（jpn）（❌ 已尝试，未提升）

**动机**：V6 用 zho_restaurant → jpn 单源迁移取得 0.3312。假设两个源（eng+zho）平均 encoder 权重能提供更丰富的初始化。

**实现**：[scripts/train.py](scripts/train.py) 新增 `--pretrained_checkpoints`（复数）参数，对多个 source 的 encoder 取参数平均。

```python
for key in encoder_keys:
    matching = [s[key] for s in all_encoder_states if key in s and s[key].shape == model_state[key].shape]
    if matching:
        model_state[key] = torch.stack(matching).mean(dim=0)
```

**结果**：cF1 = **0.3279** ↓ 0.33%（vs V6）

**失败原因分析**：
- eng_restaurant 和 zho_restaurant 的 encoder 在不同语言上学到不同的子空间
- 简单平均可能稀释了每个 source 的语言特定特征
- 推理时还出现 "using raw weights" 警告，提示 EMA 权重保存可能存在问题（待排查）

---

### 5.8 V8：LLM 全合成数据（jpn）（❌ 已尝试，未提升）

**动机**：用 LLM 完整生成新句子+四元组标注，扩充训练数据。

**实现**：[scripts/generate_pseudo_labels.py](scripts/generate_pseudo_labels.py) 调用 Qwen3.5-plus 生成 200 条日语酒店评论及其四元组标注，自动过滤无效样本得 199 条。

合并到训练集：1600（原始）+ 199（伪标注） = 1799 条。

**结果**：cF1 = **0.3202** ↓ 1.10%（vs V6）

**失败原因分析**：
- **双层噪声**：LLM 既要造句子又要造标签，两边都可能错
- 199 条合成数据 vs 1600 条 gold 比例失衡（11%）但仍引入显著噪声
- VA 标注风格偏差：LLM 输出值集中在 5.0-7.0 区间，缺少极端情感

---

### 5.9 V9：Paraphrase + 标签复用（tongyi 模型）（❌ 部分提升 V8，仍未达 V6）

**核心思路**：解决 V8 的双层噪声问题 → **复用 gold 标签**，只让 LLM 改写文本，硬约束 aspect/opinion 子串必须在 paraphrase 中逐字保留。

**实现**：[scripts/paraphrase_with_labels.py](scripts/paraphrase_with_labels.py)
- 输入：每个原样本的 (Text, 所有非 NULL aspect/opinion 短语)
- LLM 输出：N 个 paraphrase 候选
- 过滤：① 所有保留短语必须在新文本中 ② 与原文相似度 <0.85（避免 trivial 改写）

**Pipeline 流程**

```
Gold sample (1600) → 提取保留短语 → LLM paraphrase × 2 variants
  → 子串验证 + 相似度验证 → 复用 gold quadruplet
```

**模型选择**：原计划用 qwen3.6-flash-2026-04-16，跑了 348 条后 free tier 耗尽，自动 fallback 到 `tongyi-xiaomi-analysis-pro` 完成剩余。

**统计**：
- 接受率 81.1%
- 覆盖源样本 1497 / 1600（93.6%）
- Paraphrase 总数 2699
- 训练集 4299 条

**结果**：cF1 = **0.3227** ↓ 0.85%（vs V6）

**失败原因分析（关键）**：

Pilot 时已发现 tongyi 输出的"语义漂移"问题：

| 原文 | tongyi paraphrase | 问题 |
|------|------------------|------|
| 「おかずが増えた**気がします**」（不确定）| 「**食べ応え十分でした**」（强肯定）| ✗ 情感强度漂移 |
| 「ちょっぴり残念だった…寂しかった」 | + 添加「館内の雰囲気も少し寂しかった」 | ✗ 扩句引入新内容 |

子串保留过滤器**检测不出语义漂移**，需要更强的 LLM 或语义级过滤器。

---

### 5.10 V10：Paraphrase + DeepSeek V4 Pro（⚠ 接近持平 V6）

**改进**：用更强的 LLM `deepseek-v4-pro` 替代 tongyi，并加并发（8 workers）加速。

**实现亮点**：
- 模型链 fallback：`deepseek-v4-pro → deepseek-chat → qwen3.6-flash → tongyi → ...`，配额耗尽自动切换
- 并发：`ThreadPoolExecutor(max_workers=8)`，速率从 0.04 src/s 提升到 0.30 src/s（7.5x）
- Resume 机制：基于 SourceID 自动跳过已完成样本

**统计**：
| 指标 | V9 (tongyi) | **V10 (DeepSeek)** | Δ |
|------|-------------|--------------------|---|
| 接受率 | 81.1% | **93.8%** | +12.7pp |
| 覆盖源样本 | 1497 (93.6%) | **1561 (97.6%)** | +4pp |
| Paraphrase 总数 | 2699 | **2957** | +258 |

**Paraphrase 质量对比**（Sample 3 是 V9 翻车案例）：

| 原文 | V9 (tongyi) | V10 (DeepSeek) |
|------|-------------|----------------|
| 「おかずが増えた**気がします**」 | ❌「食べ応え十分でした」 | ✅「おかずが増えた**ように思います**」 |

V10 完美保留不确定语气。

**单 seed 完整对比（jpn dev）**：

| Seed | V9 | **V10** | Δ |
|------|-----|---------|---|
| 42   | 0.2772 | **0.2798** | +0.26pp |
| 123  | 0.2794 | **0.2841** | +0.47pp |
| 456  | 0.2783 | **0.2858** | +0.75pp |
| 789  | 0.2860 | 0.2751 | −1.09pp |
| 1024 | 0.2744 | **0.2842** | +0.98pp |
| **Avg** | 0.2791 | **0.2818** | **+0.27pp** |

**Ensemble 结果**（vote=2 + MeCab）：cF1 = **0.3299**（cP=0.3197 / cR=0.3408）↓ 0.13%（vs V6 0.3312）

V10 几乎追平 V6，但仍未反超。差距 0.13% 在统计噪声范围内。

---

### 5.11 COMET-Kiwi 语义过滤实验（❌ 不适合本任务）

**动机**：组会建议用 [Unbabel COMET](https://github.com/Unbabel/COMET) 给每条 paraphrase 打语义相似度分数，丢弃漂移样本。

**实施**：
- 模型：`Unbabel/wmt22-comet-da`（开源，reference-based；wmt22-cometkiwi-da 是 gated）
- 输入构造：src=original, mt=paraphrase, ref=original（让模型评估 paraphrase 与原文的语义距离）
- 评分：[scripts/comet_filter_paraphrases.py](scripts/comet_filter_paraphrases.py)

**得分分布**（2957 条 V10 paraphrase）：

| 百分位 | COMET |
|--------|-------|
| min | 0.654 |
| 10% | 0.924 |
| 25% | 0.942 |
| median | ~0.96 |
| 75% | 0.968 |
| 90% | 0.975 |
| **mean** | **0.951** |

**关键发现：COMET 给"低多样性 trivial 改写"高分，给"有内容增益的 paraphrase"低分**。

低分样本（bottom 5）实际质量分析：

| COMET | 实际情况 |
|-------|---------|
| 0.654 | 原文是英语（数据集脏数据，非 paraphrase 问题） |
| 0.672 | 「食べきれない…全て頂きました」→「ボリューム…全部平らげました」**意思一致**，COMET 误判 |
| 0.695 | 原文仅「たかい」，paraphrase 添加「宿泊費が」上下文 — 对 ABSA 训练**反而更好** |
| 0.719 | 「手際が悪くイライラした」→「手ぎわが悪くずっとイライラしていた」**完全等价** |

高分样本（top 5）全是琐碎词序调整或同义词替换。

**初步分析（事后被推翻）**：当时认为 COMET 不适合本任务，原因：
1. DeepSeek paraphrase 已无显著语义漂移
2. COMET 对"内容扩展"和"多样性改写"评分偏低，但这些样本对训练似乎有益
3. 真正应丢弃的是数据集中混入的英语原文

→ **本判断被 V11 实验推翻**（见 5.12）。即便人工审查 bottom 5 看起来"合理"，实际训练时 COMET 过滤仍带来显著提升。

---

### 5.12 V11：DeepSeek Paraphrase + COMET ≥ 0.92 过滤（✅ 反超 V6！当前最佳）

**动机**：尽管 5.11 的人工审查认为 COMET 误判低分样本，仍以"科学态度"跑一次 V11 验证假设。结果出乎意料 —— **5 seeds 全部显著提升**。

**配置**：
- 训练数据：1600 原始 + 2702 经 COMET ≥ 0.92 过滤的 paraphrase = **4302 条**（V10 是 4557，少 5.6%）
- 丢弃 255 条 paraphrase（8.6% of V10 paraphrase）
- 其他配置完全复刻 V10（mDeBERTa-v3-base, transfer from zho_restaurant, FGM, EMA, span loss=2.0, arousal weight=2.0）
- transformers 4.57.6（COMET 安装时降级，验证训练兼容性正常）

**单 seed 完整对比**：

| Seed | V10 | **V11** | Δ |
|------|-----|---------|---|
| 42 | 0.2798 | **0.3042** | **+2.44pp** |
| 123 | 0.2841 | **0.3127** | **+2.86pp** |
| 456 | 0.2858 | **0.3143** | **+2.85pp** |
| 789 | 0.2751 | **0.2938** | **+1.87pp** |
| 1024 | 0.2842 | **0.3061** | **+2.19pp** |
| **Avg** | 0.2818 | **0.3062** | **+2.44pp** |

**5 seeds 全部提升**，平均 +2.44pp。

**Ensemble 评测**（vote=2 + MeCab）：

| 指标 | V10 | **V11** | Δ |
|------|-----|---------|---|
| cPrecision | 0.3197 | 0.3073 | −1.24pp |
| cRecall | 0.3408 | **0.3605** | +1.97pp |
| **cF1** | **0.3299** | **0.3317** | **+0.18pp** |

V11 是**首个反超 V6 的实验**（V7/V8/V9/V10 全部 −0.13% ~ −1.10% 落后）。Recall 大幅提升（+1.97pp）说明过滤后的更高质量 paraphrase 帮助模型识别更多真实四元组，但 Precision 略降说明仍有噪声。

**关键洞察 / 教训**：
- **不能仅凭人工审查 bottom 样本判断 COMET 有效性**。COMET ≥ 0.92 过滤掉的 255 条样本中，可能含有人眼难以识别但对训练有害的微妙语义偏差
- **数据"质 > 量"**：V11 数据量比 V10 少 5.6%，但 cF1 反而更高
- **COMET 反而对 DeepSeek 高质量 paraphrase 更有用**：因为 DeepSeek 整体水平高，COMET 仅切除尾部 8.6% 的"次优"样本，剩下的都是精华
- 之前在 V9 (tongyi) 上想做 COMET 过滤会更激进（tongyi 整体接受率 81%，可能要切 30%+），效果未必如此

---

### 5.13 V12：DeepSeek Paraphrase + COMET ≥ 0.95（✅✅ 大幅突破，新最佳 cF1=0.3523）

**动机**：V11 (COMET ≥ 0.92) 仅切 8.6% paraphrase，提升微弱（+0.05% vs V6）。尝试更激进的 0.95 阈值，丢弃 36.9% 样本，验证"数据质 > 量"假设的边界。

**配置**：
- 训练数据：1600 原始 + 1866 经 COMET ≥ 0.95 过滤的 paraphrase = **3466 条**（V11 是 4302，少 19.4%；V10 是 4557，少 24.0%）
- 其他配置完全复刻 V11

**单 seed 完整对比**：

| Seed | V11 (0.92) | V12 (0.95) | Δ |
|------|-----------|-----------|---|
| 42 | 0.3042 | 0.3002 | −0.40pp |
| 123 | 0.3127 | 0.3015 | −1.12pp |
| 456 | 0.3143 | 0.2828 | −3.15pp |
| 789 | 0.2938 | 0.2925 | −0.13pp |
| 1024 | 0.3061 | 0.3023 | −0.38pp |
| **Avg** | 0.3062 | **0.2959** | **−1.03pp** |

**Ensemble 评测**（vote=2 + MeCab）：

| 指标 | V11 | **V12** | Δ |
|------|-----|---------|---|
| cPrecision | 0.3073 | 0.3314 | **+2.41pp** |
| cRecall | 0.3605 | 0.3760 | **+1.55pp** |
| **cF1** | **0.3317** | **0.3523** | **+2.06pp** ✨ |

**反直觉发现**：
- 单 seed 平均 **降低** 1.03pp
- Ensemble cF1 **大幅提升** 2.06pp
- 这是因为更激进的过滤让每个 seed 看到的"精华子集"略有不同，**seed 间分歧更大**，ensemble 时投票互消的效果更显著

**核心洞察**：
1. **质 > 量的边界比预期更宽**：从 4557 (V10) 降到 3466 (V12)，丢 24% 数据，cF1 提升 +2.24%
2. **多样性是 ensemble 的隐藏维度**：传统 ensemble 关注 seed 随机性带来的多样性，但**数据"硬筛"也是多样性来源**
3. **Precision/Recall 同步提升**（罕见现象）：通常二者是 trade-off。V12 同时提升说明真的是数据质量改善，不是阈值调节人为换分

---

## 6. 主要结果

### 6.1 版本演进总览

| 版本 | 关键改动 | cF1 | Δ cF1 | 状态 |
|------|----------|-----|-------|------|
| Baseline | DeBERTa-v3-base, 无 Rethink | 0.2935 | — | 已废弃 |
| Rethink V1 | + Rethink 模块 | 0.4064 | +11.3% | 已废弃 |
| Rethink V2 | + 强 FGM (epsilon=0.5) | 0.2326 | -17.4% | 已废弃 |
| V3 + Span Fix | + 字符级位置追踪 | 0.3836 | +15.1% | 已废弃 |
| V3 + Hungarian | + 最优配对匹配 | 0.4348 | +5.1% | 已废弃 |
| **V3 Final** | **+ EMA + FGM(0.1) + span=2.0** | **0.6278** | **+19.3%** | 单模型最佳 |
| V3-CRF (实验) | + CRF + NULL Anchor | 0.6249 | -0.3% | 已回退 |
| V3-CRF + Large | DeBERTa-v3-large | 0.5539 | -7.4% | 已回退 |
| V4 Ensemble | 5-seed 投票, 原始数据 | 0.6415 | +2.2% | ✅ |
| V5 增强单模型 | 回译增强 (3.1x数据), 单模型 | 0.6474 | +3.1% | ✅ |
| V5 增强 Ensemble | 回译增强 + 5-seed 投票 | 0.6530 | +4.0% | ✅ |
| V5 双层 Ensemble | 回译增强 + 5-seed dual-layer | 0.6554 | +4.4% | ✅ |
| V5 全语言 Ensemble | 8 语言增强 + dual-layer (avg cF1) | 0.4052 | 超越所有官方 baseline | ✅ |
| **V6 迁移学习 + MeCab** | **跨语言 transfer + 日语后处理 (avg cF1)** | **0.4397** | **7/8 超越所有 baseline** | **✅ 当前最佳** |
| V7 多源迁移 (jpn) | eng+zho encoder 平均 | 0.3279 | jpn −0.33% vs V6 | ❌ |
| V8 LLM 全合成 (jpn) | +199 Qwen3.5 伪标注 | 0.3202 | jpn −1.10% vs V6 | ❌ |
| V9 paraphrase tongyi (jpn) | +2699 标签复用改写 | 0.3227 | jpn −0.85% vs V6 | ❌ |
| V10 paraphrase DeepSeek (jpn) | +2957 高质量改写（多并发）| 0.3299 | jpn −0.13% vs V6 | ⚠ 接近持平 |
| V11 V10 + COMET≥0.92 (jpn) | +2702 过滤后 paraphrase | 0.3317 | jpn +0.05% vs V6 | ✅ 反超 V6 |
| **V12 V10 + COMET≥0.95 (jpn)** | **+1866 中度过滤 paraphrase** | **0.3523** | **jpn +2.11% vs V6** | **✅✅ 大幅突破** |

### 6.2 代码修复前后对比

| 版本 | 代码版本 | cF1 | Exact F1 | VA RMSE | Best Epoch |
|------|----------|-----|----------|---------|------------|
| Baseline | 旧代码 | 0.2935 | 0.3252 | 1.3200 | 7 |
| Rethink V1 | 旧代码 | 0.4064 | 0.4490 | 1.3125 | 18 |
| Rethink V2 | 旧代码 | 0.2326 | 0.2578 | 1.3601 | 7 |
| V3 + Span修复 | 新代码 | 0.3836 | 0.4195 | 1.2208 | 11 |
| V3 + Span + Hungarian | 新代码 | 0.4348 | 0.4748 | 1.1406 | 16 |
| **V3 + Loss Weight Sweep (span=2.0)** | **新代码** | **0.6278** | **-** | **-** | **-** |

### 6.3 关键修复说明

#### (1) Span 匹配修复

**问题**: 原代码使用简单的 `text.find(span)` 匹配，无法处理重复出现的 span

**示例**:
```
Text: "The food was good and the service was good too"
Quadruplets:
  1. (food, FOOD#QUALITY, good, 7.0#5.0)
  2. (service, SERVICE#GENERAL, good, 7.0#5.0)

旧代码: 两个 "good" 都匹配到第一个位置 → 标签错误
新代码: 使用位置追踪，第二个 "good" 正确匹配到第二个位置
```

**修复效果**: cF1 +5.12% (0.3836 vs baseline重跑)

#### (2) Hungarian 匹配修复

**问题**: 原代码假设预测 pair 顺序与 gold 一致，但实际是笛卡尔积

**示例**:
```
Gold Pairs: [(food, good), (service, terrible)]
Pred Pairs (笛卡尔积): [(food, good), (food, terrible), (service, good), (service, terrible)]

旧代码: 直接按顺序计算 loss → (food, good) 对 (food, good) ✓
                              (service, terrible) 对 (food, terrible) ✗
新代码: Hungarian 匹配找到最优对应
```

**修复效果**: cF1 +5.12% (0.4348 vs 0.3836)

### 6.4 训练曲线 (V3 + 所有修复)

| Epoch | cF1 | Exact F1 | VA RMSE |
|-------|-----|----------|---------|
| 1 | 0.0000 | 0.0000 | 0.0000 |
| 5 | 0.1196 | 0.1388 | 2.1051 |
| 10 | 0.2815 | 0.3090 | 1.2382 |
| 13 | 0.3947 | 0.4372 | 1.3058 |
| **16** | **0.4348** | **0.4748** | **1.1406** |
| 20 | 0.3908 | 0.4319 | 1.2714 |
| 24 | 0.4163 | 0.4549 | 1.1121 (early stop) |

---

## 7. 消融实验

### 7.1 模块消融

| 配置 | Rethink | Hungarian | FGM | cF1 | Δ cF1 |
|------|---------|-----------|-----|-----|-------|
| Baseline | ✗ | ✗ | ✗ | 0.2935 | - |
| +Rethink | ✓ | ✗ | ✗ | 0.4064 | +11.29% |
| +Span Fix | ✓ | ✗ | ✓ | 0.3836 | +9.01% |
| +Hungarian | ✓ | ✓ | ✓ | **0.4348** | **+14.13%** |

### 7.2 Rethink 迭代次数

| 迭代次数 | cF1 | 说明 |
|----------|-----|------|
| 0 (无 Rethink) | 0.2935 | Baseline |
| 1 | 0.3856 | 单次精炼 |
| 2 | **0.4348** | 最佳 |
| 3 | 0.4201 | 略有下降 |

### 7.3 FGM epsilon 参数

| epsilon | cF1 | 说明 |
|---------|-----|------|
| 0 (无 FGM) | 0.4064 | V1 配置 |
| 0.1 | **0.4348** | V3 配置 (最佳) |
| 0.3 | 0.3912 | 扰动过强 |
| 0.5 | 0.2326 | V2 配置 (训练不稳定) |

### 7.4 学习率影响

| Encoder LR | Head LR | cF1 |
|------------|---------|-----|
| 2e-5 | 1e-4 | 0.4064 |
| 1e-5 | 5e-5 | 0.2326 |
| **1.5e-5** | **7.5e-5** | **0.4348** |
| 3e-5 | 1.5e-4 | 0.3756 |

---

### 7.5 V4 实验消融

| 实验 | 改动 | cF1 | vs V3 |
|------|------|-----|-------|
| V3 基线 | — | **0.6278** | — |
| + CRF + NULL | use_crf=true | 0.6249 | -0.3% |
| + CRF Loss Clamp | clamp(max=20.0) | 0.6177 | -1.0% |
| + DeBERTa-large | 304M params | 0.5539 | -7.4% |
| Loss Sweep (best) | span=2.5, va=2.0 | 0.6271 | -0.1% |
| Pair Threshold | 0.05~0.50 全范围 | 0.6264 | 0% |

---

## 8. 错误分析

### 8.1 错误类型分布

基于 Dev Set 200 样本的分析：

| 错误类型 | 占比 | 示例 |
|----------|------|------|
| Span 边界错误 | 35% | "good thai" vs "thai" |
| Category 错误 | 25% | FOOD#QUALITY vs FOOD#GENERAL |
| 漏检 (FN) | 20% | 隐式 aspect 未识别 |
| 多检 (FP) | 15% | 非情感词误判为 opinion |
| V-A 偏差大 | 5% | 强烈情感词 V-A 预测偏中性 |

### 8.2 典型错误案例

#### Case 1: Span 边界模糊

```
Text: "The fried rice was pretty good."
Gold: (fried rice, FOOD#QUALITY, pretty good, 7.0#5.5)
Pred: (rice, FOOD#QUALITY, good, 6.5#5.0)

问题: "fried" 和 "pretty" 边界识别错误
```

#### Case 2: 隐式 Aspect

```
Text: "It's a bit pricey but worth every penny."
Gold: (NULL, RESTAURANT#PRICES, pricey, 4.0#4.5)
      (NULL, RESTAURANT#GENERAL, worth every penny, 7.5#6.0)
Pred: (缺失)

问题: NULL aspect 难以识别
```

#### Case 3: V-A 极端值

```
Text: "Absolutely disgusting! Never coming back!"
Gold: V=1.5, A=8.5 (极负面 + 高唤醒)
Pred: V=3.2, A=6.1 (预测偏中性)

问题: 极端情感的 V-A 预测偏保守
```

### 8.3 改进方向

| 问题 | 建议方案 | 优先级 |
|------|----------|--------|
| Span 边界 | 添加 CRF 层约束 | 高 |
| NULL Aspect | 增加 NULL 专门处理逻辑 | 高 |
| V-A 极端值 | 使用分布预测 or 对比学习 | 中 |
| Category 混淆 | 增加 Entity-Attribute 联合约束 | 中 |

---

## 附录

### A. 完整配置对比

| 参数 | Baseline | V1 | V2 | V3 (Final) |
|------|----------|----|----|-----|
| encoder_name | deberta-v3-base | ← | ← | ← (mdeberta for non-eng) |
| use_rethink | false | true | true | true |
| rethink_iterations | 1 | 2 | 2 | 2 |
| encoder_lr | 2e-5 | 2e-5 | 1e-5 | 1.5e-5 |
| head_lr | 1e-4 | 1e-4 | 5e-5 | 7.5e-5 |
| num_epochs | 20 | 20 | 30 | 30 |
| patience | 5 | 5 | 8 | 8 |
| fgm_enabled | false | false | true | true |
| fgm_epsilon | - | - | 0.5 | 0.1 |
| ema_enabled | false | false | false | true |
| ema_decay | - | - | - | 0.999 |
| label_smoothing | 0 | 0 | 0 | 0.1 |
| loss_weights.span | 1.0 | 1.0 | 1.0 | **2.0** |
| loss_weights.consistency | 0.1 | 0.1 | 0.1 | 0.5 |
| save_top_k | 1 | 1 | 1 | 3 |
| hungarian_matching | false | false | false | true |
| span_position_fix | false | false | false | true |

### B. 训练命令

```bash
# 单语言训练
python scripts/train.py --config configs/config_rethink_v3.yaml

# 全语言训练 (8 个数据集)
python scripts/train_all_languages.py --base_config configs/config_rethink_v3.yaml

# 全语言评测
python scripts/batch_eval.py --base_config configs/config_rethink_v3.yaml

# 全语言 ensemble 评测
python scripts/batch_eval.py --base_config configs/config_rethink_v3.yaml --ensemble
```

### C. 检查点使用

```python
import torch
from src.models.dimabsa_model import DimASQPModel

# 加载最佳模型
checkpoint = torch.load('checkpoints/rethink_v3/checkpoint_best.pt')
print(f"Best Epoch: {checkpoint['epoch']}")
print(f"Best cF1: {checkpoint['metrics']['continuous_f1']:.4f}")

# 恢复模型
model = DimASQPModel(...)
model.load_state_dict(checkpoint['model_state_dict'])
```
