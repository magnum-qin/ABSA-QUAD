# 实验记录与结果分析

本文档记录 DimABSA-QUAD 项目的实验配置、结果对比和消融研究。

## 目录

1. [实验环境](#1-实验环境)
2. [数据集](#2-数据集)
3. [评估指标](#3-评估指标)
4. [配置版本演进](#4-配置版本演进) (Baseline → V1 → V2 → V3 → V3-CRF(失败) → V4 Ensemble → V5 增强Ensemble)
5. [V5 规划](#5-v5-规划) (数据增强✅ → 表征优化 → LLM 融合)
6. [主要结果](#6-主要结果)
7. [消融实验](#7-消融实验)
8. [错误分析](#8-错误分析)

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

**实际 cF1：0.6530**（cPrecision=0.7049，cRecall=0.6082），**+4.0% vs V3 单模型**

```bash
# 增强数据生成
python scripts/augment_data.py \
  --input data/subtask_3/eng/eng_restaurant_train_alltasks.jsonl \
  --output data/subtask_3/eng/eng_restaurant_train_augmented.jsonl \
  --langs de fr zh

# 多seed训练
python scripts/train_multi_seed.py \
  --config configs/config_rethink_v3.yaml \
  --train_file data/subtask_3/eng/eng_restaurant_train_augmented.jsonl \
  --save_prefix aug_ensemble_seed

# Ensemble推理
python scripts/inference.py --multi_seed_ensemble \
  --seed_dirs checkpoints/aug_ensemble_seed42 checkpoints/aug_ensemble_seed123 \
    checkpoints/aug_ensemble_seed456 checkpoints/aug_ensemble_seed789 \
    checkpoints/aug_ensemble_seed1024 \
  --config configs/config_rethink_v3.yaml \
  --test_file data/subtask_3/eng/eng_restaurant_dev_task3.jsonl \
  --output predictions/aug_ensemble_pred.jsonl
```

#### Phase 2: 数据增强（待完成）

| 方案 | 方法 | 预期提升 |
|------|------|----------|
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
| F2 双层 Ensemble | 每个 seed 内 top-k 权重平均 + seed 间投票 | — | +0.5~1% |

### 5.3 目标里程碑

| 阶段 | 实际/预期 cF1 | 对比 V3 基线 | 状态 |
|------|-------------|-------------|------|
| V3 单模型 | 0.6278 | 基线 | ✅ |
| V4 Ensemble (原始数据) | 0.6415 | +2.2% | ✅ |
| V5 增强单模型 | 0.6474 | +3.1% | ✅ |
| **V5 增强 Ensemble** | **0.6530** | **+4.0%** | **✅ 完成** |
| V5 Phase 2 表征优化 | 0.67~0.69 | +6~7% | 待开始 |
| V5 Phase 3 LLM 融合 | 0.70+ | +7%+ | 待开始 |

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
| **V5 增强 Ensemble** | **回译增强 + 5-seed 投票** | **0.6530** | **+4.0%** | **✅ 当前最佳** |

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
