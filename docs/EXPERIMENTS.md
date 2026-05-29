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

### P1.4 跨语言推广 V13 配方（DeepSeek paraphrase + COMET filter + vote sweep）

将 V13 在 jpn 上的成功配方推广到其他 4 个非英语语言：

| 语言 | V6 (BT) | V13 (paraphrase) | Δ | best vote | 结论 |
|------|---------|-------------------|-----|-----------|------|
| jpn (baseline=0.3312) | 0.3312 | **0.3765** | **+4.53%** | 3 | ✅✅✅ |
| rus (baseline=0.4520) | 0.4520 | **0.4724** | **+2.04%** | 5 | ✅ |
| zho_laptop (0.3526) | 0.3526 | **0.3683** | **+1.57%** | 3 | ✅ |
| ukr (0.4674) | 0.4674 | 0.4422 | −2.52% | 4 | ❌ |
| tat (0.4538) | 0.4538 | 0.4325 | −2.13% | 5 | ❌ |
| **平均** | **0.4114** | **0.4184** | **+0.70%** | — | 3 胜 2 负 |

**规律：V13 配方与 baseline 高低成反比**

- **baseline 低（<0.40）→ 成功**：jpn/zho_laptop 提升 +1.57~4.53pp，因数据噪声高、paraphrase 改进空间大
- **baseline 高（>0.45）→ 失败**：ukr/tat 回退 −2.13~2.52pp，因 V6 已用 BT 增强（~5000 samples）、数据量优势，V13 paraphrase 数据少（500-1000）反而是劣势

**重要警示：V13 vs V6 同时改了两个变量**
- 增强方法：BT → paraphrase
- 数据量：~5000 → 500-3000（COMET 过滤后）

→ ukr/tat 失败可能不是 paraphrase 输给 BT，而是"少而精"输给"多而杂"。**论文应做的对比缺失**：
- V13b: BT + paraphrase 叠加（未做）
- V13c: COMET 过滤 BT 数据（未做）

**Paraphrase 数据质量与 LLM 多语言能力强相关**

| 语言 | DeepSeek 接受率 | COMET mean | 阈值 | kept |
|------|---------------|------------|------|------|
| jpn | 93.8% | 0.951 | 0.96 | 1293 |
| zho_laptop | 86.0% | 0.891 | 0.90 | 3006 |
| rus | 73.7% | 0.926 | 0.94 | 526 |
| ukr | 75.1% | 0.926 | 0.94 | 623 |
| tat | 67.2% | **0.768** | 0.80 | 359 |

Tatar 是低资源语言，DeepSeek 训练语料少 → COMET 分数低 → 可用 paraphrase 少 → V13 配方失效最严重。

---

### P1.4b 修正：V13b（BT + Paraphrase 叠加）与 V13c（更激进 COMET）

**动机**：V13 在 ukr/tat 失败，假设原因是 paraphrase **替换** BT 损失了多样性。试 V13b = BT + paraphrase **叠加**。

| 语言 | V13 配置 | V13b 配置 |
|------|---------|-----------|
| ukr | 1240 orig + 623 para = 1863 | 1240 orig + 1104 BT + 623 para = **2967** |
| rus | 1240 orig + 526 para = 1766 | 1240 orig + 1184 BT + 526 para = **2950** |
| zho_laptop | 3490 + 3006 para@0.90 = 6496 | (BT 仅 49 条，V13b 无意义) |
| jpn, tat | (BT 失败，V13b 无意义) | — |

**V13b 结果**：

| 语言 | V6 | V13 (替换 BT) | V13b (叠加 BT) | 结论 |
|------|-----|---------------|----------------|------|
| ukr | 0.4674 | 0.4422 (−2.52) | **0.5068 (+3.94)** 🏆 | **V13b 大胜**：BT+para 互补 |
| rus | 0.4520 | **0.4724 (+2.04)** | 0.4599 (+0.79) | **V13 已饱和**，V13b 反而拖累 |

**V13c（zho_laptop 专属）**：V13 已超 V6，尝试更激进 COMET 阈值（0.90 → 0.92）：

| 语言 | V13 (0.90) | V13c (0.92) |
|------|------------|-------------|
| zho_laptop | 0.3683 (+1.57) | **0.3791 (+2.65)** 🏆 |

V13c 在 zho_laptop 单 seed avg +1.25pp，ensemble +1.08pp，**高阈值过滤在大数据集上仍有效**（类似 jpn V12→V13 0.95→0.96）。

### 跨语言最终结果

| 语言 | V6 | V13 best | V13b/c best | **采用最佳** |
|------|-----|----------|--------------|--------------|
| jpn | 0.3312 | **0.3765** | — | V13 (+4.53%) |
| rus | 0.4520 | **0.4724** | 0.4599 | V13 (+2.04%) |
| ukr | 0.4674 | 0.4422 | **0.5068** | V13b (+3.94%) 🏆 |
| tat | **0.4538** | 0.4325 | — | V6 (持平) |
| zho_laptop | 0.3526 | 0.3683 | **0.3791** | V13c (+2.65%) 🏆 |
| **平均** | **0.4114** | 0.4184 | — | **0.4377 (+2.63pp)** |

**核心论点**：每个语言的**最佳增强配方不同**，由 BT 可用性、paraphrase 质量、V6 baseline 共同决定：
- BT 失败 + 低 V6 → V13 (jpn)
- BT 成功 + 中 V6 + paraphrase 单独够 → V13 (rus)
- BT 成功 + 高 V6 + 需要互补 → V13b (ukr)
- BT 失败 + 高 V6 → V6 仍最优 (tat)
- BT 失败 + 大数据集 → V13 + 高阈值 = V13c (zho_laptop)

这是论文 Discussion 章节的核心论点。

---

### V6 → V15+topk 一览（jpn_hotel dev cF1）

| 版本 | 方法 | 训练样本 | cF1 | Δ vs V6 |
|------|------|---------|------|---------|
| V6 | zho transfer + MeCab + vote=2 | 1600 | 0.3312 | baseline |
| V7 | multi-source transfer (eng+zho) | 1600 | 0.3279 | −0.33% |
| V8 | +199 全合成（文本+标签）| 1799 | 0.3202 | −1.10% |
| V9 | +2699 paraphrase（tongyi）| 4299 | 0.3227 | −0.85% |
| V10 | +2957 paraphrase（DeepSeek 全量）| 4557 | 0.3299 | −0.13% |
| V11 | V10 + COMET≥0.92（轻度过滤）| 4302 | 0.3317 | +0.05% |
| V12 | V10 + COMET≥0.95（中度过滤）| 3466 | 0.3523 | +2.11% |
| V13 (vote=2) | V10 + COMET≥0.96（最优过滤）| 2893 | 0.3565 | +2.53% |
| V14 | V10 + COMET≥0.965（过度过滤）| 2549 | 0.3418 | +1.06% |
| V13 (vote=3) | V13 ckpts + vote=3 重调（无需重训）| 2893 | 0.3765 | +4.53% |
| V15 (XLM-R-large) | V13 数据 + xlm-roberta-large encoder | 2893 | 0.3924 | +6.12% |
| **V15 + top-k** ⭐⭐⭐⭐⭐ | **V15 + top-k span decoding（topk=0.15）** | **2893** | **0.3962** | **+6.50%** 🏆 |

**核心结论**：
1. **倒 U 型 COMET 阈值-效应曲线**：0.92→0.95→0.96→**(峰值)**→0.965，**V13 (0.96) 是甜区顶点**
2. **质 > 量但有边界**：训练数据从 4557 (V10) 减到 2893 (V13) 仍提升；但减到 2549 (V14) 出现 underfitting
3. **Encoder scaling 大幅突破**：V15 (XLM-R-large) 比 V13 (mDeBERTa-base) 单 seed avg **+7.52pp**, ensemble +1.59pp
4. **Top-k Span Decoding 解决 recall 瓶颈**：+0.38pp（仅推理参数，无需重训）
5. **V15+top-k 已超过 Llama-3.3 70B**（0.3962 vs 0.3577 = +3.85pp）；未超过 GPT-OSS 120B (0.4151)

详见 [§5.18 V15-jpn](#518-v15-jpnxlm-r-large-encoder-升级--单-seed-752pp), [§5.19 Top-k Span](#519-v15--top-k-span-decoding--单-encoder-内最佳-cf103962), [§5.20 失败实验集](#520-失败实验集v15-之后探索未带来提升的尝试)。
3. **V14 = 阈值上限标记**：5/5 seeds 全回退，平均 −1pp，证明 0.96 已是最佳过滤强度

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
| V12 V10 + COMET≥0.95（jpn）| 0.3523 | jpn 单语言 +2.11% | ✅✅ 大幅突破 |
| V13 V10 + COMET≥0.96 vote=2（jpn）| 0.3565 | jpn 单语言 +2.53% | ✅✅✅ |
| V14 V10 + COMET≥0.965（jpn）| 0.3418 | jpn 单语言 +1.06% | ❌ 过度过滤（甜区上限）|
| V13 + vote=3 重调（jpn）| 0.3765 | jpn 单语言 +4.53% | ✅✅✅✅ |
| V15 (XLM-R-large, jpn) | 0.3924 | jpn 单语言 +6.12% | ✅✅✅✅✅ |
| **V15 + Top-k Span (jpn)** | **0.3962** | **jpn 单语言 +6.50%** | **✅✅✅✅✅ 当前最佳** 🏆 |

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

### 5.14 V13：DeepSeek Paraphrase + COMET ≥ 0.96（✅✅✅ 再创新高 cF1=0.3565）

**动机**：V12 (COMET ≥ 0.95) 大幅提升后，验证更激进的 0.96 阈值是否仍有收益。

**配置**：
- 训练数据：1600 原始 + 1293 经 COMET ≥ 0.96 过滤的 paraphrase = **2893 条**（V12 是 3466，少 16.5%；V10 是 4557，少 36.5%）
- Paraphrase 数量首次低于原始数据
- 其他配置完全复刻 V12

**单 seed 完整对比**：

| Seed | V12 (0.95) | V13 (0.96) | Δ |
|------|-----------|-----------|---|
| 42 | 0.3002 | **0.3117** | +1.15pp |
| 123 | 0.3015 | 0.2974 | −0.41pp |
| 456 | 0.2828 | **0.2964** | +1.36pp |
| 789 | 0.2925 | **0.3080** | +1.55pp |
| 1024 | 0.3023 | **0.3067** | +0.44pp |
| **Avg** | 0.2959 | **0.3040** | **+0.81pp** |

**V13 单 seed 4/5 胜**，平均 +0.81pp。与 V12 的"单 seed 降低、ensemble 升高"模式不同。

**Ensemble 评测**（vote=2 + MeCab）：

| 指标 | V12 | **V13** | Δ |
|------|-----|---------|---|
| cPrecision | 0.3314 | 0.3331 | +0.17pp |
| cRecall | 0.3760 | 0.3834 | +0.74pp |
| **cF1** | **0.3523** | **0.3565** | **+0.42pp** ✨ |

**核心洞察（V11→V12→V13 阈值-效应曲线）**：

| 阈值跃迁 | 数据变化 | cF1 增益 |
|---------|---------|----------|
| 0.92→0.95 (V11→V12) | −19.4% | **+2.06pp**（爆炸式）|
| 0.95→0.96 (V12→V13) | −16.5% | **+0.42pp**（边际递减）|

- V11→V12 大幅提升，因为 0.92 阈值切得太少（仅 8.6%），多数噪声仍在
- V12→V13 增速放缓，但**仍未饱和**，可继续上探 0.965（949 paraphrase）
- 单 seed 也提升说明 0.96 不仅靠 ensemble 多样性，**模型本身也学得更好**：噪声越少，每个 seed 收敛得越好

---

### 5.15 V14：DeepSeek Paraphrase + COMET ≥ 0.965（❌ 过度过滤，找到甜区上限）

**动机**：V13 (0.96) 仍未饱和，继续上探 0.965 看是否可继续提升或开始回退。

**配置**：
- 训练数据：1600 原始 + 949 经 COMET ≥ 0.965 过滤的 paraphrase = **2549 条**
- Paraphrase 占比首次低于 1:1.7（vs 原始 1600）
- 丢弃 67.9% paraphrase，比 V10 减 44%

**单 seed 完整对比**：

| Seed | V13 (0.96) | V14 (0.965) | Δ |
|------|-----------|-------------|---|
| 42 | 0.3117 | 0.2967 | −1.50pp |
| 123 | 0.2974 | 0.2867 | −1.07pp |
| 456 | 0.2964 | 0.2928 | −0.36pp |
| 789 | 0.3080 | 0.2944 | −1.36pp |
| 1024 | 0.3067 | 0.2996 | −0.71pp |
| **Avg** | **0.3040** | **0.2940** | **−1.00pp** |

**5/5 seeds 全部回退**。

**Ensemble 评测**（vote=2 + MeCab）：

| 指标 | V13 | V14 | Δ |
|------|-----|-----|---|
| cPrecision | 0.3331 | 0.3113 | −2.18pp |
| cRecall | 0.3834 | 0.3789 | −0.45pp |
| **cF1** | **0.3565** | **0.3418** | **−1.47pp** ⚠ |

**Underfitting 证据**：
- V14 多个 seed 跑到 epoch 21-29 才早停（vs V13 的 13-22），收敛变慢
- Precision 下降明显（−2.18pp）：模型缺乏足够语料学清楚边界，预测不准
- Recall 也下降：覆盖率不足

**完整阈值-效应曲线**：

| 阈值跃迁 | 数据变化 | cF1 增益 |
|---------|---------|----------|
| 0.92→0.95 | −19.4% | **+2.06pp**（爆炸式增长）|
| 0.95→0.96 | −16.5% | **+0.42pp**（边际递减）|
| **0.96→0.965** | **−11.9%** | **−1.47pp**（**回退**）|

**结论：V13 (COMET ≥ 0.96) 是 jpn_hotel paraphrase 数据增强的最佳过滤阈值**。

倒 U 型曲线表明：
- 过滤过少（V10/V11）→ 噪声仍在，模型受干扰
- 过滤适度（V12/V13）→ 噪声去除 + 数据规模够用 → 双赢
- 过滤过多（V14）→ 数据不足以泛化 → underfitting

这给后续工作提供了清晰指引：**质量过滤是必要工序，但需要找到具体任务的最优阈值。在 V10 paraphrase 上是 0.96。**

---

### 5.16 P0 优化：V13 vote threshold 重调（✅✅✅ 零成本 +2pp）

**动机**：V13 沿用 V6 时代的 vote=2，但 V13 的 5 seeds 经 COMET 过滤后预测分布已显著不同，原 vote 阈值未必最优。

**方法**：直接在已训好的 V13 checkpoints 上跑 vote × pair_threshold sweep，无需重训。

**Vote sweep（pair=0.3 固定）**：

| vote | cP | cR | **cF1** |
|------|-----|-----|---------|
| 1 | 0.2371 | 0.4097 | 0.3004 |
| 2 (旧) | 0.3331 | 0.3834 | 0.3565 |
| **3 (新)** | **0.3946** | **0.3599** | **0.3765** ✨ +2.00pp |
| 4 | 0.4200 | 0.3127 | 0.3585 |
| 5 | 0.4825 | 0.2757 | 0.3509 |

**Pair sweep（vote=3 固定）**：

| pair | cF1 |
|------|------|
| 0.2 | 0.3765 |
| 0.3 | 0.3765 |
| 0.4 | 0.3765 |
| 0.5 | 0.3765 |

**Pair_threshold 完全无影响**——4 个值都得到完全一致的 0.3765，说明在 vote=3 + V13 数据下，pair 阶段已被 vote 阶段完全主导。

**核心洞察**：
1. **零成本提升**：仅改 1 个推理参数（vote=2 → 3），无需重训，cF1 +2.00pp
2. **vote 最优值随数据质量变化**：V6 时代 vote=2 最优；V13 因 COMET 过滤后 seeds 间一致性更高，vote=3 更严格反而 Precision 大涨
3. **Precision/Recall 折衷**：vote=3 大幅提 Precision（+6.15pp），仅小降 Recall（−2.35pp），净 +2.00pp
4. **教训**：每次重大模型/数据改动后都应**重新调 vote 阈值**（应作为 pipeline 标准流程）

**新最佳**：cF1 = **0.3765**（V13 + vote=3 + MeCab + dual-layer ensemble）

---

### 5.17 P1.4：V13 配方跨语言推广（V13/V13b/V13c）

详见目录顶部 [P1.4 跨语言推广](#p14-跨语言推广-v13-配方deepseek-paraphrase--comet-filter--vote-sweep) 章节。

**5 语言成果**（4/5 提升 V6）：

| 语言 | V6 | 最佳 V13/b/c | Δ |
|------|-----|--------------|---|
| jpn | 0.3312 | V13: 0.3765 | +4.53% |
| rus | 0.4520 | V13: 0.4724 | +2.04% |
| ukr | 0.4674 | V13b: 0.5068 | +3.94% |
| tat | 0.4538 | V6 持平 | 0% |
| zho_laptop | 0.3526 | V13c: 0.3791 | +2.65% |
| **平均** | **0.4114** | — | **+2.63pp** |

**核心论点**：每个语言的最佳增强配方不同，由 (BT 可用性, paraphrase 质量, V6 baseline, 数据规模) 决定。

---

### 5.18 V15-jpn：XLM-R-large encoder 升级（✅✅✅ 单 seed +7.52pp）

**动机**：mDeBERTa-v3-base (~270M) 是 jpn 当前 encoder。`mDeBERTa-v3-large` 不存在，可用替代品中 **`xlm-roberta-large` (~560M)** 是多语言 large 黄金标准。

**配置**：
- encoder: `xlm-roberta-large`
- batch_size: 4（base 是 8，防 OOM）
- 数据：V13-jpn (1600 原始 + 1293 COMET≥0.96 paraphrase = 2893)
- 其他超参与 V13 一致
- pretrain: 仍从 `v5_zho_restaurant_seed1024` checkpoint 迁移

**单 seed 完整对比**：

| Seed | V13 (base) | **V15 (large)** | Δ |
|------|-----------|-----------------|-----|
| 42 | 0.3117 | **0.3777** | **+6.60pp** |
| 123 | 0.2974 | **0.3725** | **+7.51pp** |
| 456 | 0.2964 | **0.3821** | **+8.57pp** |
| 789 | 0.3080 | **0.3821** | **+7.41pp** |
| 1024 | — | **0.3817** | — |
| **Avg** | 0.3040 | **0.3792** | **+7.52pp** ✨ |

**5/5 seeds 全部大幅提升** —— XLM-R-large 真实增益，非噪声。

**Vote sweep（ensemble）**：

| vote | cF1 |
|------|------|
| 2 | 0.3724 |
| **3 (best)** | **0.3924** |
| 4 | 0.3898 |
| 5 | 0.3920 |

**Ensemble 评测**：cF1 = **0.3924**（V13 best 0.3765，**+1.59pp**；V6 0.3312，**+6.12pp**）。

**历史 large 翻车的原因分析**：之前 V3-CRF + DeBERTa-v3-large 失败（−7.4%）的差异：
- 那次：英语 DeBERTa-v3-large 配 CRF 架构（CRF 部分不适配 large 表征）
- 这次：多语言 XLM-R-large 配 V13 paraphrase 数据 + Rethink 架构（无 CRF）

**注**：XLM-R-large 与 mDeBERTa 是不同 encoder（tokenizer/预训练不同），不算严格"scaling test"。但是当前可用的多语言 large 中最自然的升级。

---

### 5.19 V15 + Top-k Span Decoding（✅ 单 encoder 内最佳 cF1=0.3962）

**动机**：诊断 V15 的瓶颈在 Recall（cR=0.36 vs cP=0.40）。BIO argmax 错失置信度接近的 borderline spans。

**实现**：[src/models/dimabsa_model.py](src/models/dimabsa_model.py) `SpanExtractor.extract_spans()` 新增：
- 保留 argmax 结果（baseline）
- 额外：对每个 token 计算 softmax 概率，若 B-ASP / B-OPN 概率 ≥ `topk_threshold`（即使 argmax 是 O），生成候选 span
- 继续扩展 I-* 当其概率 ≥ `topk_threshold * 0.5`
- 不与已有 span 重叠

仅推理时启用，**不需重训**。CLI: `--span_topk_threshold 0.15`。

**Threshold sweep（V15, vote=3, ensemble）**：

| topk | cP | cR | **cF1** | Δ vs V15 baseline |
|------|-----|-----|---------|-------------------|
| 0（argmax, baseline）| ~0.40 | ~0.36 | 0.3924 | — |
| **0.15** | 0.3807 | **0.4131** | **0.3962** | **+0.38pp** 🏆 |
| 0.20 | 0.3822 | 0.4105 | 0.3958 | +0.34pp |
| 0.25 | 0.3822 | 0.4053 | 0.3934 | +0.10pp |
| 0.30 | 0.3865 | 0.4024 | 0.3943 | +0.19pp |

**核心发现**：
- **Recall 大涨 +5pp**（0.36 → 0.41），Precision 仅小降 −1.4pp
- 0.15 为甜区：进一步降低阈值会引入太多 FP

**Jpn 当前最佳完整配置**：
- Encoder: XLM-R-large
- 数据: 1600 原始 + 1293 paraphrase (COMET ≥ 0.96)
- Transfer: zho_restaurant_seed1024
- Top-k span: threshold 0.15
- Ensemble: 5 seeds, vote=3, dual-layer
- Post-processing: MeCab v1
- **→ cF1 = 0.3962**

---

### 5.20 失败实验集（V15 之后探索未带来提升的尝试）

V15 (0.3924) → V15+topk (0.3962) 之后，又尝试了 4 个改进方向，**全部失败**。详细记录便于论文 discussion / 后续不重复踩坑。

#### 5.20.1 V13 + V15 Cross-Encoder Ensemble（❌ −2.41pp）

**思路**：10 个 seeds = 5 V13 (mDeBERTa) + 5 V15 (XLM-R-large)，跨 encoder 多样性投票。

**实现**：[scripts/cross_encoder_ensemble.py](scripts/cross_encoder_ensemble.py)
- 每个 seed 用自己的 tokenizer/encoder 独立推理
- 收集所有预测后投票

**Vote sweep**：

| vote | cP | cR | **cF1** |
|------|-----|-----|---------|
| 4 | 0.2870 | 0.3863 | 0.3293 |
| 5 | 0.3490 | 0.3606 | 0.3547 |
| 6 | 0.4694 | 0.3082 | 0.3721 |
| 7 | 0.5196 | 0.2869 | 0.3697 |

**全部失败**，最佳 0.3721 vs V15+topk 0.3962（−2.41pp）。

**失败原因**：
- V13 (0.3765) 是 V15 (0.3924) 的**严格升级**版（更大模型 + top-k），不是"不同视角"的多样性
- vote=4 太低 → V13 噪声主导 → Precision 暴跌 (0.287)
- vote=6+ 太严 → V13/V15 各自的高置信集合重合度低 → Recall 暴跌 (0.31)

**教训**：cross-encoder ensemble 仅在两个 encoder 各有所长时有效。当一个是另一个的纯升级时，加入弱模型只会拖累。

#### 5.20.2 Per-Category Vote Calibration（❌ 无正收益）

**思路**：模型在 HOTEL#COMFORT (28 pred vs 14 gold)、ROOMS#QUALITY (12 vs 5) 等 category **过度预测**。对这些 category 设更高 vote 阈值，对 under-predicted category 设更低阈值。

**实现**：[scripts/per_category_vote.py](scripts/per_category_vote.py) 新增 `--over_categories`, `--over_vote`, `--under_categories`, `--under_vote`。

**Configs tested**:

| Config | cP | cR | cF1 |
|--------|-----|-----|------|
| Per-cat baseline (vote=3 all) | 0.3694 | 0.4131 | 0.3900 |
| Cfg1: 3 over-pred → vote=4 | 0.3783 | 0.4105 | 0.3937 |
| Cfg2: HOTEL#COMFORT only → vote=4 | 0.3749 | 0.4131 | 0.3931 |
| Cfg3: under-pred → vote=2 | 0.3546 | 0.4131 | 0.3816 |

**失败原因**：
- per-cat 脚本本身比 inference.py 内置 voting **低 0.62pp**（VA averaging / tie-breaking 差异）
- 即使 Cfg1 比 per-cat baseline 高 +0.37pp，绝对值 0.3937 仍低于原始 V15+topk 0.3962
- 真正实施需 refactor inference.py（1-2 天工程量），收益 ≤ +0.5pp，性价比不高

#### 5.20.3 MeCab v2：Opinion Adverb Extension（❌ −1.02pp）

**思路**：扩展 opinion span 包含前置副词（`とても親切`、`かなり良い`）。统计显示 5.8% 的 gold opinion 含 intensifier。

**v2 实现（宽泛规则）**：若 opinion 前是任何 `副詞` POS，向左扩展。
- 结果：cF1 = **0.3860**（−1.02pp）

**诊断失败原因**：MeCab 把许多 opinion 词本身（`広々`、`よく`、`ゆっくり`）标为 `副詞`，导致误识别相邻独立 opinion 为修饰：

| 原 opinion | v2 扩展后 | 实际 gold | 错误类型 |
|-----------|----------|----------|----------|
| 満足 | **良く満足** | 満足 | "良く" 是另一个独立 opinion |
| あずましく | **広々とあずましく** | あずましく | "広々" 是独立 opinion |
| よかっ | **とてもよかっ** | よかった | gold 不含 "とても" |

**v2.1 修复（intensifier 白名单）**：只对 {とても, かなり, 本当に, ...} 等 22 个真正 intensifier 扩展：
- 结果：cF1 = **0.3962**（**完全无变化**）
- **新触发扩展仅 1 次**且错误
- **说明模型 BIO + top-k 已经正确捕获 intensifier**，post-processing 在这层无新信息可加

#### 5.20.4 LLM Fusion 候选融合（❌ 全部策略失败）

**思路**：让 DeepSeek 给 jpn 抽取 (Aspect, Category, Opinion, VA) 候选 → 合并到 V15 预测。

**实现**：
- [scripts/llm_extract_quads_jpn.py](scripts/llm_extract_quads_jpn.py)：调用 DeepSeek 抽取（5-shot, valid category list, JSON output）
- [scripts/merge_llm_v15.py](scripts/merge_llm_v15.py)：4 个 baseline 策略 + 4 个 smart_overlap 策略

**步骤 1：模型选择**

| 模型 | 输出 quads/sample | 失败率 |
|------|-------------------|--------|
| deepseek-v4-pro | 0.59 | 51%（reasoning 模型，max_tokens=1500 不足）|
| **deepseek-chat** | **2.03**（接近 gold 1.82）| **0%** |

deepseek-v4-pro 内部 reasoning 占用 token，max_tokens=4096 仍不够稳定。**deepseek-chat 是正确选择**。

**步骤 2：合并策略对比（deepseek-chat, 200 dev samples, 406 LLM quads）**

| 策略 | cP | cR | **cF1** | vs baseline |
|------|-----|-----|---------|-------------|
| **V15+topk baseline** | 0.3807 | 0.4131 | **0.3962** | — |
| llm_only | 0.2260 | 0.2521 | 0.2383 | −15.79pp |
| union (all LLM added) | 0.2516 | **0.4874** | 0.3319 | −6.43pp |
| intersection (LLM ∩ V15) | **0.6682** | 0.1762 | 0.2789 | −11.73pp |
| smart_any (LLM 加 if asp 或 opn 共享) | 0.2934 | 0.4780 | 0.3636 | −3.26pp |
| smart_opinion (LLM 加 if opn 共享) | 0.3298 | 0.4295 | 0.3731 | −2.31pp |
| smart_aspect (LLM 加 if asp 共享) | 0.3152 | 0.4729 | 0.3783 | −1.79pp |
| smart_both (LLM 加 if asp+opn 都共享) | 0.3617 | 0.4243 | **0.3905** | **−0.57pp** |

**关键洞察**：
- **intersection 揭示信号**：LLM+V15 一致时 cP=0.668（vs V15 单独 0.38），说明"两者同意"是强质量信号
- **但 smart_both（同时 overlap aspect+opinion）仍只到 0.3905**：即使 span 一致，**LLM 的 category 决策不可靠**——V15 给 cat A，LLM 给同一 span cat B，多数 V15 是对的
- **DeepSeek-chat zero-shot 不够好做无监督 ABSA 候选生成**：~22% precision

**让 LLM Fusion 真正反超需要**：
1. **更强 LLM**（GPT-4o / Claude / GPT-OSS 120B）— 无 API
2. **模型手术**（让 V15 给 LLM 候选打 category 分）— 3-5 天工程
3. **多轮 LLM 投票**（3-5 次 DeepSeek 调用 + 一致性过滤）— 成本 3-5x，效果未必

#### 失败实验汇总（V15+topk 之后）

| # | 实验 | cF1 | Δ vs 0.3962 |
|---|------|------|-------------|
| 1 | V13+V15 cross-encoder | 0.3721 | −2.41 |
| 2 | Per-category vote calibration | 0.3937 | −0.25 |
| 3 | MeCab v2 (opinion adverb) | 0.3860 | −1.02 |
| 4 | MeCab v2.1 (intensifier 白名单) | 0.3962 | 0 |
| 5 | LLM Fusion smart_both（最佳）| 0.3905 | −0.57 |

**Jpn 当前最佳**仍是 V15+top-k+MeCab v1+vote=3 = **cF1=0.3962**。

---

#### 5.20.5 Confidence-Weighted Ensemble（❌ 8-dataset mean −0.73pp）

**思路**：5 seeds 投票时，强 seed 加权更大（weight = dev cF1）。
**实现**：`scripts/confidence_weighted_ensemble.py`，threshold = vote_threshold × mean_weight（保持 vote=3 语义）。

| Dataset | Uniform | CW | Δ |
|---|---|---|---|
| jpn | 0.4500 | 0.4335 | **−1.65** |
| rus | 0.5267 | 0.5294 | +0.27 |
| zho_laptop | 0.4487 | 0.4357 | **−1.30** |
| ukr | 0.5068 | 0.5062 | −0.06 |
| tat | 0.4821 | 0.4885 | +0.64 |
| eng_restaurant | 0.6949 | 0.6927 | −0.22 |
| eng_laptop | 0.3035 | 0.2975 | −0.60 |
| zho_restaurant | 0.5669 | 0.5672 | +0.03 |
| **MEAN** | **0.4974** | **0.4901** | **−0.73** |

**为什么失败**：5 seeds 的 dev cF1 极接近（标准差 ≤ 0.015），权重差异不到 ±5%，weighting 信息差不足；但 threshold 在 weight 不均匀时**净变严**（弱 seeds 联合投票总权重 < threshold），损失 recall。

**论文意义**：homogeneous ensemble（同 base + 同数据 + 不同 seed）的 CW 不适用；CW 应当用于 heterogeneous ensemble（不同 base 或不同 data，需重新调参）。

---

#### 5.20.6 Uncertainty-Weighted Multi-Task Loss / UML（❌ 不可靠）

**思路**：借鉴 LogSigma (Subtask 1) — 学习 per-task log σ²，自动平衡 span/pair/entity/attr/va/consistency 6 个 loss。Loss = Σ (0.5 e^{-s_i} L_i + 0.5 s_i)。

**实现**：在 `DimASQPModel` 加 `use_uml` flag + 6 个 `nn.Parameter` log_sigma；config 加 `model.use_uml: true`。代码改动 ~30 行。

**Sanity check（seed 42 each）**：

| Dataset | V16 (fixed weights span=2 va_arousal=2) | V17 (UML) | Δ |
|---|---|---|---|
| jpn (no_aug, deberta-v2-jp) | 0.4109 | 0.3892 | **−2.17** ❌ |
| eng_restaurant (deberta-v3-large) | 0.6759 | 0.6772 | +0.13 ✅ marginal |

**为什么失败**：
1. V16 的 fixed weights（span=2.0, pair=1.0, ent=1.0, attr=1.0, va_arousal=2.0）是人工调过的强先验；UML 学到的 sigma 倾向"趋近均匀"，丢失任务结构 prior
2. log_sigma 初始化 0 → 初始 weight = 0.5 vs V16 的 1-2，训练早期 effective LR 缩小 ~3 倍
3. 正则项 +0.5 s_i 持续推 s_i ↓ → exp(-s_i) ↑ → loss 项放大；jpn 任务间方差大时容易陷入"VA loss 被压制"局部解

**未扩展至 5 seeds**：seed 42 已证不稳定，1/2 dataset 显著负，不值得 ~12h 训练换 ±0.1 pp 的不确定收益。

**论文意义**：UML 适用于**任务先验未知**的 multi-task 场景（如 LogSigma 在 Subtask 1 的 V/A 双回归任务上是 2 个对称任务）。但 DimASQP 的 6 个 loss 有强结构差异，人工权重 > 学习权重。

---

#### 5.20.7 V18 eng_laptop Paraphrase Augmentation（❌ −2.73pp ensemble）

**思路**：沿用 V13 配方（DeepSeek-chat paraphrase + COMET 过滤），尝试改善 eng_laptop 的小数据短板（4076 样本 × 24 entity 类）。

**数据**：
- DeepSeek-chat 生成 7504 paraphrases，92.1% acceptance
- COMET wmt22-comet-da 评分，**英文 score 分布偏低**（median 0.8537 vs jpn ~0.92），所以阈值 0.96 仅留 19 条 ⇒ 改用 **0.85 阈值**保留 ~50% (3960 条) 与 V13 jpn 比例对齐
- 最终 V13 train：4076 原 + 3960 paraphrase = **8036 样本**

**5-seed 结果**（V16 baseline = no augmentation）：

| seed | V18 (V13 aug) | V16 (no aug) | Δ |
|---|---|---|---|
| 42 | 0.2573 | 0.2943 | −3.70 |
| 123 | 0.2565 | 0.2929 | −3.64 |
| 456 | 0.2687 | 0.2887 | −2.00 |
| 789 | 0.2616 | 0.2874 | −2.58 |
| 1024 | 0.2697 | 0.2829 | −1.32 |
| **mean** | **0.2628** | **0.2892** | **−2.65** |
| **best ensemble** (vote=4 topk=0) | **0.2762** | **0.3035** | **−2.73** |

**为什么失败**：与 §7.6.3 jpn no_aug 发现一致 — **强 native encoder + 大类别空间** 上 paraphrase 引入的 semantic drift 净负面。eng_laptop 的 24 entity × 10 attribute = 240 (cat, attr) 组合**对类别漂移高度敏感**：合成数据中即使 aspect/opinion 文字保留，paraphrase 重写可能让原本"LAPTOP#DESIGN_FEATURES"的句子读起来更像"LAPTOP#GENERAL"，混淆分类边界。

**论文意义**：数据增强的边际效益不仅随 encoder 容量衰减（§7.6.3），还随**类别空间复杂度**反向放大噪声。eng_laptop 是这两个因素同时极端的 case。

---

### 5.24 V19 eng_laptop Encoder Sweep（✅ roberta-large +0.27pp ensemble）

**动机**：V18 paraphrase 失败后，转向 encoder 替换 — 也许 deberta-v3-large 不是 English ABSA 的最佳 base。

**3 个候选 smoke test (seed 42)**：

| Encoder | 参数 | seed 42 cF1 | vs V16 (0.2943) |
|---|---|---|---|
| `roberta-large` | 355M | **0.3024** | **+0.81** ✅ |
| `nghuyong/ernie-2.0-large-en` | 335M | 0.2948 | +0.05 持平 |
| `microsoft/deberta-v2-xxlarge` | 1.5B | OOM ❌ | — (24 GB GPU 不够) |

**roberta-large 扩到 5 seeds**：

| seed | V19 (roberta-large) | V16 (deberta-v3-large) | Δ |
|---|---|---|---|
| 42 | 0.3024 | 0.2943 | +0.81 |
| 123 | 0.2828 | 0.2929 | −1.01 |
| 456 | 0.3158 | 0.2887 | +2.71 |
| 789 | 0.2930 | 0.2874 | +0.56 |
| 1024 | 0.3145 | 0.2829 | +3.16 |
| **mean (single)** | **0.3017** | **0.2892** | **+1.25** |
| **best ensemble** (vote=4 topk=0) | **0.3062** | **0.3035** | **+0.27** |

**结论**：roberta-large 略优于 deberta-v3-large 在 eng_laptop 上。single-seed 收益 +1.25 pp 但 ensemble 收敛到 +0.27 pp（V16 ensemble 本来就强，新 encoder 多样性增益小）。

**eng_laptop submission 从 V16 切换到 V19（dev 0.3035 → 0.3062）**。

**遗憾**：deberta-v2-xxlarge (1.5B params) 因 GPU OOM 未测；理论上参数空间最适合 eng_laptop 的 24×10 类别 sparsity。需要 8-bit Adam / gradient checkpointing / batch=1 改造才能放下。

**论文意义**：和 §5.23 中 rus/ukr 的 encoder swap 行为类似 — 当 baseline encoder 已经较强（deberta-v3-large），同尺度 alternative encoder（roberta-large）只能给微弱的 ensemble 增量。**多样性是 ensemble 的关键，但同源模型的多样性已被 5-seed 吃完**。要真正突破，需要 cross-architecture diversity（如 mixed mDeBERTa+XLM-R+RoBERTa），但 §5.20.1 cross-encoder ensemble 已证此方向也失败。eng_laptop 性能瓶颈是**数据 + 类别复杂度的固有 sparsity**，不是 encoder 选择。

---

### 5.25 V20 eng_laptop = roberta-large + BT + paraphrase（✅ +0.53pp 累计）

**动机**：V18 (deberta-v3-large + paraphrase) 失败 (−2.73 pp)，但换 V19 roberta-large encoder 后 raw 数据已 +0.27pp。问 augmentation × encoder 是否有 **交互效应** — roberta-large 对 paraphrase 噪声是否更鲁棒？

**数据 V13b (V13b-style)**：
- 4076 raw + 8556 BT (MarianMT en→de/fr/zh, 自动 fuzzy validation) + 3960 paraphrase (DeepSeek + COMET 0.85) = **16592 samples** (4× 原始)
- BT 保留率：de 83.2%, fr 74.8%, zh 51.9%
- 配置：V19 yaml + train_file 替换

**5-seed 结果**：

| seed | V20 (rob + V13b) | V19 (rob + raw) | V16 (deberta + raw) |
|---|---|---|---|
| 42 | 0.3049 | 0.3024 | 0.2943 |
| 123 | 0.2935 | 0.2828 | 0.2929 |
| 456 | 0.2961 | 0.3158 | 0.2887 |
| 789 | 0.2988 | 0.2930 | 0.2874 |
| 1024 | 0.2968 | 0.3145 | 0.2829 |
| **mean (single)** | **0.2980** | **0.3017** | **0.2892** |
| **best ensemble** (vote=4 topk=0) | **0.3088** | **0.3062** | **0.3035** |

**关键 finding — Encoder × Augmentation 交互效应**：

| (Encoder, Augmentation) | single mean | ensemble best | aug Δ |
|---|---|---|---|
| deberta-v3-large + raw | 0.2892 (V16) | 0.3035 | — |
| **deberta-v3-large + V13 aug** | 0.2628 (V18) | 0.2762 | **−2.65 / −2.73** ❌ |
| roberta-large + raw | 0.3017 (V19) | 0.3062 | — |
| **roberta-large + V13b aug** | 0.2980 (V20) | **0.3088** | **−0.37 / +0.26** ✅ |

**解读**：
- deberta-v3-large 对 paraphrase 噪声**脆弱**（−2.7 pp 两个 metric 都负）
- roberta-large 对 augmentation **鲁棒**（single seed 微弱负 −0.37，但 ensemble level 反而 +0.26 pp）
- 推测：roberta 的 BPE + sentencepiece 预训练 vs deberta-v3 的 ELECTRA-style 预训练，前者对**词序变化和 paraphrastic style** 的容忍度更高

**eng_laptop 最佳 = V20**（dev 0.3088）。Submission 已切换 V20，V19/V16 备份保留。

**全局视角**：从 V13 时代的 augmentation +2-4 pp，到 V16 时代衰减为持平/负，到 V20 在 roberta-large 上微弱回正。**augmentation 不是过时，而是高度 encoder-dependent**。论文可以写：「The marginal utility of synthetic data augmentation is not monotonically decaying with encoder capacity; rather it interacts non-trivially with encoder pretraining choice — RoBERTa-large benefits while DeBERTa-v3-large is harmed by the same augmentation set」。

---

### 5.21 Jpn 跨基线对比

| 模型 | jpn_hotel dev cF1 | 备注 |
|------|-------------------|------|
| V6 | 0.3312 | baseline |
| V13 | 0.3765 | +paraphrase+COMET |
| V15 | 0.3924 | +XLM-R-large |
| **V15+top-k** | **0.3962** | **+top-k spans** ✨ |
| Official Llama-3.3 70B | 0.3577 | ✅ 我们高 +3.85pp |
| Official GPT-OSS 120B | 0.4151 | ❌ 我们低 −1.89pp |

**结论**：超过 Llama-3.3 70B 较显著（+3.85pp），但未超过 GPT-OSS 120B（−1.89pp）。差距主要在 LLM 对**隐式 aspect / 罕见 category** 的识别能力，无法通过单纯架构/数据增强方法弥补。

---

### 5.22 V15 跨语言推广（rus / zho_laptop / ukr / tat）

**动机**：jpn 上 V15 (XLM-R-large) 单 seed +7.52pp，是否能推广到其他 4 个语言？

**配置**：保留各语种 V13/V13b/V13c 数据配方 + Rethink 架构 (无 CRF)，仅切 encoder 为 `xlm-roberta-large`。pretrain 沿用 V13 的 `v5_zho_restaurant_seed1024`（**注：训练日志显示 `Loaded & averaged 0 encoder parameters` —— mDeBERTa-base 768d 与 XLM-R-large 1024d shape mismatch，pretrain 实际是 no-op，所有 V15 都从 HF `xlm-roberta-large` 冷启动**）。

**单 seed cF1 完整结果**（V15 vs 对应最佳前代 V13/V13b/V13c）：

| seed | rus V15 | rus V13 | Δ | zho_laptop V15 | zho_laptop V13c | Δ | ukr V15 | ukr V13b | Δ | tat V15 | tat V13 | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.4817 | 0.4610 | +2.07 | 0.3562 | 0.3424 | +1.38 | 0.4423 | 0.4410 | +0.13 | 0.3984 | 0.3738 | +2.46 |
| 123 | 0.4469 | 0.4388 | +0.81 | 0.3446 | 0.3533 | −0.87 | 0.4239 | 0.4472 | −2.33 | 0.4282 | 0.4213 | +0.69 |
| 456 | 0.4785 | 0.4483 | +3.02 | 0.3443 | 0.3484 | −0.41 | 0.4643 | 0.4674 | −0.31 | 0.4047 | 0.4112 | −0.65 |
| 789 | 0.4646 | 0.4760 | −1.14 | 0.3427 | 0.3359 | +0.68 | 0.4546 | 0.4736 | −1.90 | 0.3942 | 0.4288 | −3.46 |
| 1024 | 0.4780 | 0.4487 | +2.93 | 0.3432 | 0.3492 | −0.60 | 0.4579 | 0.4415 | +1.64 | 0.3979 | 0.3849 | +1.30 |
| **mean** | **0.4699** | **0.4546** | **+1.53** | **0.3462** | **0.3458** | **+0.04** | **0.4486** | **0.4541** | **−0.55** | **0.4047** | **0.4040** | **+0.07** |
| **best** | **0.4817** | **0.4760** | **+0.57** | **0.3562** | **0.3533** | **+0.29** | **0.4643** | **0.4736** | **−0.93** | **0.4282** | **0.4288** | **−0.06** |

**结论：encoder upgrade 跨语言收益高度不均**

| 语种 | V13/V13b/V13c baseline | V15 单 seed avg | Δ | 结论 |
|---|---|---|---|---|
| **jpn** | 0.3312 → 0.3765 | 0.3924 | **+7.52** | ✅✅✅ 显著突破 |
| **rus** | 0.4546 | 0.4699 | **+1.53** | ✅ 真实增益 |
| zho_laptop | 0.3458 | 0.3462 | +0.04 | ⚠️ 持平（饱和） |
| ukr | 0.4541 | 0.4486 | −0.55 | ❌ 略弱 |
| tat | 0.4040 | 0.4047 | +0.07 | ⚠️ 持平（高方差） |

**模式发现**：
1. **baseline 低 + 中等数据集 → encoder 收益最大**（jpn 0.33 → +7.52pp；rus 0.45 → +1.53pp）
2. **baseline 高 + 大数据集 → 收益饱和**（zho_laptop 5500+ 样本，mDeBERTa 中文预训练 SOTA，已贴近极限）
3. **baseline 高 + 数据增强已耗尽空间 → 反而轻微下降**（ukr V13b 已 BT+paraphrase 叠加，模型容量不是瓶颈）
4. **tat 高方差** (5 seed std ~1.2pp)：seed 42/123/1024 显著提升，seed 789 大跌 (−3.46pp)，整体被一个 outlier 拖平。说明 XLM-R-large 在小数据集上**初始化敏感**，ensemble 才能稳定增益。

**XLM-R-large 不是普适升级**：
- 跨语言通才 (XLM-R) vs 中文专才 (mDeBERTa) 各有所长
- "大模型 = 更好"假设在已经收敛的语言上不成立
- jpn 之所以收益巨大，可能因 mDeBERTa-base 对日语形态学（粘着语 + 假名）支持有限

**ENSEMBLE + VOTE + TOP-K SWEEP（5 seeds × 3 vote × 4 topk = 12 configs/lang）**：

| Lang | V15 best | Config | V13/V13b/V13c ensemble baseline | Δ |
|---|---|---|---|---|
| **jpn** | **0.3962** | vote=3 topk=0.15 | V13 0.3765 | **+1.97 pp** ✅✅✅ |
| **rus** | **0.5267** | vote=4 topk=0.10 | V13 0.4724 | **+5.43 pp** 🎉🎉🎉 |
| **tat** | **0.4821** | vote=3 topk=0.20 | V6 0.4538 / V13 0.4040 | **+2.83 pp / +7.81 pp** 🎉🎉 |
| **zho_laptop** | **0.3904** | vote=3 topk=0.15 | V13c 0.3791 | **+1.13 pp** ✅ |
| ukr | 0.4957 | vote=4 topk=0.0 | V13b 0.5068 | **−1.11 pp** ❌ |

**反转发现**：
- 单 seed 看似 zho_laptop/tat/ukr 平均"持平或更差"，但 **ensemble 之后 3/4 都反超**。
- V15 多样性更高（large model 在小数据上方差大），**ensemble vote 把噪声转化为多样性优势**，单 seed metric 严重低估了 V15 真实潜力。
- 唯一例外 ukr：V13b 已经用 BT+paraphrase 叠加榨干了多样性预算，V15 编码器单独升级反不及。

**Best vote/topk 模式**：
- `vote=3` 在小 baseline 任务（jpn, tat, zho_laptop）最优 — 中等严格度平衡 P/R
- `vote=4` 在 Cyrillic 语种（rus, ukr）最优 — 数据较多 + 高 baseline 时严格度提高质量
- `topk` 普遍有 0.1-0.2pp 微正贡献（jpn/tat/zho_laptop），rus 0.10 最优，ukr 反而 topk=0 最优

**Per-language 最终最佳配置**：

| 语种 | 推荐 | best dev cF1 | 配置 | 备注 |
|---|---|---|---|---|
| **jpn** | V15 ensemble | **0.3962** | vote=3, topk=0.15 | 超 Llama-3.3 70B (+3.85pp) |
| **rus** | V15 ensemble | **0.5267** 🏆 | vote=4, topk=0.10 | 跨语言绝对最佳 |
| **tat** | V15 ensemble | **0.4821** | vote=3, topk=0.20 | 单 seed 高方差，ensemble 解决 |
| **zho_laptop** | V15 ensemble | **0.3904** | vote=3, topk=0.15 | 反超 V13c |
| **ukr** | V13b ensemble | **0.5068** | (V13b 配方, vote=3) | V15 仍弱 1.1pp |

**意义**：
- ✅ V15 在 4/5 语种成为 single best；**仅 ukr 保留 V13b**
- ✅ 推翻了 §5.22 早期版本「encoder scaling 不普适」的结论 — 是 single-seed metric 误导
- 🔑 **结论修正**：encoder scaling 的真实价值需要 ensemble 评估才能体现；single-seed 高方差是 large model 的特征不是 bug
- 🚀 5 语种合计平均 cF1 = **0.4604**（jpn 0.3962 + rus 0.5267 + zho_laptop 0.3904 + ukr 0.5068 + tat 0.4821）/ 5，**vs P1.4 时代平均 0.4322 → +2.82 pp**

---

### 5.23 V16：Language-Native Encoders（最终架构升级）

**动机**：§5.22 发现 V15 (XLM-R-large) 是跨语言通才，per-lang 收益不均（jpn +7.5, rus +1.5, zho_laptop +0.04, ukr −0.55, tat +0.07）。借鉴 LogSigma 思路，给每语种选 **best-fit native pretrained encoder**。

**配置**（仅替换 encoder，其余 V15 设置不变）：

| 语种 | encoder | 参数 | tokenizer 类型 | 备注 |
|---|---|---|---|---|
| jpn | `ku-nlp/deberta-v2-large-japanese` | 320M | Fast (DebertaV2) | 京大 NLP 实验室 |
| zho_laptop | `hfl/chinese-roberta-wwm-ext-large` | 330M | Fast (BERT) | 哈工大讯飞，全词 mask |
| rus | `ai-forever/ruRoberta-large` | 355M | Fast (RoBERTa) | Sber 俄语大模型 |
| ~~ukr~~ | 跳过 | — | — | V13b 已最佳 |
| ~~tat~~ | 跳过 | — | — | Turkic 系无 large 模型可选 |

**踩坑记录**：第一次尝试 `cl-tohoku/bert-large-japanese-v2` 失败 — 用 MeCab Python tokenizer，**不支持 `return_offset_mapping`**（DimASQP span 提取强依赖 offset）。换 `ku-nlp/deberta-v2-large-japanese`（带 Fast tokenizer）成功。

**单 seed cF1（5 seeds per lang）**：

| seed | jpn V16 | jpn V15 | Δ | zho_lap V16 | zho_lap V15 | Δ | rus V16 | rus V15 | Δ |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.4019 | 0.3924 | +0.95 | 0.3970 | 0.3562 | +4.08 | 0.4433 | 0.4817 | −3.84 |
| 123 | 0.4035 | 0.3924 | +1.11 | 0.3861 | 0.3446 | +4.15 | 0.4450 | 0.4469 | −0.19 |
| 456 | 0.4128 | 0.3924 | +2.04 | 0.3941 | 0.3443 | +4.98 | 0.4534 | 0.4785 | −2.51 |
| 789 | 0.3926 | 0.3924 | +0.02 | 0.4010 | 0.3427 | +5.83 | 0.5028 | 0.4646 | +3.82 |
| 1024 | 0.4023 | 0.3924 | +0.99 | 0.3917 | 0.3432 | +4.85 | 0.4912 | 0.4780 | +1.32 |
| **mean** | **0.4026** | **0.3924** | **+1.02** | **0.3940** | **0.3462** | **+4.78** | **0.4671** | **0.4699** | **−0.28** |

**Ensemble + vote + top-k sweep（5 seeds × 3 vote × 4 topk = 12 configs/lang）**：

| Lang | V16 best | Config | V15 best | Δ | 推荐 |
|---|---|---|---|---|---|
| **jpn** | **0.4466** 🏆 | vote=4, topk=0 | 0.3962 | **+5.04** 🎉 | **V16** |
| **zho_laptop** | **0.4487** 🏆 | vote=4, topk=0.10 | 0.3904 | **+5.83** 🎉 | **V16** |
| **rus** | 0.5152 | vote=3, topk=0 | 0.5267 | −1.15 | **V15** (XLM-R 更优) |

**关键发现**：
1. **zho_laptop encoder upgrade 巨大收益**（+5.83 pp）：中文专用全词 mask 比通才 XLM-R-large 在中文领域文本上显著优秀
2. **jpn deberta-v2 同样巨大**（+5.04 pp）：京大 DeBERTa 的 disentangled attention + 日语 morphology 适配
3. **rus 反而轻微下降**（−1.15 pp）：俄语本就是 XLM-R 预训中 high-resource 语种，已经接近最优；ruRoberta-large 没有显著领域优势
4. **Tokenizer 兼容性是隐患**：MeCab/Jumanpp 基于的日语模型用 Python slow tokenizer，DimASQP 框架不兼容（offset_mapping 依赖）。选模型时必须先验证 `tokenizer.is_fast`

**5 语种最终最佳配置 + dev cF1**：

| 语种 | 推荐版本 | encoder | dev cF1 | Test SOTA (top 1) | Δ vs SOTA |
|---|---|---|---|---|---|
| **jpn** | V16 | ku-nlp/deberta-v2-large-japanese | **0.4466** | PALI 0.4252 | **+2.14** (反超 🎉) |
| **rus** | V15 | xlm-roberta-large | **0.5267** | PAI 0.5599 | −3.32 |
| **zho_laptop** | V16 | hfl/chinese-roberta-wwm-ext-large | **0.4487** | NYCU 0.4824 | −3.37 |
| **ukr** | V13b | microsoft/mdeberta-v3-base | **0.5068** | PAI 0.5437 | −3.69 |
| **tat** | V15 | xlm-roberta-large | **0.4821** | Takoyaki 0.4736 | **+0.85** (反超 🎉) |
| **5-lang mean** | per-lang mix | mixed | **0.4822** | — | — |

**意义**：
- ✅ **jpn / tat 在 dev 上反超 test 1st**（注意 dev≠test，最终需要 test 集验证）
- ✅ 5 语种平均 cF1 从 V15 时代 0.4604 → **0.4822** (+2.18 pp)
- ✅ 验证了**「per-lang best encoder + ensemble + per-lang sweep」**才是最优策略
- ⚠️ 局限：zho_laptop 距 NYCU SOTA 仍差 3.4 pp（NYCU 可能用了 unpublished 中文领域适应 pretrain）；ukr 距 PAI SOTA 差 3.7 pp（PAI 论文未公开方法）
- 🎯 **5 语种 dev mean 0.4822 已具有相当强的 leaderboard 竞争力**

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
| V12 V10 + COMET≥0.95 (jpn) | +1866 中度过滤 paraphrase | 0.3523 | jpn +2.11% vs V6 | ✅✅ 大幅突破 |
| V13 V10 + COMET≥0.96 vote=2 (jpn) | +1293 最优过滤 paraphrase | 0.3565 | jpn +2.53% vs V6 | ✅✅✅ |
| V14 V10 + COMET≥0.965 (jpn) | +949 过度过滤 paraphrase | 0.3418 | jpn +1.06% vs V6 | ❌ 回退（甜区上限）|
| V13 + vote=3 重调 (jpn) | 零成本：仅改推理参数 | 0.3765 | jpn +4.53% vs V6 | ✅✅✅✅ |
| V15 (XLM-R-large, jpn) | encoder 升级到 xlm-roberta-large | 0.3924 | jpn +6.12% vs V6 | ✅✅✅✅✅ |
| **V15 + Top-k Span (jpn)** | **+top-k BIO decoding (topk=0.15)** | **0.3962** | **jpn +6.50% vs V6** | **✅✅✅✅✅ 当前最佳** 🏆 |
| ❌ V15+topk + Cross-encoder | V13+V15 共 10 seeds 投票 | 0.3721 | −2.41pp vs V15+topk | ❌ |
| ❌ V15+topk + LLM Fusion smart_both | DeepSeek 候选合并 | 0.3905 | −0.57pp vs V15+topk | ❌ |

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

### 7.6 Final V16 消融（基于 native encoder 配置）

§7.1–7.5 是 mDeBERTa-base 时代的消融。这里在 V16（per-language native encoder）上重做最关键的 3 个 ablation。

#### 7.6.1 Ensemble 贡献（5-seed vote）

| Dataset | best single seed | mean (5 seeds) | 5-seed ensemble best | Δ ens − mean |
|---|---|---|---|---|
| jpn_hotel | 0.4128 | 0.4026 | 0.4466 | **+4.40** |
| rus_restaurant | 0.4817 | 0.4699 | 0.5267 | **+5.68** |
| zho_laptop | 0.4010 | 0.3940 | 0.4487 | **+5.47** |
| ukr_restaurant | 0.4736 | 0.4541 | 0.5068 | **+5.27** |
| tat_restaurant | 0.4282 | 0.4047 | 0.4821 | **+7.74** |
| eng_restaurant | 0.6759 | 0.6728 | 0.6949 | **+2.21** |
| eng_laptop | 0.2943 | 0.2892 | 0.3035 | **+1.43** |
| zho_restaurant | 0.5234 | 0.5083 | 0.5669 | **+5.86** |
| **8-dataset MEAN** | **0.4614** | **0.4495** | **0.4970** | **+4.76** |

**结论**：5-seed ensemble 在 8 数据集上一致给出 **+1.43~+7.74 pp** 的提升，平均 **+4.76 pp**。这是论文最有力的 method-level 贡献。

#### 7.6.2 Rethink 模块（V16-eng_restaurant, deberta-v3-large, 5 seeds）

| seed | WITH Rethink | WITHOUT Rethink | Δ |
|---|---|---|---|
| 42 | 0.6759 | 0.6854 | +0.95 |
| 123 | 0.6697 | 0.6567 | −1.30 |
| 456 | 0.6740 | 0.6723 | −0.17 |
| 789 | 0.6715 | 0.6667 | −0.48 |
| 1024 | 0.6731 | 0.6643 | −0.88 |
| **mean (single)** | **0.6728** | **0.6691** | **−0.37** |
| **ensemble best** | **0.6949** | **0.6913** | **−0.36** |

**对比 §7.1 (mDeBERTa-base, V3 时代)**：Rethink **+11.29 pp** → 在 deberta-v3-large 上衰减到 **+0.36 pp**。

**结论**：Rethink 模块在小 encoder 上是核心贡献，在大 encoder 上**边际效益接近零**。强 encoder 一次性输出已经足够好，迭代精炼带来的提升有限。但仍是正贡献，**所有 V16 实验保留 Rethink**。

#### 7.6.3 Data Augmentation（V16-jpn, deberta-v2-japanese, 5 seeds）

| seed | WITH aug (V13: 2893) | WITHOUT aug (raw: 1600) | Δ |
|---|---|---|---|
| 42 | 0.4019 | 0.4109 | +0.90 |
| 123 | 0.4035 | 0.3967 | −0.68 |
| 456 | 0.4128 | 0.3940 | −1.88 |
| 789 | 0.3926 | 0.4055 | +1.29 |
| 1024 | 0.4023 | 0.3978 | −0.45 |
| **mean (single)** | **0.4026** | **0.4010** | **−0.16** |
| **ensemble best** | **0.4466** | **0.4500** | **+0.34** |

**对比 §5.18 (V13 时代, mDeBERTa)**：paraphrase + COMET ≥0.96 给 jpn **+4.53 pp** → 在 deberta-v2-japanese 上 ensemble 反而 **−0.34 pp**。

**结论**：数据增强在 mDeBERTa-base 时代有效，但在强 native encoder + 5-seed ensemble 下变成**轻微 noise**。ensemble 已经吃掉了 augmentation 的正则化好处，paraphrase 引入的语义漂移留下来。**最终 jpn submission 切换为 no_aug ensemble** (dev 0.4500 vs aug 0.4466)。

#### 7.6.4 Per-language Native Encoder（最有效的单一改动）

| 语种 | Universal (XLM-R-large) | Native | Δ ensemble |
|---|---|---|---|
| jpn | 0.3962 (V15) | 0.4500 (V16 deberta-v2-jp + no_aug) | **+5.38** |
| zho_laptop | 0.3904 (V15) | 0.4487 (V16 chinese-roberta) | **+5.83** |
| eng_restaurant | n/a (XLM-R not tried) | 0.6949 (V16 deberta-v3-large) | — |
| rus | 0.5267 (V15) | 0.5152 (V16 ruRoberta) | **−1.15** ❌ |

**结论**：Native encoder 在 zh / jp / eng 上是巨大 win (+5pp 量级)，在 rus 上反而轻微 hurt（俄语在 XLM-R 预训练中已是 high-resource）。**「先验证 → 再换」**比盲目升级更靠谱。

---

### 7.7 Final Ablation 总结表（论文用）

| Ablation | 在 mDeBERTa-base (V3) | 在 V16 native encoder | 解读 |
|---|---|---|---|
| Rethink 模块 | **+11.29 pp** (§7.1) | +0.36 pp ensemble | 大模型自带 ≈ 弱 Rethink |
| Data augmentation (paraphrase+COMET) | **+4.53 pp** (§5.18, jpn) | −0.34 pp ensemble (jpn) | ensemble 已吃掉 reg 好处 |
| 5-seed ensemble | +1.4 pp (§5.7, baseline) | **+4.76 pp** (§7.6.1, 8-dataset) | **encoder 越强，ensemble 收益越大** |
| Native encoder | n/a | **+5.38 pp** (jpn) / **+5.83 pp** (zho_lap) | **核心贡献** |

**论文 narrative**：旧的方法学（V3 时代）的两大 trick（Rethink、augmentation）在大 encoder 时代衰减为接近零。但 **per-language native encoder + 5-seed ensemble** 这两个简单选择共同贡献了 ~10 pp 的提升。

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
