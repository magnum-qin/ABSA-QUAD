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

**4 语言回译增强**（已生成，并完成重训）：

| 语言 | 保留率 | 新增样本 |
|------|--------|---------|
| DE (德语) | 82.0% | 1,874 |
| FR (法语) | 76.0% | 1,735 |
| ZH (中文) | 53.1% | 1,213 |
| RU (俄语) | 62.8% | 1,434 |
| **合计** | — | **6,256** |
| **原始 + 增强** | — | **8,540 (3.74x)** |

**V5.1 4语言增强 5-seed 结果**（增强数据 8540 条）：
- **Ensemble (vote≥3)**: cF1 = **0.6788** (cP=0.7156, cR=0.6455)
- **提升幅度**: 相比 3 语言增强 (0.6554) 提升 **+2.34 pp**；相比 V3 基线 (0.6278) 提升 **+5.10 pp**。
- 结果说明：在 eng_restaurant 上，加入俄语 pivot 后的 4 语言回译增强优于 3 语言增强，继续扩大高质量训练样本仍有收益。

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
| 4语言增强重训 | 用 8540 样本（含 RU）重训 5-seed | ✅ 已完成 (+3.57%, cF1 0.6788) |
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
| V5.1 4语言增强 Ensemble (eng_rest) | **0.6788** | **+10.1%** | ✅ |
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

**V20 是 V16/V19/V20 后续系列中的最佳**（dev 0.3088），但仍低于早期 V5/V6 BT dual-layer ensemble 的 0.3304。若按全历史 dev cF1 最大值选择，eng_laptop 仍应保留 V5/V6 结果；V20 主要作为 encoder × augmentation 交互效应的证据。

**全局视角**：从 V13 时代的 augmentation +2-4 pp，到 V16 时代衰减为持平/负，到 V20 在 roberta-large 上微弱回正。**augmentation 不是过时，而是高度 encoder-dependent**。论文可以写：「The marginal utility of synthetic data augmentation is not monotonically decaying with encoder capacity; rather it interacts non-trivially with encoder pretraining choice — RoBERTa-large benefits while DeBERTa-v3-large is harmed by the same augmentation set」。

---

### 5.26 V21 jpn_hotel DeepSeek 增强 + 分布式 5-seed + LLM 后处理（❌ 未超过 V16 no_aug）

**动机**：V16 no_aug 已把 jpn_hotel 提到 0.4500，但 §7.6.3 也说明强 native encoder 下 paraphrase augmentation 可能变成轻微噪声。Gemini 进一步尝试更激进的 DeepSeek 增强、跨节点并行 5-seed 训练，以及 LLM 后处理纠错，验证是否还能突破 V16 no_aug 上限。

**配置**：
- 数据：在原始 jpn_hotel 训练集基础上加入 **939 条 DeepSeek V4 Pro 生成增强样本**。
- 训练：seed = 42 / 123 / 456 / 789 / 1024，分布到 RTX 4090 节点并行训练。
- 融合：`inference.py --multi_seed_ensemble` 做 5-seed logit-level ensemble。
- 后处理：`llm_post_correction.py` 两版对比。V1 让 LLM zero-shot 直接纠错；V2 改成 Draft-Revision，只把模型预测作为草稿交给 LLM，要求仅在明显不合理时修正 category，并清洗日语 aspect/opinion 边界。

**结果**：

| 方案 | cF1 | Exact F1 | vs V16 no_aug 0.4500 | 结论 |
|---|---:|---:|---:|---|
| 最佳单 seed（789） | 0.4003 | 0.4230 | −4.97 pp | 单模型仍明显低于 V16 no_aug ensemble |
| 5-seed logit ensemble | 0.4315 | 0.4007 | −1.85 pp | ensemble 有效，但未超过 V16 no_aug |
| 5-seed + LLM V1 zero-shot correction | 0.2919 | — | −15.81 pp | LLM 脱离模型先验会严重破坏 category |
| 5-seed + LLM V2 Draft-Revision | **0.4318** | **0.4010** | −1.82 pp | 只带来 +0.03 pp，收益可忽略 |

**结论**：
1. 分布式 5-seed ensemble 仍然有效：从最佳单 seed 0.4003 提到 0.4315。
2. LLM 后处理必须保留模型先验；zero-shot category correction 会把强模型预测改坏。
3. Draft-Revision 能避免灾难性下降，但只提升 0.0003，不能弥补增强数据带来的噪声。
4. **V21 不采用**。jpn_hotel 仍保留 V16 no_aug ensemble（0.4500）作为当前最佳。

---

### 5.27 V22 多语种 DeepSeek 增强与大模型重跑（❌ 未替换现有最佳）

**动机**：在 V21 之后，把同一套“增强数据 + 5-seed ensemble”思路推广到 rus_restaurant / ukr_restaurant，并对 eng_laptop 做无增强大模型重跑，检查是否存在被前期实验低估的后期收益。

**配置**：
- `rus_restaurant`：mDeBERTa-base + DeepSeek 增强 + 5-seed ensemble。
- `ukr_restaurant`：mDeBERTa-base + DeepSeek 增强 + 5-seed ensemble。
- `eng_laptop`：DeBERTa-v3-large + raw data + 5-seed ensemble（无增强）。

**结果**：

| Dataset | V22 方案 | V22 cF1 | 当前采用最佳 | Δ vs 当前最佳 | 结论 |
|---|---|---:|---:|---:|---|
| ukr_restaurant | mDeBERTa-base + DeepSeek aug | 0.5063 | **0.5068** (V13b) | −0.05 pp | 接近持平，但未超过 V13b |
| rus_restaurant | mDeBERTa-base + DeepSeek aug | 0.4696 | **0.5267** (V15) | −5.71 pp | 明显低于 XLM-R-large V15 |
| eng_laptop | DeBERTa-v3-large + raw | 0.3056 | **0.3304** (V5/V6 BT) | −2.48 pp | 高于 V16 raw 0.3035，但低于历史最佳 |

**rus_restaurant 训练日志补充**（`pipeline_rus.log`，mDeBERTa-base + DeepSeek aug）：

| Seed | cF1 | Exact F1 | VA RMSE | Best Epoch |
|---:|---:|---:|---:|---:|
| 42 | 0.4067 | 0.4561 | 1.3721 | 26 |
| 123 | 0.4380 | 0.4893 | 1.3323 | 20 |
| 456 | 0.3509 | 0.3913 | 1.3403 | 9 |
| 789 | 0.3664 | 0.4123 | 1.4275 | 18 |
| 1024 | 0.3639 | 0.4087 | 1.3970 | 30 |
| **mean** | **0.3852** | - | - | - |

该日志只覆盖单 seed 训练摘要，后续 ensemble 推理命令因参数传递错误中断；V22 表中 `rus_restaurant` 的 0.4696 来自后续已完成 ensemble 结果。

**结论**：
- `ukr_restaurant` 的 DeepSeek 增强几乎复现 V13b，但没有超过 0.5068，说明 V13b 的 BT+paraphrase 组合已接近当前数据配方上限。
- `rus_restaurant` 回退到 mDeBERTa-base 后明显弱于 V15 XLM-R-large，说明 encoder scaling 比额外 DeepSeek 增强更关键。
- `eng_laptop` raw large 重跑略高于 V16 0.3035，但低于 V19/V20，也低于 V5/V6 的 0.3304；不替换最终配置。

---

> **2026-07-18 回溯修正**：§5.28–§5.29.4 是实验过程日志，其中把 dev cF1 与官方 test 榜单分数直接比较，属于无效的 cross-split comparison。该区间所有“超过/接近 test SOTA”“最终 submission”措辞仅保留为历史记录，均已由 §5.29.7 的一次性 official-test 结果撤回；TALLIP 不得引用这些 SOTA 判断。

### 5.28 V23 eng_laptop = LLM 知识蒸馏（dev 显著提升；历史 SOTA claim 已撤回）

**动机（历史假设）**：eng_laptop 有 240 个 (entity#attribute) 组合但仅 4076 个样本，类别覆盖稀疏。此前“推理时引入 LLM”的配方（§5.20.4 fusion、§7.7.1 post-correction）在本项目中失败，因此改测训练期 silver KD；推理时不使用 LLM。后续证据只支持该配方在部分域有效，不支持把它称为普适“正解”。

**Pipeline**：
1. **无标注池**：8,578 句笔记本评论（7,580 Amazon `naga-jay/amazon-laptop-reviews-enriched` + 998 eng_laptop test 文本，in-domain transductive）
2. **DeepSeek silver-labeling**：`scripts/llm_silver_label_eng_laptop.py`，deepseek-v4-pro 抽 (Aspect, Category, Opinion, VA) 四元组，121 类约束 + few-shot + 严格校验（category 合法 / span 子串 / VA∈[1,9]）→ **8,541 样本 / 26,723 silver quads**（keep_rate 0.99）
3. **蒸馏训练**：deberta-v3-base + raw 4076 + 4-pivot BT 10973 + silver 8541 = **23,590 行**，5-seed dual-layer
4. **推理**：纯 ensemble，**无 LLM**

**关键工程发现 — 关闭 thinking 省 19-29×**：deepseek-v4-pro 默认推理模式每句吐 ~2200 输出 token（大部分是隐藏 CoT），实际 JSON 只需 ~76-145。加 `extra_body={"thinking":{"type":"disabled"}}` 后输出降 19-29×、无 402 余额耗尽。同句 head-to-head：thinking ON 仅多 ~13% recall（0.39 vs 0.34 valid quads），有效率两者 ~98-100%，但精确重合仅 ~37% —— 说明 LLM 抽取本身方差大（解释了 LLM 队伍为何用 k=10 self-consistency）。silver 质量两模式等价，OFF 性价比完胜。

**单 seed cF1（5 seeds）**：

| seed | 42 | 123 | 456 | 789 | 1024 | mean |
|---|---|---|---|---|---|---|
| V23 KD | 0.3973 | 0.3906 | 0.3920 | 0.3940 | 0.3938 | **0.3935** |

对比所有历史 eng_laptop 单 seed（V16/V19/V20 ~0.29，V5-repro ~0.305）→ **KD 单 seed +8.8pp**。dev 为干净留出（silver 池只含 test 文本不含 dev），0.39 是真实泛化增益。

**Ensemble sweep**：

| 配置 | cF1 |
|---|---|
| std vote=3 topk=0 | 0.4038 |
| **std vote=4 topk=0** | **0.4141** 🏆 |
| std vote=4 topk=0.10 | 0.4138 |
| std vote=5 topk=0.10 | 0.4094 |
| dual vote=4 topk=0.10 | 0.4092 |

**eng_laptop 全程演进**：

| 方法 | dev cF1 | Δ |
|---|---|---|
| V16 deberta-v3-large raw | 0.3035 | baseline |
| V5-repro deberta-v3-base + 4-pivot BT | 0.3254 | +2.19 |
| Route A 6-pivot BT (10-seed) | 0.3109 | es/it 稀释，死路 |
| **V23 LLM 蒸馏** | **0.4141** 🚀 | **+11.06 vs V16 / +8.87 vs V5-repro** |

**已撤回的历史 cross-split 对标**：本节早期曾把 dev=0.4141 与 test leaderboard
0.4006/0.4227 比较；二者 split 不同，不能据此判断追平或超过榜单系统。
released official test 上 clean KD 为 0.3532，未超过榜单最佳 0.4227。

**允许保留的贡献表述**：440M deberta-v3-base 可在推理时不调用 LLM 的前提下吸收 silver supervision；其 eng_laptop official-test 分数相对 clean no-KD baseline 提升 +0.0822，95% CI [+0.0629, +0.1021]。该结果支持本域的 KD 收益，不支持相对 27B 系统的效率或优劣结论，因为没有同协议成本/效果对照。

**eng_laptop 最终 submission 切换为 V23**（std vote=4 topk=0），V20/V19/V16 备份保留。

---

### 5.29 V23 KD 推广到 zho_laptop / rus / ukr（历史过程记录；最终结论见 §5.29.5–§5.29.7）

**动机**：eng_laptop dev 上观察到 +8.87pp 后，将相同思路推广到 zho_laptop、rus 和 ukr，检验跨域可迁移性。此前记录的“SOTA 差距”是无效 cross-split 比较，已撤回。

**Pipeline 泛化**：
- `scripts/llm_silver_label.py`（通用版，`--domain` + `--lang`，category 从各语种 train 自动构建）
- `scripts/collect_unlabeled_multi.py`（多语种无标注采集）
- 数据源：zho_laptop = `SetFit/amazon_reviews_multi_zh`（笔记本关键词过滤）；rus = `d0rj/geo-reviews-dataset-2023`（Yandex，restaurant rubric 过滤，直接下 parquet 绕过 streaming segfault）；ukr = 无外部语料

**Silver 产量**（thinking-off, keep_rate 0.98-0.99）：

| 数据集 | 无标注池 | silver 样本 | silver quads | gold 基线 |
|---|---|---|---|---|
| zho_laptop | 7,415 | 5,156 | 17,133 | v13c 5,534 |
| rus | 8,630 | 8,364 | 38,396 | v13 1,766 |
| ukr | 6,551 (ru→uk MarianMT 翻译 geo-reviews + test) | 6,138 | 19,960 | v13b 2,967 |

**ukr 无外部乌语语料解法**：HF 无乌克兰语餐厅评论 → 用 MarianMT `opus-mt-ru-uk` 把俄语 geo-reviews 餐厅评论**翻译成乌语**构建无标注池，再 DeepSeek 标注。翻译质量良好（语法自然），DeepSeek 标注正常。

**最终结果（5-seed ensemble；tat 用 4-seed，见下）**：

| 数据集 | gold 量 | 旧最佳 dev | KD ensemble dev | Δ | 历史 test 榜单参考† | 当阶段选择 |
|---|---|---|---|---|---|---|
| **tat** | 1,240（最稀缺）| V15 0.4821 | **0.5196** (vote=2) | **+3.75** | Takoyaki 0.4736 | dev 提升；后续配方已替换 |
| **rus** | 1,766（稀缺）| V15 0.5267 | **0.5753** (vote=3) | **+4.86** | PAI 0.5599 | dev 提升；后续配方已替换 |
| **ukr** | 2,967（稀缺）| V13b 0.5068 | **0.5560** (vote=3) | **+4.92** | PAI 0.5437 | dev 提升；后续 clean 重训 |
| **eng_laptop** | 4,076（稀缺）| V5-repro 0.3254 | **0.4141** (vote=4) | **+8.87** | Takoyaki 0.4227 | dev 提升；后续 clean 重训 |
| zho_laptop | 5,534（充足）| V16 0.4487 | 0.4428 (vote=4) | **−0.59** | NYCU 0.4824 | ❌ 保留 V16 |

† 仅保留当时记录；dev 与 test 不可直接比较，不构成 SOTA 证据。

**tat 无外部鞑靼语语料解法**：HF 无鞑靼语餐厅评论 → **DeepSeek 直接生成 3986 条原生鞑靼语评论**（探针确认生成流利、含鞑靼语特有字母 ә ө ү җ ң һ，标注正确）+ test 文本 → silver 标注 9415 quads。这是 KD 对低资源语言可行性的强证据。

**各数据集 KD 单 seed**：
- tat：0.4994 / 0.4886 / 0.4210 / 0.4514 / **0.2318**（seed 1024 坏初始化，XLM-R-large 偶发）
- rus：0.4883 / 0.5155 / 0.5370 / 0.5111 / 0.5371（mean 0.5178，全 ≥ V15）
- ukr：0.4211 / 0.4358 / 0.4375 / 0.4750 / 0.5305（mean 0.4600，高方差；ensemble vote=3 翻盘到 0.5560，同 eng_laptop 模式）
- zho_laptop：0.3783 / 0.3763 / 0.3769 / 0.3847 / 0.3851（mean 0.3803，全 < V16 ~0.39）

**tat 坏 seed 的鲁棒性观察**：seed 1024=0.2318 是坏初始化，但 **5-seed ensemble 与 4-seed（丢 1024）逐位相同 = 0.5196** —— vote≥2 时坏 seed 的低质量 quad 拿不到 2 票，被投票机制自动剔除。最终 tat 采用 **4-seed vote=2**。说明 **ensemble vote 对单个坏 seed 鲁棒**。

**当阶段观察 — 多个较小数据集 dev 提升，但“收益与稀缺度单调”已被后续证据否定**：

| 数据集 | gold 量 | KD ensemble Δ | 验证 |
|---|---|---|---|
| **tat** | 1,240（最稀缺）| **+3.75pp** | dev 方向性提升 |
| **rus** | 1,766（稀缺）| **+4.86pp** | dev 提升 |
| **ukr** | 2,967（稀缺）| **+4.92pp** | dev 方向性提升 |
| **eng_laptop** | 4,076（稀缺）| **+8.87pp** | dev 提升 |
| **zho_laptop** | 5,534（充足）| **−0.59pp** ❌ 稀释 | 无益 |

**回溯解读**：这一小样本矩阵同时改变了语言、encoder、pool、teacher 和
gold 规模，不能识别“数据稀缺度”的独立效应。后续 zho_restaurant、teacher
fidelity、clean-inductive 与 official-test 结果进一步否定了单调规律和
cross-split SOTA 叙述。允许保留的解释是：silver 的质量、覆盖及其相对 gold
权重共同影响收益，低资源本身既非充分条件也非必要条件。

**该阶段采用**：tat → V23 KD (0.5196, 4-seed)；rus → V23 KD (0.5753)；ukr → V23 KD (0.5560)；eng_laptop → V23 KD (0.4141)；zho_laptop → 保留 V16。后续 tat teacher、rus pool/规模和 zho_restaurant pool 消融继续刷新，见 §5.29.2.1、§5.29.3.1–2。

#### 5.29.1 消融：原生生成 silver vs 机器翻译 silver（ukr）

**动机**：ukr 的 silver 来自 MarianMT 翻译的俄语评论，怀疑 translationese 限制上限。对照实验：用 DeepSeek 直接生成 3994 条**原生**乌语餐厅评论 → silver 标注（9255 quads）→ 同样 mdeberta 5-seed。

| silver 来源 | silver 量 | 单 seed mean | 单 seed std | ensemble best (vote=3) |
|---|---|---|---|---|
| 机器翻译 (MarianMT, 6138) | 19,960 quads | 0.4400 | ~0.041（高方差）| **0.5560** 🏆 |
| 原生生成 (DeepSeek, 3991) | 9,255 quads | **0.4937** | ~0.027（更稳）| 0.5446 |

**🔍 反直觉发现 —— 单 seed native 赢，ensemble translated 赢**：
- **per-sample 质量**：原生生成明显更高（单 seed +5.4pp，无 translationese，且用更少数据）
- **ensemble**：机器翻译反而 +1.14pp。原因：① 翻译版单 seed 方差大 → seed 间多样性高，ensemble 投票收益大（"高方差+ensemble 救回"，同 eng_laptop）② 翻译版 silver 量更大（6138 vs 3991）→ 覆盖广，vote recall 高
- **结论**：对 KD，**silver 的量+多样性 在 ensemble 层面 ≈ 甚至 > per-sample 质量**。论文叙述：「Native LLM-generated in-domain text yields higher per-seed silver quality (+5pp single-seed, no translationese), but machine-translated text's larger volume and higher seed-diversity win at ensemble level (+1.1pp).」

**ukr 最终保留机器翻译版 V23 KD (0.5560)** —— ensemble 最优。

---

#### 5.29.2 消融：teacher 选择对 KD 的影响 — 云端 DeepSeek vs 本地 Gemma-31B vs 本地 qwen3.6-36B（eng_laptop）

**动机**：eng_laptop KD 用云端 DeepSeek 做 silver-labeling。teacher 能力对 KD 增益有多关键？换成**本地部署**的开源模型能否复现、甚至追平？这是 KD 范式可复现性 / 成本（云 API vs 自有 GPU）的关键问题。测三个 teacher：云端 `deepseek-v4-pro`、本地 `gemma-4-31B-it-qat`(Q4，ollama)、本地 `qwen3.6:latest`(36B Q4_K_M，ollama)。

**严格对照设计**：同一组 **8578** 条 in-domain 笔记本评论（直接取 DeepSeek silver 文件的 ID+Text，逐字相同）；gold + 4-pivot BT 基座 **15049 行逐字不变**；同 config（deberta-v3-base + Rethink + 5-seed dual-layer ensemble，patience=8）。**唯一变量 = silver teacher**。

**踩坑记录**：① gemma-4-QAT / qwen3.6 都是推理模型，CoT 走 `thinking` channel 导致 OpenAI-compat `content` 为空 → ollama native `/api/chat` 加 `think:false` 修复（qwen3-vl:4b 不支持该参数，弃用）；② 模型曾被 fallback 到 CPU → `num_gpu:99` 强制全量上 GPU；③ gemma 31B 用 `OLLAMA_NUM_PARALLEL=4` 并行（0.25→0.8 rows/s）；qwen3.6-36B 占 23.6/24.5GB 显存无法并行，单流 ~1.9s/样本；④ qwen3.6 对隐式 aspect 输出 JSON `null` 而非 `"NULL"` 字符串 → 脚本加 None→"NULL" 归一化。

| teacher | 部署 | 规模 | silver 量 | quads/text | 单 seed mean | ensemble dev (vote=4 topk=0) | vs dev 基线 |
|---|---|---|---|---|---|---|---|---|
| 无 KD (V5-repro) | — | — | — | — | ~0.305 | 0.3254 | — |
| **Gemma-31B-Q4** | 本地 | 31B | 8538 / 24,049 | 2.82 | ~0.355 | **0.3764** (topk=0.20) | **+5.10pp** |
| **qwen3.6:latest** | 本地 | 36B | 8426 / 23,541 | 2.79 | **0.3777** | **0.4094** | **+8.40pp** |
| **DeepSeek-v4-pro** | 云端 | frontier | 8541 / 26,723 | 3.13 | ~0.39 | **0.4141** | **+8.87pp** |

ensemble sweep 三者一致偏好高 vote：qwen vote=4(0.4094) > vote=3(0.4009) > vote=2(0.3631)；gemma vote=4(0.3764) > vote=3(0.3674) > vote=2(0.3517)。

**🔍 核心发现 —— teacher 重要，但「云 vs 本地」不是决定因素，模型质量才是**：
- **本地 Qwen 在该配方的 dev 上接近 DeepSeek**：ensemble 仅差 0.47pp，并回收了按点估计计算的 95% KD 增益；这是单域工程可行性证据，不代表所有域等价。
- **本地模型点估计有差异**：Qwen 比 Gemma 高 3.3pp；但学生分数还混合了 teacher–student 适配等因素，不能仅凭该结果把差异全部归因于 teacher 标注正确率。
- **silver 密度不是 fidelity 指标**：Qwen 的 quads/text 略低于 Gemma 而学生分数更高，只能否定“密度越高必然越好”；直接正确性结论应以后续 8×2 gold-dev teacher fidelity 为准。
- **允许的论文表述**：该 eng_laptop 对照显示本地 Qwen 可接近云端 DeepSeek，部署位置本身不足以预测 KD 效果；跨域 teacher 结论见 §5.29.5–5.29.7。

**eng_laptop 提交不切换** —— 仍用 DeepSeek 臂 V23 KD (0.4141，三者最高)。Gemma/qwen 臂作 teacher-quality 消融证据。脚本：`scripts/llm_silver_label_ollama.py`、`scripts/build_eng_v23_{gemma,qwen}.py`、`scripts/run_v23_{gemma,qwen}_eng_laptop.sh`。

##### 5.29.2.1 teacher 等价性补充（qwen vs DeepSeek，完整有效 n=4）

**动机**：§5.29.2 的"本地 qwen ≈ 云端 DeepSeek"只有 eng_laptop 一个数据点。在 3 个 DeepSeek 已胜的数据集（rus/ukr/tat）上补跑 qwen，**复用同一池、同 encoder/config，仅换 teacher**，检验泛化性。用 `scripts/kd/pipeline.py` 主干驱动。

| 数据集 | 语言（资源）| qwen-KD | DeepSeek-KD | 基线 | qwen vs DeepSeek |
|---|---|---|---|---|---|
| eng_laptop | 英语（高）| 0.4094 | 0.4141 | 0.3254 | −0.5pp |
| rus_restaurant | 俄语（高）| 0.5365 | 0.5753 | 0.5267 | −3.9pp |
| ukr_restaurant | 乌克兰语（低，翻译池）| 0.4886 | 0.5560 | 0.5068 | −6.7pp |
| **tat_restaurant** | **鞑靼语（极低，原生生成池）**| **0.5327** | 0.5196 | 0.4821 | **+1.3pp（历史非 matched）** |

qwen3.6 teacher = `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`（35B-A3B MoE，Unsloth Studio /v1 服务）。

**⚠️ 两次错误修正（过程教训）**：① 最初只有 eng_laptop（n=1）得出"本地≈云端"——过度泛化。② 第一次补跑 ukr/tat 时，rus 跑完后本地 LLM 服务的模型被换掉，silver 标注**全部 404 失败**、distill 静默退化为 gold-only（ratio 0），得出"低资源语言塌陷 −15pp"——**坏数据，已撤回**。本表是换用 Unsloth Studio 重跑后的**有效真 KD 数据**（ukr silver 6162、tat silver 3915 条，keep_rate≈0.96）。pipeline 已加防护：silver 0 有效 quad 时 distill 直接报错。

**历史结果的限制**：上述 tat DeepSeek 0.5196 与 Qwen 0.5327 并非严格 matched control；两臂的 silver 文本、样本数和旧 DeepSeek prompt 均不同，因此不能据此断言 Qwen teacher 在 Tatar 上稳定反超。

**Tatar matched-control（2026-07-18）**：固定同一 1,240 条 gold、同一 3,915 条 silver 文本及顺序、相同 prompt/schema、相同五个 seeds 与固定 vote=3/topk=0，唯一变量为 teacher 标签。

| Teacher | 5-seed mean±SD | vote=3 ensemble cF1 |
|---|---:|---:|
| DeepSeek | **0.4708±0.0065** | **0.5365** |
| Qwen | 0.4637±0.0147 | 0.5327 |

DeepSeek−Qwen ensemble Δ=**+0.0038**，10,000 次 paired bootstrap 95% CI **[−0.0823, +0.0852]**，判定为 **inconclusive**。因此原先的 Qwen +1.3pp 优势属于不稳定/混杂证据；严格控制后两者下游表现近似，既不支持 Qwen 反转，也不支持 DeepSeek 在 Tatar 下游显著更优。允许的结论是：**本地 Qwen 可以接近云端 DeepSeek，但 teacher 排名需直接 fidelity 与严格 matched downstream control 共同判断。**

---

#### 5.29.3 KD 推广到剩余 3 个数据集（jpn_hotel / eng_restaurant / zho_restaurant，teacher=本地 qwen3.6）

**动机**：把 KD 推广到尚未蒸馏的全部 3 个数据集。三者都**没有 in-domain 无标注语料**，用 qwen3.6 原生生成（中/日/英三语，质量经探针确认）~4000 条 + test 文本构成池，qwen3.6 silver 标注，各自 V16 专用 encoder 5-seed 训练。脚本：`scripts/gen_native_ollama.py`、`scripts/run_v23_multi_qwen.sh`。

| 数据集 | encoder | gold | silver | **silver:gold** | gold-only dev | qwen-KD ensemble | Δ | 类别数 |
|---|---|---|---|---|---|---|---|---|
| jpn_hotel | deberta-v2-large-japanese | 2893 | 4558 | 1.6 | 0.4466 | 0.4043 (v4 t0.10) | **−4.2pp ❌** | 44 |
| eng_restaurant | deberta-v3-large | 2284 | 4884 | **2.1** | 0.6554 | 0.4043 (v3 t0.0) | **−25.1pp ❌❌** | 14 |
| **zho_restaurant** | chinese-roberta-wwm-ext-large | **6050** | 4776 | **0.79** | 0.4745† | **0.5263** (v3 t0.0) | **+5.2pp vs historical ref** | 12 |
| zho_laptop（capped silver）| chinese-roberta-wwm-ext-large | 5534† | 2700 | **0.49** | 0.4487 | 0.4454 (v4 t0.10) | **−0.3pp ⚠️** | 113 |

†zho_laptop base 用 V16 配方 v13c（5534，含 paraphrase aug）。zho_restaurant 的 0.4745 后来确认不是最强 no-KD baseline；最强冻结 dev baseline 为 0.5669。其细 sweep 仅说明该 KD 臂对阈值不敏感，不能恢复整体改善结论。

**回溯解释：信息互补性是与结果一致的候选机制，不是已独立验证的统一判据**：
- 中途“KD 模型收敛到 teacher 标注质量 ~0.40 天花板”的假设被 zho_restaurant 0.5263 推翻。以下是当时观察到的关联，不能拆分 silver:gold、gold 质量、类别稀疏度和数据集差异的独立效应：
  - **eng_laptop**：ratio 2.1、历史参考较弱且类别稀疏，dev 点估计为正；新增类别覆盖是可能解释。
  - **zho_restaurant**：ratio 0.79，相对 0.4745 历史参考点估计为正；恢复强 baseline 后该结论不再成立。
  - **eng_restaurant**：ratio 2.1、历史 baseline 较强，Qwen-KD dev 大幅下降；DeepSeek 也未超过 baseline。
  - **jpn_hotel**：两个 teacher 的 KD dev 臂均低于 native-encoder baseline。
  - **zho_laptop**：capped ratio 0.49 与 full ratio 0.86 均略低于 baseline；两个点不足以识别普适比例效应。
- **候选解释**：eng_laptop 的稀疏类别覆盖与 zho_restaurant 的新增表达多样性可能解释 dev 改善；zho_laptop 的 capped/full 两臂均未改善。由于这些对照还混合了数据集、base 和 teacher，不把“信息互补性”写成充要条件。
- **允许的论文叙述**：现有结果表明 KD 收益具有数据集依赖性，并与 silver 的互补覆盖、标注质量及 gold:silver 权重共同变化；后续 official test 支持四域收益，但 zho_restaurant 证据不确定。

##### 5.29.3.1 消融：真实公开语料池 vs LLM 生成池（排除池来源 confound）

**动机**：失败的 3 个数据集都用 LLM 生成池，怀疑败于生成文本质量而非 KD 机制。用真实公开语料 + 少数派 silver 重试，控制"池来源"+"碾压比例"两个变量。jpn_hotel 因 HF 无可访问日语酒店语料（amazon_ja 是商品评论、Rakuten Travel 需申请）**无法测**。脚本：`scripts/collect_real_pools.py`（Yelp / amazon_zh+opencc 繁化）。

| 数据集 | 真实语料 | 生成池 KD | 真实语料池 KD | 基线 | Δ vs 基线 |
|---|---|---|---|---|---|
| eng_restaurant | Yelp（ratio 0.79）| 0.4043 (ratio 2.1) | **0.4190** | 0.6554 | **−23.6pp ❌** |
| zho_laptop | amazon_zh→繁体（ratio 0.25，1392 silver）| 0.4454 (ratio 0.49) | **0.4264** | 0.4487 | **−2.2pp ❌** |

**🔍 该局部对照的结论：更换 pool 未反转两个失败臂**。
- eng_restaurant：真实 Yelp + 少数派 silver 比生成池高 +1.5pp（0.4190 vs 0.4043），但仍低于基线 −23.6pp；这只能说明该 pool 替换不足以恢复收益。
- zho_laptop：真实语料反而略差（受语料稀缺 silver 仅 1392、ratio 0.25 拖累），两版都低于基线。
- **注意**：以上两个数据集的两臂都低于基线，因此该对照对 pool 排名的识别力有限；不能写成“池来源无所谓”。

**补充 ablation：历史 dev 参考下点估计为正的数据集，真实 vs 生成池（zho_restaurant，同 teacher DeepSeek，仅换池）**：

| 池来源 | dev cF1 | vs 生成 |
|---|---|---|
| **真实 dianping（繁化，4000+test）** | **0.5579** | — |
| 生成（qwen 原生，当前提交）| 0.5351 | 真实 **+2.3pp** |
| historical reference arm† | 0.4745 | |

**🔑 修正结论（关键）**：在 zho_restaurant 的该组 dev 配方中，真实 in-domain 语料池比生成池高 +2.3pp；但这不是跨数据集的普适排序，rus 的 size-matched 对照反而是生成池最佳（§5.29.3.2）。更准确的结论是：**pool 效果与域互补性、teacher 标注质量和 silver:gold 比共同变化，现有对照不能分离所有因素**。脚本：`scripts/collect_dianping.py`（twn39/dianping + opencc s2twp）。

**历史 dev 选择**：zho_restaurant 当时从 0.4745 参考臂切到 DeepSeek 真实池
0.5579；后续发现更强 no-KD dev baseline=0.5669，clean paired CI 跨 0，
official test 的 KD 点估计更低且 CI 跨 0。因此 zho_restaurant 不计为得到统计
支持的 KD 成功案例。

##### 5.29.3.2 rus_restaurant：池来源三路对照 + 8k 独立批次复验

**严格三路对照**：同一 DeepSeek teacher、同 gold 1240、同 XLM-R-large 配置；real / translated / generated 池尽量 size-matched，并统一 `silver_cap=3000`，最终训练集均为 4240 行（translated 原池 3466，其余 4000）。

| variant | pool / silver used | 5-seed ensemble best | vs baseline 0.5267 | 结论 |
|---|---:|---:|---:|---|
| real4k | 4000 / 3000 | 0.5586 | +3.19pp | 有效但最弱 |
| trans4k | 3466 / 3000 | 0.5761 | +4.94pp | 居中 |
| **gen4k** | **4000 / 3000** | **0.6162** | **+8.95pp** | **三路最佳，提交采用** |
| gen8k full（独立生成批次） | 8000 / 7988 | 0.5858 | +5.91pp | 比 gen4k 低 3.04pp，淘汰 |

8k 完整复验的单 seed best 为 **0.5286 / 0.5648 / 0.5207 / 0.5501 / 0.5463**（mean **0.5421**）；ensemble sweep 最佳为 **vote=4, topk=0.10 → 0.5858**。所有 5 seeds 均完成，训练集 9228 行（1240 gold + 7988 valid silver）。审计发现 gen4k 与 gen8k 来自两次独立生成，按 `(ID, Text)` 的交集为 **0**；因此二者同时改变了 pool batch 与 silver 数量，不是干净的 dose-response。

**同池数量控制（paired-seed gate）**：复用 gen8k 的同一份 DeepSeek silver，仅将 `silver_cap` 从 0（7988 条）改为 3000；训练集从 9228 行变为 4240 行。先跑相同 seeds 42/123，预注册 gate 为 paired mean 相对 full-8k 提升至少 +1.5pp。

| 同一 gen8k silver | seed 42 | seed 123 | paired mean | Δ vs full |
|---|---:|---:|---:|---:|
| full 7988 | 0.5286 | 0.5648 | **0.5467** | — |
| cap 3000 | 0.5123 | 0.5584 | **0.5353** | **−1.14pp** |

gate 未通过，因此按计划停止，不补 seeds 456/789/1024、不跑 ensemble、也不补 cap5000。**修正结论**：① pool 来源没有跨数据集固定优先级——zho 上 real 胜，rus 的 size-matched 三路对照中 generated 胜；② 对同一 gen8k batch，减至 3000 silver 没有改善，故 gen4k 0.6162 相对独立 gen8k 0.5858 的优势更可能来自**生成批次质量/信息互补性**，而非简单的 silver 数量效应。rus 最终保留 gen4k capped 提交 0.6162。

#### 5.29.4 完整 teacher 矩阵（8 数据集 × DeepSeek / qwen，同池仅换 teacher）

补齐 DeepSeek 在最初只跑过 qwen 的 3 个数据集（zho_restaurant / jpn_hotel / eng_restaurant），形成全 8 数据集的云端 vs 本地历史学生分数矩阵（dev cF1）：

| 数据集 | 当时 reference | DeepSeek-KD | qwen-KD | DS − qwen | KD 最优 vs reference |
|---|---|---|---|---|---|
| eng_laptop | 0.3254 | **0.4141** | 0.4094 | +0.5 | +8.9 ✅ |
| rus_restaurant | 0.5267 | **0.5753** | 0.5365 | +3.9 | +4.9 ✅ |
| ukr_restaurant | 0.5068 | **0.5560** | 0.4886 | +6.7 | +4.9 ✅ |
| tat_restaurant | 0.4821 | 0.5196 | **0.5327** | −1.3 | +5.1 ✅ |
| zho_restaurant | 0.4745† | **0.5351** | 0.5263 | +0.9 | +6.1（仅历史参考） |
| zho_laptop | 0.4487 | 0.4428 | 0.4454 | −0.3 | −0.3 ❌ |
| jpn_hotel | 0.4466 | 0.4281 | 0.4043 | +2.4 | −1.9 ❌ |
| eng_restaurant | 0.6554 | 0.6013 | 0.4043 | **+19.7** | −5.4 ❌ |

† zho_restaurant 后续恢复的最强 no-KD dev baseline 为 0.5669；本表 0.4745
不能用于最终 KD−baseline 结论。

**历史学生分数矩阵的限制**：
- 该表是不同下游学生臂的 dev 点估计，不是直接 teacher fidelity；旧的“DeepSeek 赢 6、tat 输、zho_laptop 平”不能替代后续 8×2 gold-dev 标注质量评测。直接 fidelity 为 DeepSeek 点估计 8/8 更高，但仅三域差异得到统计支持。
- **eng_restaurant 点估计差异很大**：DeepSeek 0.6013 vs Qwen 0.4043（+19.7pp），与后续直接 fidelity 中 DeepSeek 的显著优势方向一致；但 DeepSeek 仍低于 dev baseline 0.6554，因此本配方的 KD 净负。
- **jpn 两个 teacher 臂均未超过 dev baseline**：这说明仅替换这两个 teacher 没有救回该配方；不能进一步断言失败与 teacher 完全无关。
- **“local ≈ cloud” 需限定到域**：多数学生臂点估计接近，但 eng_restaurant 差异很大。最终 teacher 结论以后续直接 fidelity 及其 CI 为准，而非部署类型。
- **同生成池的阶段性切换**：zho_restaurant 由 qwen 0.5263 → DeepSeek 0.5351；加入真实池消融后最终采用 0.5579（§5.29.3.1）。jpn/eng_restaurant 仍保留 baseline。

#### 5.29.5 最终证据收敛（论文与汇报统一口径）

早期“低资源必然受益”“真实池普遍更优”“silver 减量就是甜区”等说法，均被后续跨 teacher、跨 pool、同池数量控制和 official test 进一步限定。最终只保留以下六条结论：

1. **KD 的收益具有条件性。** 结果与“silver 提供 gold/base 尚未覆盖且标注较准确的类别先验、表达多样性或域信息时更可能有效”一致；后续 post-lock exposure control 在 eng/rus/ukr 排除了“仅增加训练行数”这一替代解释，但 tat/zho 未得到支持。信息互补性仍是解释性机制，不宣称为已独立验证的普适因果定律。
2. **DeepSeek teacher 整体更稳健，Qwen 在部分域接近，但不存在可靠的排名反转。** 完整 8×2 直接 fidelity 中 DeepSeek cF1 在 8/8 域更高；eng_restaurant、zho_laptop、zho_restaurant 的 paired CI 下界 > 0，eng_laptop/jpn/rus/tat/ukr 则跨 0。Tatar matched downstream control 同样跨 0，故直接与下游证据均不支持“某些域 Qwen 显著更优”。
3. **Pool 来源没有普适排序。** zho_restaurant 是 real > generated，而 rus 的 size-matched 对照是 generated > translated > real。决定因素是 pool 与目标任务的互补性及其被 teacher 转化为有效四元组的质量。
4. **同池减量未改善，rus 的 gen4k 优势主要来自生成批次质量。** 对完全相同的 gen8k silver，cap3000 paired mean 比 full 低 1.14pp；独立 gen4k 批次的 0.6162 不能归因于“只用 3000 条”。
5. **最佳 silver 不是“越多越好”，而是质量与覆盖合适。** 数量、类别覆盖、文本多样性、teacher 标注正确性和 gold:silver 权重共同决定结果；当前证据不支持跨数据集的统一 cap 或 dose-response 最优点。
6. **官方 test 支持条件性收益，但不支持 SOTA。** dev 冻结系统在 eng/rus/tat/ukr 的 KD−baseline paired CI 下界大于 0，zho_restaurant 跨 0；8 个主系统均未超过官方榜单最佳（§5.29.7）。

**统一表述**：KD is effective when silver data contributes complementary, accurately labeled coverage beyond the gold set. Teacher and pool quality are domain-dependent, and the best silver set is not the largest one but the one with an appropriate quality–coverage trade-off. The released official test supports gains over matched no-KD baselines in four domains, but the method does not establish a new test SOTA.

#### 5.29.6 TALLIP clean-inductive 审计与统计闭环（2026-07-13—2026-07-17）

对所有 gold / silver / distill / dev / test 采用 **Unicode NFKC + casefold + 空白折叠**进行文本泄漏审计。原 V23 中三个已采用 KD 训练集含有 dev/test 文本，因此 §5.29.2–§5.29.5 中相应结果在 clean 重跑完成前必须标记为 **transductive**：

| 数据集/系统 | 原行数 | 删除 | clean 行数 | dev 重合 | test 重合 |
|---|---:|---:|---:|---:|---:|
| eng_laptop KD | 23,590 | 995 | **22,595** | 1 | 994 |
| eng_laptop no-KD baseline | 15,049 | 1 | **15,048** | 1 | 0 |
| ukr_restaurant KD | 9,105 | 605 | **8,500** | 1 | 604 |
| ukr_restaurant no-KD baseline | 2,967 | 1 | **2,966** | 1 | 0 |
| zho_restaurant KD-real | 10,911 | 979 | **9,932** | 1 | 979 |

rus-gen4k 与 tat-qwen 的 adopted train 对 dev/test 规范化文本交集均为 0，无需重训。由于 eng/ukr baseline 也各含 1 条 dev 重合，严格 paired 方案由原计划的 15 个 KD seeds 扩展为 **25 个任务**：3×5 clean KD + 2×5 clean baseline。所有新 prefix 均含 `inductive`，旧 checkpoint 不覆盖；ensemble 阈值冻结为 eng vote=4、ukr vote=3、zho vote=3，top-k 均为 0。

**额外基线口径修正（已闭环）**：此前 KD 章节将 zho_restaurant no-KD baseline 写为 0.4745；仓库中冻结、可复现的更强 V16 no-KD dev ensemble 为 **0.5669**。clean KD dev=0.5738，但 paired CI 跨 0；official test 上 clean KD=0.4660、baseline=0.4798，Δ=−0.0137 [−0.0289, +0.0014]。因此 zho_restaurant 最终不计为得到统计支持的 KD 成功案例，0.4745 仅保留为历史参考臂。

直接 teacher fidelity 采用 8 个 gold dev（每个 teacher 共 1,344 条）去除 `Quadruplet`，类别与 6-shot 示例只来自 gold train。DeepSeek 与 Qwen 均完成 **1,344/1,344**，模型身份、ID 覆盖、重复和错误检查全部通过，最终均为 0 调用错误。主指标为官方 Task-3 cF1，并补充 categorical exact F1、aspect/opinion/category F1、matched-VA MAE/RMSE、NULL/稀有类别切片及错误率。所有 teacher 和 clean KD 对比均固定 10,000 次 paired bootstrap、seed=20260713；95% CI 下界 >0 / 跨 0 / 上界 <0 分别记为 supported / inconclusive / harmful。

**Clean-inductive 最终结果（25/25 训练任务、25/25 best checkpoint 完成）**：

| 数据集 | 5-seed mean±SD | 固定阈值 ensemble | baseline | ΔcF1 [95% CI] | 判定 |
|---|---:|---:|---:|---:|---|
| eng_laptop clean KD | 0.3785±0.0107 | **0.4102** | 0.3045 | +0.1058 [+0.0601, +0.1537] | supported |
| rus_restaurant adopted KD | 0.5553±0.0099 | **0.6162** | 0.5267 | +0.0895 [+0.0191, +0.1631] | supported |
| tat_restaurant adopted KD | 0.4637±0.0147 | **0.5327** | 0.4821 | +0.0506 [−0.0302, +0.1353] | inconclusive |
| ukr_restaurant clean KD | 0.4851±0.0355 | **0.5377** | 0.4905 | +0.0472 [−0.0466, +0.1449] | inconclusive |
| zho_restaurant clean KD-real | 0.5400±0.0026 | **0.5738** | 0.5669 | +0.0069 [−0.0204, +0.0349] | inconclusive |

清洗前后差异均无统计支持：eng clean−transductive = −0.0038 [−0.0444, +0.0351]；ukr = −0.0183 [−0.0738, +0.0292]；zho = +0.0159 [−0.0075, +0.0398]。这说明删除重合行没有可靠地改变三域结论，但论文主表仍必须使用 clean-inductive 结果。仅就 **dev evidence** 而言，五个 adopted KD 数据集中 eng 与 rus 得到统计支持，tat/ukr/zho 仅为方向性改善；official test 的预注册对照见 §5.29.7。不得再写“5/5 数据集均确认提升”。

完整 teacher fidelity 结果（DeepSeek prompt 与 Qwen byte-identical 后的最终重跑）：DeepSeek/Qwen cF1 分别为 eng_laptop 0.3575/0.3170、eng_restaurant 0.5529/0.3638、jpn_hotel 0.2146/0.2004、rus 0.3904/0.3649、tat 0.4359/0.3954、ukr 0.3786/0.3598、zho_laptop 0.2821/0.2374、zho_restaurant 0.4653/0.4060。DeepSeek−Qwen 仅在 eng_restaurant、zho_laptop、zho_restaurant 得到统计支持，其余五域 inconclusive；没有任何域支持 Qwen > DeepSeek。Tatar matched downstream control 为 DeepSeek 0.5365 vs Qwen 0.5327，Δ=+0.0038 [−0.0823, +0.0852]，同样 inconclusive。故最终表述为“DeepSeek 整体更稳健，Qwen 在部分域接近；teacher 优势幅度具有域差异”，而非“teacher 排名发生可靠反转”。

可复现入口与最终结果见 `scripts/kd/tallip.yaml`、`scripts/kd/tallip_status.py` 和 `docs/audits/tallip_kd/`。原五张核心表在 2026-07-18 冻结；post-lock dev-only controls 由 v1+v2 两个结果前注册协议完成五域矩阵并追加为 Table 5，原 official-test 表顺延为 Table 6。六张表于 2026-08-13 重新锁定，由 `tallip_core_tables.lock.json` 固定输出、生成器、协议和全部证据输入的 SHA256；引用前运行 `python scripts/kd/build_tallip_report.py --verify`。默认命令只验证，更新必须显式使用 `--update-lock`。永久停止 cap5000、rus cap3000 剩余 seeds 和新的 teacher/pool 网格。

#### 5.29.7 官方 test 一次性评测（2026-07-18）

官方在赛后公开了 Track A Task 3 的 test gold。本节是 **post-competition local evaluation**，不是新的 CodaBench 官方提交或排名。评测矩阵只按 dev 结果选择，并在取得 gold 前冻结：官方仓库 commit `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`，8 个无标签本地 test 与 gold 的 ID、顺序和 Text 均逐条完全一致；所有 prediction、checkpoint、config、gold 和 evaluator 均保存 SHA256。评分使用官方 `metrics_subtask_1_2_3.py`，paired bootstrap 固定 10,000 次、seed=20260713。取得 test 标签后未重新训练、调阈值或替换系统。

| 数据集 | dev 冻结主系统 | official test cF1 | 官方榜单最佳 | 差距 |
|---|---|---:|---:|---:|
| eng_laptop | clean-inductive DeepSeek KD | **0.3532** | 0.4227 | −0.0695 |
| eng_restaurant | V16 no-KD | **0.5904** | 0.6514 | −0.0610 |
| jpn_hotel | V16 no-aug | **0.4217** | 0.4252 | −0.0035 |
| rus_restaurant | DeepSeek gen4k KD | **0.5189** | 0.5599 | −0.0410 |
| tat_restaurant | Qwen KD | **0.4311** | 0.4736 | −0.0425 |
| ukr_restaurant | clean-inductive DeepSeek KD | **0.4608** | 0.5437 | −0.0829 |
| zho_laptop | V16 no-KD | **0.4309** | 0.4824 | −0.0515 |
| zho_restaurant | clean-inductive DeepSeek-real KD | **0.4660** | 0.5521 | −0.0861 |

官方榜单参考：`https://docs.google.com/spreadsheets/d/1geyC9aFjAmXFE_7w1LnjGYX4F0y2jttN/`。8 个冻结主系统均未超过榜单最佳，因此早期仅凭 dev 对比写出的“击败 test SOTA”或“反超官方第一”必须撤回。最接近榜首的是 jpn_hotel（−0.35pp），但仍不能宣称 SOTA。

预注册的 adopted KD 对照如下；zho_restaurant 的主系统仍保留 gold 前按 dev 选出的 clean KD，不允许看完 test 后换回 V16：

| 数据集 | KD | baseline | ΔcF1 [95% CI] | 判定 |
|---|---:|---:|---:|---|
| eng_laptop | 0.3532 | 0.2709 | +0.0822 [+0.0629, +0.1021] | supported |
| rus_restaurant | 0.5189 | 0.4978 | +0.0211 [+0.0005, +0.0431] | supported |
| tat_restaurant | 0.4311 | 0.4037 | +0.0274 [+0.0071, +0.0482] | supported |
| ukr_restaurant | 0.4608 | 0.4215 | +0.0393 [+0.0150, +0.0639] | supported |
| zho_restaurant | 0.4660 | 0.4798 | −0.0137 [−0.0289, +0.0014] | inconclusive |

**结论更新**：official test 在 eng/rus/tat/ukr 四域支持 KD 相对预注册 no-KD baseline 的改善；zho_restaurant 点估计转为负向但 CI 跨 0，不能再作为 KD 成功案例。dev 上 tat/ukr 为方向性改善，test 因样本更多而 CI 下界转正；zho 在 dev/test 均不确定且方向不一致。总体结论可以保留为“KD 在信息互补条件下有效”，但不能写成普适改善或 SOTA 方法。

Tatar matched-pool teacher sensitivity 在 test 上为 DeepSeek 0.4421 vs Qwen 0.4311，Δ=+0.0109 [−0.0069, +0.0293]，仍为 inconclusive。这与 dev 的 Δ=+0.0038、CI 跨 0 一致，说明下游 teacher 差异容易受 seed/ensemble/test sample 扰动；没有可靠证据支持 teacher 排名反转。

可复现协议、冻结哈希、gold 审计、官方原始输出与完整统计位于 `scripts/kd/official_test.yaml`、`scripts/kd/official_test.py` 和 `docs/audits/tallip_kd/official_test/`。从本节完成起永久停止基于 official test 的模型、阈值、teacher、pool 或增强选择。

#### 5.29.8 与 ICONIP V20 的贡献边界（2026-07-18）

ICONIP V20 的研究问题是如何构建并按数据集选择稳定的多语言 DimASQP 学生
系统；LEADER-DimASQP、Rethink、Hungarian matching、language-specific
encoder/recipe、five-seed ensemble 及早期增强/配置消融均属于 V20 已有贡献。

TALLIP 固定并复用上述学生系统，新增研究问题是：离线 LLM silver 在什么
条件下改善学生。可作为新贡献的证据仅包括 8×2 teacher fidelity、matched
teacher control、pool/quantity 边界、clean-inductive 清理与五种子重训、
paired bootstrap，以及 dev-locked 的一次性 official-test 复核。相同的任务、
8 个数据集、学生架构和推理机制必须披露为复用项，不能重复计算 novelty。

Post-lock 的 exposure-matched repeated-gold 与 same-text self-training control
同样属于 TALLIP 新增的 KD 机制证据，而不是新的学生架构贡献。

完整的逐项差异、prior-publication 建议文本、贡献—证据映射和禁止表述见
[`TALLIP_VS_ICONIP_V20.md`](TALLIP_VS_ICONIP_V20.md)。TALLIP 不得用 official
test 与 V20 dev 做跨 split 提升比较，也不得把本研究表述为新 SOTA。

#### 5.29.9 Post-lock exposure / self-training controls（2026-08-02—2026-08-13）

**研究问题**：adopted LLM-KD 的提升是否仅来自更多训练行/更新步数；如果在
完全相同的 silver 文本上使用学生 ensemble 自己的伪标签，是否能获得同等收益？
该协议在 official-test gold 公开后注册，但只使用既有 train、dev 和冻结预测，
`official_test_evaluation=forbidden`，不能用于修改任何 official-test 系统、阈值、
teacher、pool 或数据增强选择。

**Exposure-matched repeated-gold**：对五个 adopted KD 数据集，将当前 no-KD
训练配方确定性重复到与相应 KD 训练集完全相同的行数；保持学生 config、
五个 seeds（42/123/456/789/1024）和冻结 ensemble 阈值不变。这里控制的是
训练行曝光，不把重复样本误称为新增信息。rus/tat 的源文件是各自已采用的
增强 no-KD 配方，因此该对照相对“只重复原始 gold”更保守。

| Dataset | repeated-gold rows | 5-seed mean±SD | Ensemble | LLM-KD − control [95% CI] | 判定 |
|---|---:|---:|---:|---:|---|
| eng_laptop | 22,595 | 0.2971±0.0055 | 0.3154 | +0.0948 [+0.0494, +0.1426] | supported |
| rus_restaurant | 4,240 | 0.4897±0.0099 | 0.5238 | +0.0924 [+0.0247, +0.1634] | supported |
| tat_restaurant | 5,155 | 0.4581±0.0181 | 0.5256 | +0.0071 [−0.0621, +0.0754] | inconclusive |
| ukr_restaurant | 8,500 | 0.4122±0.0187 | 0.4641 | +0.0736 [+0.0036, +0.1507] | supported |
| zho_restaurant | 9,932 | 0.5084±0.0042 | 0.5478 | +0.0260 [−0.0028, +0.0543] | inconclusive |

因此 eng/rus/ukr 的 LLM-KD dev 优势不能仅由训练曝光解释；tat/zho 的 CI
跨 0，只能写“未建立可靠的 exposure-adjusted 优势”。这与 official-test
KD−baseline 的问题不同：post-lock 表专门检验训练曝光这一替代解释，不替换
§5.29.7 的一次性 test 结论。

**Same-text self-training**：在五个 adopted 数据集上，取 LLM 臂所用 silver
的完全相同文本和顺序，去掉 LLM 标签，由 repeated-gold 五种子 ensemble 生成伪标签；
再以相同行数、学生 config、seeds 和阈值训练。空预测保留为合法 no-quad 标签，
不做基于 dev 的过滤。伪标签非空率为 eng 92.3%、rus 82.8%、tat 75.9%、
ukr 89.3%、zho 96.7%。

| Dataset | Self-training mean±SD | Ensemble | LLM-KD − self [95% CI] | 判定 |
|---|---:|---:|---:|---|
| eng_laptop | 0.2973±0.0067 | 0.3169 | +0.0933 [+0.0476, +0.1407] | supported |
| rus_restaurant | 0.4767±0.0171 | 0.4980 | +0.1183 [+0.0531, +0.1854] | supported |
| tat_restaurant | 0.4268±0.0098 | 0.4758 | +0.0569 [−0.0203, +0.1402] | inconclusive |
| ukr_restaurant | 0.4525±0.0198 | 0.4940 | +0.0438 [−0.0231, +0.1069] | inconclusive |
| zho_restaurant | 0.5505±0.0096 | 0.5754 | −0.0016 [−0.0294, +0.0256] | inconclusive |

补充的 self-training−repeated-gold paired bootstrap 为：eng +0.0015
[−0.0174, +0.0215]、rus −0.0258 [−0.0773, +0.0229]、tat −0.0498
[−0.1085, +0.0143]、ukr +0.0298 [−0.0245, +0.0867]，均 inconclusive；
zho +0.0276 [+0.0079, +0.0463]，supported。由此可写：只有 eng/rus 的
LLM-KD 可靠优于 same-text self-training；tat/ukr/zho 均未建立 LLM teacher
相对学生伪标签的独占价值。不能外推为“LLM 总优于 self-training”或
“self-training 普遍足够”。

**Direct complementarity / error analysis**：五域 silver 的显式 aspect、
opinion、category inventory 与 VA 格式检查均为 0 异常；随后直接计算
base/silver 对 dev 的 category JSD、aspect/opinion 类型覆盖、row-level
repaired/harmed 和 multi-quad/rare/implicit slices。eng 的 dev span 覆盖增量
为 +21/+38，且 KD dev 增益最大；zho 的 direct teacher fidelity 最高，但
span 覆盖变化为 −30/−56，KD dev 增益最小。跨域指标不呈单调关系，因此这些
结果仅支持“质量、覆盖与互补性共同作用”的描述性解释，不证明单一因果指标。

**Multiplicity**：对 teacher fidelity（8）、adopted KD dev（5）、exposure
control（5）、same-text control（5）和冻结 official test（5）分别做 Holm
FWER 与 BH FDR。same-text family 中 eng/rus 在两种校正后保持显著；exposure
family 中 eng/rus 保持 Holm 显著，ukr 的单项 CI 虽下界大于 0，但 Holm
p=0.1542；official-test family 中 eng/tat/ukr 保持 Holm 显著，rus 不保持。
official-test multiplicity 是事后稳健性分析，不改变冻结系统或原单项 CI。

**Cost / efficiency**：8×2 fidelity 的最终 validated coverage 为每个 teacher
1,344 条、0 error；Qwen 的保留 timing metadata 只覆盖 1,264 条，吞吐率只按
这 1,264 条计算，不将其误称为最终覆盖。历史 silver token 数和账单未保留，
因此不虚构货币成本；学生部署阶段 LLM 调用为 0。post-lock 训练时长按日志
wall time 报告并明确包含 checkpoint I/O，不能作为跨硬件速度基准。

**审计与保留策略**：60/60 个 dev 预测均与 gold dev 的 ID、行数和顺序完全
一致；10 个 ensemble 均可由五个单 seed 预测按冻结阈值精确重建；50/50 个训练
目录均保留 `checkpoint_best.pt` 与完成标记。15 组 10,000 次 paired bootstrap
以 seed=20260713 重复两次，结果逐字段完全一致。训练收尾后删除 60 个
v1 `checkpoint_latest.pt`/`checkpoint_epoch*.pt`，释放 365,233,267,664 bytes；
v2 又删除 40 个此类文件名，逻辑大小 282,327,138,180 bytes，因 hardlink
去重实际释放 202,340,503,552 bytes。全部 best、预测和审计证据保留。

权威证据：`scripts/kd/postlock_v1.yaml`、`postlock_v2.yaml`、
`scripts/kd/postlock_experiments.py`、`tallip_postlock_analysis.py`，以及
`docs/audits/tallip_kd/postlock_complete_{metrics,audit}.json`、
`complementarity_error_analysis.json`、`statistical_robustness.json` 和
`cost_efficiency.json`。本节完成后不再增加常规 KD/augmentation 训练网格。

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

| 语种 | 推荐版本 | encoder | dev cF1 |
|---|---|---|---|
| **jpn** | V16 | ku-nlp/deberta-v2-large-japanese | **0.4466** |
| **rus** | V15 | xlm-roberta-large | **0.5267** |
| **zho_laptop** | V16 | hfl/chinese-roberta-wwm-ext-large | **0.4487** |
| **ukr** | V13b | microsoft/mdeberta-v3-base | **0.5068** |
| **tat** | V15 | xlm-roberta-large | **0.4821** |
| **5-lang mean** | per-lang mix | mixed | **0.4822** |

**意义**：
- 历史版本曾将 jpn/tat dev 与 test leaderboard 比较；该 cross-split 结论已撤回。released official test 分别为 0.4217/0.4311，均未超过榜单最佳。
- ✅ 5 语种平均 cF1 从 V15 时代 0.4604 → **0.4822** (+2.18 pp)
- ✅ 验证了**「per-lang best encoder + ensemble + per-lang sweep」**才是最优策略
- 5 语种 dev mean 仅用于仓库内部版本选择，不作为 leaderboard 竞争力证据。

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
| ❌ V21 jpn DeepSeek aug + LLM correction | +939 DeepSeek aug + 5-seed + Draft-Revision | 0.4318 | −1.82pp vs V16 no_aug 0.4500 | ❌ 未采用 |
| ❌ V22 rus/ukr/eng_laptop reruns | DeepSeek aug 或 raw large 5-seed | 最高 0.5063 (ukr) | 均未超过当前采用最佳 | ❌ 未替换 |

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

### 7.7 V21/V22 补充消融（Gemini 后续实验）

#### 7.7.1 LLM 后处理纠错（jpn_hotel）

| 方案 | 输入先验 | cF1 | Δ vs 5-seed ensemble | 结论 |
|---|---|---:|---:|---|
| 5-seed logit ensemble | 模型融合输出 | **0.4315** | — | 基准 |
| LLM V1 zero-shot correction | 不保留 category 先验 | 0.2919 | −13.96 pp | 大幅破坏预测 |
| LLM V2 Draft-Revision | 保留模型草稿，仅做保守修正 | 0.4318 | +0.03 pp | 安全但收益极小 |

**结论**：LLM 适合做保守的边界清洗，不适合替代模型做 category 决策。Draft-Revision 是正确方向，但当前收益只有 +0.0003，论文中应作为负面/边界实验，而不是主方法贡献。

#### 7.7.2 DeepSeek 增强在不同语言上的边际收益

| Dataset | DeepSeek 增强方案 | cF1 | 对照最佳 | Δ | 结论 |
|---|---|---:|---:|---:|---|
| jpn_hotel | DeepSeek aug + 5-seed | 0.4318 | 0.4500 (V16 no_aug) | −1.82 pp | 强 native encoder 下增强仍为噪声 |
| ukr_restaurant | DeepSeek aug + 5-seed | 0.5063 | 0.5068 (V13b) | −0.05 pp | 近似持平，未超越 |
| rus_restaurant | DeepSeek aug + 5-seed | 0.4696 | 0.5267 (V15) | −5.71 pp | encoder 选择比新增增强更关键 |
| eng_laptop | raw large rerun | 0.3056 | 0.3304 (V5/V6 BT) | −2.48 pp | 未恢复历史最佳 |

**结论**：DeepSeek 增强不是可迁移的稳定增益。它在不同语言上的效果受 encoder、类别空间、原始数据规模和已有增强配方共同制约；后续应优先做 per-language 验证，而不是默认扩展。

---

### 7.8 Final Ablation 总结表（论文用）

| Ablation | 在 mDeBERTa-base (V3) | 在 V16 native encoder | 解读 |
|---|---|---|---|
| Rethink 模块 | **+11.29 pp** (§7.1) | +0.36 pp ensemble | 大模型自带 ≈ 弱 Rethink |
| Data augmentation (paraphrase+COMET) | **+4.53 pp** (§5.18, jpn) | −0.34 pp ensemble (jpn) | ensemble 已吃掉 reg 好处 |
| 5-seed ensemble | +1.4 pp (§5.7, baseline) | **+4.76 pp** (§7.6.1, 8-dataset) | **encoder 越强，ensemble 收益越大** |
| Native encoder | n/a | **+5.38 pp** (jpn) / **+5.83 pp** (zho_lap) | **核心贡献** |

**论文 narrative**：旧的方法学（V3 时代）的两大 trick（Rethink、augmentation）在大 encoder 时代衰减为接近零。但 **per-language native encoder + 5-seed ensemble** 这两个简单选择共同贡献了 ~10 pp 的提升。

---

### 7.9 历史 pre-clean 冻结与 rus silver 批次质量审计（2026-07-13）

当时的 8 数据集方案冻结到 [`submission/final_manifest.json`](../submission/final_manifest.json)，包含每个预测文件和 config 的 SHA-256、checkpoint 前缀、vote/topk 与 dev cF1。该 manifest 现仅代表历史 pre-clean submission，不是 TALLIP clean-inductive/official-test 权威源。只读审计覆盖 8 个 JSONL、6,690 行、ZIP 成员字节、test ID/顺序、schema/VA/category 和 checkpoint 存在性，结果为 **0 error、2 warning**；两项 warning 均来自已删除的 eng_restaurant 与 jpn_hotel 历史五种子 checkpoint，冻结预测本身及 ZIP 字节一致。

rus 同池 stopping gate 的 paired mean 为：full-gen8k **0.5467**，cap3000 **0.5353**，变化 **−1.14pp**，未达到预设的 **+1.5pp** 继续门槛。因此停止剩余 3 seeds 与 cap5000，不再做 dose-response。最终保留 gen4k cap3000 五种子 ensemble **0.6162**；full-gen8k ensemble 为 **0.5858**。

低成本质量审计显示：gen4k/gen8k 分别有 4,000/8,000 行，规范化重复均为 0；自动有效性标记为 10/31 行；相对 gold 的 category JSD 为 **0.0651/0.0719**。这些代理信号略偏向 gen4k，但差异小，不能独立证明语义标注更准。因此正式结论限定为：**性能差更符合生成批次质量敏感性，而非通用的 cap=3000 最优规律**。

最终论文/汇报口径以 §5.29.5–§5.29.7 的六条证据收敛及 [`docs/CONCLUSIONS.md`](CONCLUSIONS.md) 为准；本轮不修改 LaTeX。

- 可执行 notebook：[`notebooks/rus_silver_quality_audit.ipynb`](../notebooks/rus_silver_quality_audit.ipynb)
- 技术报告：[`docs/audits/kd_freeze_report/report.html`](audits/kd_freeze_report/report.html)
- 最终 KD 汇报：[`docs/KD_ablation_final_20260713.pptx`](KD_ablation_final_20260713.pptx)
- 100 条平衡人工复核样本：[`docs/audits/rus_silver_manual_sample_100.jsonl`](audits/rus_silver_manual_sample_100.jsonl)（尚未人工判定，不报告人工 precision）

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
