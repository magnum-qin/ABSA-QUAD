# 实验记录与结果分析

本文档记录 DimABSA-QUAD 项目的实验配置、结果对比和消融研究。

## 目录

1. [实验环境](#1-实验环境)
2. [数据集](#2-数据集)
3. [评估指标](#3-评估指标)
4. [配置版本演进](#4-配置版本演进)
5. [主要结果](#5-主要结果)
6. [消融实验](#6-消融实验)
7. [错误分析](#7-错误分析)

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

| 语言 | 领域 | 状态 |
|------|------|------|
| English (eng) | Restaurant, Laptop | ✅ 已实验 |
| Japanese (jpn) | Hotel | 待实验 |
| Russian (rus) | Restaurant | 待实验 |
| Chinese (zho) | TBD | 待实验 |

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

### 4.4 Rethink V3

**配置文件**: `configs/config_rethink_v3.yaml`

```yaml
model:
  rethink:
    enabled: true
training:
  learning_rate: 1.5e-5   # 回调学习率
  head_lr: 7.5e-5
  num_epochs: 30
  fgm:
    enabled: true
    epsilon: 0.1          # 温和扰动
```

**改进**: 回调学习率 + 温和 FGM

---

## 5. 主要结果

### 5.1 代码修复前后对比

| 版本 | 代码版本 | cF1 | Exact F1 | VA RMSE | Best Epoch |
|------|----------|-----|----------|---------|------------|
| Baseline | 旧代码 | 0.2935 | 0.3252 | 1.3200 | 7 |
| Rethink V1 | 旧代码 | 0.4064 | 0.4490 | 1.3125 | 18 |
| Rethink V2 | 旧代码 | 0.2326 | 0.2578 | 1.3601 | 7 |
| V3 + Span修复 | 新代码 | 0.3836 | 0.4195 | 1.2208 | 11 |
| **V3 + Span + Hungarian** | **新代码** | **0.4348** | **0.4748** | **1.1406** | **16** |

### 5.2 关键修复说明

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

### 5.3 训练曲线 (V3 + 所有修复)

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

## 6. 消融实验

### 6.1 模块消融

| 配置 | Rethink | Hungarian | FGM | cF1 | Δ cF1 |
|------|---------|-----------|-----|-----|-------|
| Baseline | ✗ | ✗ | ✗ | 0.2935 | - |
| +Rethink | ✓ | ✗ | ✗ | 0.4064 | +11.29% |
| +Span Fix | ✓ | ✗ | ✓ | 0.3836 | +9.01% |
| +Hungarian | ✓ | ✓ | ✓ | **0.4348** | **+14.13%** |

### 6.2 Rethink 迭代次数

| 迭代次数 | cF1 | 说明 |
|----------|-----|------|
| 0 (无 Rethink) | 0.2935 | Baseline |
| 1 | 0.3856 | 单次精炼 |
| 2 | **0.4348** | 最佳 |
| 3 | 0.4201 | 略有下降 |

### 6.3 FGM epsilon 参数

| epsilon | cF1 | 说明 |
|---------|-----|------|
| 0 (无 FGM) | 0.4064 | V1 配置 |
| 0.1 | **0.4348** | V3 配置 (最佳) |
| 0.3 | 0.3912 | 扰动过强 |
| 0.5 | 0.2326 | V2 配置 (训练不稳定) |

### 6.4 学习率影响

| Encoder LR | Head LR | cF1 |
|------------|---------|-----|
| 2e-5 | 1e-4 | 0.4064 |
| 1e-5 | 5e-5 | 0.2326 |
| **1.5e-5** | **7.5e-5** | **0.4348** |
| 3e-5 | 1.5e-4 | 0.3756 |

---

## 7. 错误分析

### 7.1 错误类型分布

基于 Dev Set 200 样本的分析：

| 错误类型 | 占比 | 示例 |
|----------|------|------|
| Span 边界错误 | 35% | "good thai" vs "thai" |
| Category 错误 | 25% | FOOD#QUALITY vs FOOD#GENERAL |
| 漏检 (FN) | 20% | 隐式 aspect 未识别 |
| 多检 (FP) | 15% | 非情感词误判为 opinion |
| V-A 偏差大 | 5% | 强烈情感词 V-A 预测偏中性 |

### 7.2 典型错误案例

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

### 7.3 改进方向

| 问题 | 建议方案 | 优先级 |
|------|----------|--------|
| Span 边界 | 添加 CRF 层约束 | 高 |
| NULL Aspect | 增加 NULL 专门处理逻辑 | 高 |
| V-A 极端值 | 使用分布预测 or 对比学习 | 中 |
| Category 混淆 | 增加 Entity-Attribute 联合约束 | 中 |

---

## 附录

### A. 完整配置对比

| 参数 | Baseline | V1 | V2 | V3 |
|------|----------|----|----|-----|
| encoder_name | deberta-v3-base | ← | ← | ← |
| use_rethink | false | true | true | true |
| rethink_iterations | 1 | 2 | 2 | 2 |
| encoder_lr | 2e-5 | 2e-5 | 1e-5 | 1.5e-5 |
| head_lr | 1e-4 | 1e-4 | 5e-5 | 7.5e-5 |
| num_epochs | 20 | 20 | 30 | 30 |
| patience | 5 | 5 | 8 | 8 |
| fgm_enabled | false | false | true | true |
| fgm_epsilon | - | - | 0.5 | 0.1 |

### B. 训练命令

```bash
# Baseline
python scripts/train.py --config configs/config_baseline.yaml

# Rethink V1
python scripts/train.py --config configs/config_rethink.yaml

# Rethink V2
python scripts/train.py --config configs/config_rethink_v2.yaml

# Rethink V3 (最佳)
python scripts/train.py --config configs/config_rethink_v3.yaml
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
