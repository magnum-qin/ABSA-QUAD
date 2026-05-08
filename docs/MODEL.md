# DimABSA-QUAD 模型架构详解

本文档详细说明 DimABSA-QUAD 模型的完整架构、各模块设计原理及实现细节。

## 目录

1. [任务定义](#1-任务定义)
2. [整体架构](#2-整体架构)
3. [数据流程](#3-数据流程)
4. [核心模块](#4-核心模块)
   - 4.1 [编码器 (Encoder)](#41-编码器-encoder)
   - 4.2 [Span 提取器 (SpanExtractor)](#42-span-提取器-spanextractor)
   - 4.3 [Pair 分类器 (PairClassifier)](#43-pair-分类器-pairclassifier)
   - 4.4 [V-A 回归器 (VARegressor)](#44-v-a-回归器-varegressor)
   - 4.5 [Feature-level Rethink 模块](#45-feature-level-rethink-模块)
5. [训练策略](#5-训练策略)
   - 5.1 [Hungarian 匹配](#51-hungarian-匹配)
   - 5.2 [损失函数](#52-损失函数)
   - 5.3 [FGM 对抗训练](#53-fgm-对抗训练)
6. [推理流程](#6-推理流程)

---

## 1. 任务定义

### 1.1 DimASQP 任务

**Dimensional Aspect Sentiment Quad Prediction (DimASQP)** 是 SemEval-2026 Task 3 的核心任务，要求从文本中提取四元组：

```
(Aspect, Category, Opinion, V-A Score)
```

| 元素 | 说明 | 示例 |
|------|------|------|
| **Aspect** | 评价对象/目标 | "thai food", "service" |
| **Category** | 实体#属性格式 | "FOOD#QUALITY", "SERVICE#GENERAL" |
| **Opinion** | 情感表达词 | "delicious", "terrible" |
| **V-A Score** | 效价-唤醒度 (1-9) | "7.50#6.25" |

### 1.2 V-A 情感维度

与传统的离散情感标签 (正/负/中) 不同，DimABSA 使用二维连续情感空间：

```
        Arousal (唤醒度)
           9 |  excited, thrilled
             |       ↑
           5 |  calm, neutral
             |       ↓
           1 |  sleepy, bored
             +------------------→
             1       5       9
                Valence (效价)
              negative  positive
```

- **Valence (效价)**: 情感极性，1=极负面，9=极正面
- **Arousal (唤醒度)**: 情感强度，1=平静，9=激动

---

## 2. 整体架构

### 2.1 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Input Text                                   │
│            "The thai food was delicious but service terrible"        │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Tokenizer (DeBERTa-v3)                           │
│                 [CLS] The thai food was ... [SEP]                   │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DeBERTa-v3 Encoder                             │
│                                                                      │
│   Input:  input_ids [B, L], attention_mask [B, L]                   │
│   Output: hidden_states [B, L, 768]                                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  SpanExtractor  │  │  PairClassifier │  │   VARegressor   │
│                 │  │                 │  │                 │
│  BIO Tagging    │  │  Entity Class   │  │  Valence Head   │
│  [B,L,5]        │  │  Attribute Class│  │  Arousal Head   │
│                 │  │  [B,P,7]/[B,P,6]│  │  [B,P,1]/[B,P,1]│
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
                    Stage 1 Predictions
                              │
         ┌────────────────────┴────────────────────┐
         │         Feature-level Rethink           │
         │                                         │
         │  1. Label Embedding (soft embedding)    │
         │  2. Feature Fusion (gated)              │
         │  3. Self-Attention Refinement           │
         └────────────────────┬────────────────────┘
                              ▼
                    Stage 2 Predictions
                              │
                              ▼
              ┌───────────────────────────────┐
              │     Output Quadruplets        │
              │  (Aspect, Category, Opinion,  │
              │         Valence#Arousal)      │
              └───────────────────────────────┘
```

### 2.2 两阶段预测机制

模型采用 **Draft-then-Refine** 策略：

| 阶段 | 输入 | 输出 | 作用 |
|------|------|------|------|
| **Stage 1** | 编码器隐状态 | 初步预测 (Draft) | 快速生成候选结果 |
| **Stage 2** | 隐状态 + Stage 1 预测 | 精细预测 (Refined) | 利用标签信息修正 |

---

## 3. 数据流程

### 3.1 数据预处理

```python
# 文件: src/data/dataset.py

Input JSONL:
{
    "ID": "R001",
    "Text": "average to good thai food, but terrible delivery.",
    "Quadruplet": [
        {"Aspect": "thai food", "Category": "FOOD#QUALITY",
         "Opinion": "average to good", "VA": "6.75#6.38"},
        {"Aspect": "delivery", "Category": "SERVICE#GENERAL",
         "Opinion": "terrible", "VA": "2.88#6.62"}
    ]
}
```

### 3.2 Tokenization

```
Text: "average to good thai food, but terrible delivery."
      ↓ DeBERTa Tokenizer
Tokens: [CLS] average to good thai food , but terrible delivery . [SEP]
IDs:    [1]   2087   268 1070 13679 1978 6 46 12404  3985   5  [2]
Positions: 0    1     2   3    4     5   6  7   8      9    10  11
```

### 3.3 Span 匹配算法

**问题**: 文本中可能有重复的 span（如 "good food and good service" 中的两个 "good"）

**解决方案**: 使用位置追踪避免重复匹配

```python
def _find_span_positions(self, text, span, encoding, used_positions):
    """
    查找 span 在文本中的 token 位置，避免重复匹配

    1. 找到 span 的所有字符级出现位置
    2. 过滤已被占用的位置
    3. 转换为 token 级位置
    4. 标记该位置为已使用
    """
    occurrences = self._find_all_occurrences(text, span)

    for start_char, end_char in occurrences:
        # 检查是否与已用位置重叠
        if not self._overlaps_with_used(start_char, end_char, used_positions):
            start_token = encoding.char_to_token(start_char)
            end_token = encoding.char_to_token(end_char - 1)
            if start_token and end_token:
                used_positions.add((start_char, end_char))
                return start_token, end_token

    return -1, -1
```

### 3.4 BIO 标签生成

```
Tokens:    [CLS] average to good thai food , but terrible delivery . [SEP]
BIO Labels:  O    B-OPN  I-OPN I-OPN B-ASP I-ASP O  O   B-OPN  B-ASP   O   O

标签说明:
- O: Outside (非目标词)
- B-ASP: Begin-Aspect (Aspect 起始)
- I-ASP: Inside-Aspect (Aspect 内部)
- B-OPN: Begin-Opinion (Opinion 起始)
- I-OPN: Inside-Opinion (Opinion 内部)
```

---

## 4. 核心模块

### 4.1 编码器 (Encoder)

**文件**: `src/models/dimabsa_model.py`

**选择**: DeBERTa-v3-base (Microsoft)

**优势**:
- Disentangled Attention: 分离内容和位置的注意力计算
- Enhanced Mask Decoder: 更好的 MLM 预训练
- 在 NLU 任务上超越 RoBERTa 和 BERT

```python
self.encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
# 输出: hidden_states [batch_size, seq_len, 768]
```

### 4.2 Span 提取器 (SpanExtractor)

**文件**: `src/models/dimabsa_model.py` (class `SpanExtractor`)

**任务**: 序列标注，识别 Aspect 和 Opinion 的边界

**架构**:
```
hidden_states [B, L, 768]
       ↓
   Dropout(0.1)
       ↓
   Linear(768 → 384)
       ↓
     GELU
       ↓
   Dropout(0.1)
       ↓
   Linear(384 → 5)
       ↓
bio_logits [B, L, 5]  # 5 = O, B-ASP, I-ASP, B-OPN, I-OPN
```

**Span 解码**:
```python
def extract_spans(self, logits, attention_mask):
    """
    从 BIO logits 解码出 span 列表

    Returns:
        List of (start_idx, end_idx, span_type)
        span_type: "aspect" or "opinion"
    """
    predictions = torch.argmax(logits, dim=-1)
    # B-* 标签开始新 span
    # I-* 标签延续当前 span
    # O 标签结束当前 span
```

### 4.3 Pair 分类器 (PairClassifier)

**文件**: `src/models/dimabsa_model.py` (class `PairClassifier`)

**任务**: 对每个 (Aspect, Opinion) pair 进行分类

**输入**: Aspect 和 Opinion 的表示向量（span 内 token 的平均池化）

**架构**:
```
aspect_repr [B, P, 768] + opinion_repr [B, P, 768]
                    ↓ Concatenate
            pair_repr [B, P, 1536]
                    ↓
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│  Pair   │   │ Entity  │   │Attribute│
│ Valid?  │   │ Class   │   │  Class  │
│ [B,P,2] │   │ [B,P,7] │   │ [B,P,6] │
└─────────┘   └─────────┘   └─────────┘
```

**Entity 标签** (eng_restaurant, 字母序): AMBIENCE, DRINKS, FOOD, LOCATION, RESTAURANT, SERVICE, NULL

**Attribute 标签** (eng_restaurant, 字母序): GENERAL, MISCELLANEOUS, PRICES, QUALITY, STYLE_OPTIONS, NULL

> 注意: 不同领域/语言有不同标签集，由 config 中 `data.entity_labels` / `data.attribute_labels` 指定。

### 4.4 V-A 回归器 (VARegressor)

**文件**: `src/models/va_module.py`

**任务**: 预测连续的 Valence 和 Arousal 值

**架构**:
```
pair_repr [B, P, 1536]
         ↓
    Shared layer
    Linear(1536 → 256)
    GELU + Dropout
    LayerNorm                       ← 注意：源码末尾有 LN
         ↓
    shared_features [B, P, 256]
         │
    ┌────┴─────┐
    ▼          ▼
┌─────────┐ ┌─────────┐
│ Valence │ │ Arousal │
│  Head   │ │  Head   │
│ Lin 256 │ │ Lin 256 │
│   →128  │ │   →128  │             ← 实际是两层 MLP，不是单层
│  GELU + │ │  GELU + │
│ Dropout │ │ Dropout │
│ Lin 128 │ │ Lin 128 │
│   → 1   │ │   → 1   │
└────┬────┘ └────┬────┘
     ↓           ↓
  σ(·)·8+1   σ(·)·8+1               ← sigmoid 缩放（不是 clamp）
   ∈[1,9]     ∈[1,9]
```

**输出范围限制**（由 sigmoid 直接保证，不使用 clamp）:
```python
# va_module.py:106-107
valence = torch.sigmoid(valence_logits) * self.va_range + self.va_min  # va_range=8
arousal = torch.sigmoid(arousal_logits) * self.va_range + self.va_min
```

> 文档历史版本曾写成 `clamp([1,9])`，与源码不符；实际通过 `sigmoid(x)·8+1` 平滑映射到 `[1, 9]`，避免边界处梯度截断。

### 4.5 Feature-level Rethink 模块

**文件**: `src/models/feature_rethink.py`

**核心思想**: 利用 Stage 1 的预测结果作为额外特征，增强 Stage 2 的预测

#### 4.5.1 Label Embedding

将 Stage 1 的预测概率转换为可学习的嵌入向量：

```python
class LabelEmbeddingModule:
    # BIO 标签嵌入（token-level，参与 token_fusion）
    self.bio_embedding = nn.Embedding(5, 64)  # 5 BIO labels → 64d

    # Pair-level 嵌入（pair_valid / entity / attribute）
    self.pair_embedding      = nn.Embedding(2, 64)  # valid / invalid
    self.entity_embedding    = nn.Embedding(7, 64)
    self.attribute_embedding = nn.Embedding(6, 64)

    # V-A 值投影
    self.va_projector = nn.Linear(2, 64)

    # Fusion 层
    self.token_fusion = nn.Linear(hidden_dim + 64, hidden_dim)         # ✓ 生效
    self.pair_fusion  = nn.Linear(pair_dim + 64*4, hidden_dim)         # ⚠ 已定义但未使用

    def embed_bio_labels(self, bio_probs):
        """软嵌入: 使用概率加权"""
        # bio_probs: [B, L, 5]   bio_embedding.weight: [5, 64]
        return torch.matmul(bio_probs, self.bio_embedding.weight)
        # 输出: [B, L, 64]
```

> ⚠ 实现注意：`forward()` 中虽然计算了 `fused_pair = pair_fusion([pair_repr; entity_emb; attribute_emb; pair_emb; va_emb])`，
> 但返回给上层 (`DimASQPModel`) 后**未被任何 stage-2 head 使用**。
> 实际生效的 Rethink 路径只有 token-level 融合（§4.5.2 → §4.5.3 → §4.5.4）。
> 论文中应仅描述 token-level 融合分支以与代码一致。

#### 4.5.2 Feature Fusion

```
Original Hidden [B, L, 768] + BIO Embedding [B, L, 64]
                    ↓ Concatenate
              [B, L, 832]
                    ↓
           Linear(832 → 768)
                    ↓
           Fused Hidden [B, L, 768]
```

#### 4.5.3 Self-Attention Refinement

```
Fused Hidden [B, L, 768]
         ↓
Multi-Head Self-Attention (8 heads)
         ↓
   LayerNorm + Residual
         ↓
   Feed-Forward Network
   Linear(768 → 3072) → GELU → Linear(3072 → 768)
         ↓
   LayerNorm + Residual
         ↓
Refined Hidden [B, L, 768]
```

#### 4.5.4 Gated Fusion

控制原始特征和精炼特征的混合比例：

```python
# 计算门控权重
gate_input = torch.cat([original_hidden, refined_hidden], dim=-1)
gate = self.gate(gate_input)  # Linear(1536 → 768) + Sigmoid

# 加权融合
output = gate * refined_hidden + (1 - gate) * original_hidden
```

---

## 5. 训练策略

### 5.1 Hungarian 匹配

**问题**: 模型预测的 pairs 是 Aspect × Opinion 的笛卡尔积，与 gold pairs 顺序不对应

**解决方案**: 使用 Hungarian 算法进行最优二分匹配

```python
from scipy.optimize import linear_sum_assignment

def _compute_pair_cost_matrix(self, pred_spans, gold_pairs, ...):
    """
    计算代价矩阵 [num_gold, num_pred]

    代价 = 2.0 * span_cost      # Span IoU 距离
         + 1.0 * entity_cost    # Entity 分类损失
         + 1.0 * attr_cost      # Attribute 分类损失
         + 0.5 * va_cost        # V-A 回归距离
    """

# 匹配
row_indices, col_indices = linear_sum_assignment(cost_matrix)
# row_indices: 匹配的 gold pair 索引
# col_indices: 匹配的 pred pair 索引
```

**Span IoU 计算**:
```python
def _compute_span_iou(self, span1, span2):
    """
    计算两个 span 的 IoU (Intersection over Union)

    span1 = (start1, end1), span2 = (start2, end2)
    intersection = max(0, min(end1, end2) - max(start1, start2) + 1)
    union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection
    IoU = intersection / union
    """
```

### 5.2 损失函数

**总损失** (V3):
```
L_total = 2.0 * L_span + 1.0 * L_entity + 1.0 * L_attribute + 1.0 * L_va + 0.5 * L_consistency
```

> V3 通过 loss weight sweep 确定 span=2.0 为最优权重（cF1 +4.25%）。

| 损失项 | 公式 | 作用 |
|--------|------|------|
| L_bio | CrossEntropy(bio_logits, bio_labels) | BIO 序列标注 |
| L_entity | CrossEntropy(entity_logits, entity_labels) | Entity 分类 |
| L_attribute | CrossEntropy(attr_logits, attr_labels) | Attribute 分类 |
| L_va | HuberLoss(pred_va, gold_va) | V-A 回归 |
| L_consistency | max(0, L_s2 - L_s1 + margin) | Stage 2 应优于 Stage 1 |

**Huber Loss** (用于 V-A 回归):
```
           { 0.5 * (y - ŷ)²           if |y - ŷ| < δ
L_huber = {
           { δ * |y - ŷ| - 0.5 * δ²   otherwise
```
- 对小误差使用 MSE（敏感）
- 对大误差使用 MAE（鲁棒）

### 5.3 FGM 对抗训练

**文件**: `scripts/train.py` (class `FGM`)

**Fast Gradient Method**: 在 embedding 层添加对抗扰动

```python
class FGM:
    def attack(self):
        """添加对抗扰动"""
        for name, param in model.named_parameters():
            if 'word_embeddings' in name and param.requires_grad:
                # 保存原始 embedding
                self.backup[name] = param.data.clone()
                # 计算扰动
                norm = torch.norm(param.grad)
                r_at = self.epsilon * param.grad / norm
                # 添加扰动
                param.data.add_(r_at)

    def restore(self):
        """恢复原始 embedding"""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
```

**训练流程**:
```
1. 前向传播 → 计算 loss → 反向传播
2. FGM attack (添加扰动)
3. 前向传播 → 计算 adv_loss → 反向传播
4. FGM restore (恢复 embedding)
5. 梯度更新
```

### 5.4 EMA (Exponential Moving Average)

维护模型参数的指数移动平均，用于评估和推理：

```python
shadow[name] = decay * shadow[name] + (1 - decay) * param.data
# decay = 0.999
```

评估时使用 shadow 参数替代原始参数，推理时优先加载 EMA 权重。

### 5.5 Label Smoothing

分类损失使用 label_smoothing=0.1，将 one-hot 标签平滑为：

```
y_smooth = (1 - 0.1) * y_onehot + 0.1 / num_classes
```

防止模型对分类过度自信。

### 5.6 Top-K Checkpointing + Weight Averaging

训练时保留 cF1 最高的 K 个 checkpoint (默认 K=3)。推理时可对这 K 个 checkpoint 的权重取平均 (ensemble)：

```bash
python scripts/inference.py --ensemble --checkpoint_dir checkpoints/all_lang/eng_restaurant
```

---

## 6. 推理流程

```python
def predict(self, input_ids, attention_mask, texts, offset_mapping):
    """
    推理流程:

    1. 编码器前向传播
       hidden_states = encoder(input_ids, attention_mask)

    2. Stage 1 预测
       bio_logits = span_extractor(hidden_states)
       spans = extract_spans(bio_logits)

    3. 构建 Pair 表示
       aspect_repr, opinion_repr = get_span_representations(hidden_states, spans)

    4. Pair 分类 + V-A 预测
       entity_logits, attr_logits = pair_classifier(aspect_repr, opinion_repr)
       valence, arousal = va_regressor(pair_repr)

    5. (可选) Feature-level Rethink
       refined_hidden = rethink_module(hidden_states, stage1_predictions)
       # 重复 2-4 步

    6. 后处理
       - 过滤无效 pairs
       - Token 位置 → 原文 span
       - 格式化输出
    """
```

**Token → Text 映射**:
```python
def _tokens_to_text(self, text, offset_mapping, start_tok, end_tok):
    """
    使用 offset_mapping 将 token 位置转换回原文

    offset_mapping[i] = (char_start, char_end) 表示第 i 个 token 对应的字符范围
    """
    char_start = offset_mapping[start_tok][0]
    char_end = offset_mapping[end_tok][1]
    return text[char_start:char_end].strip()
```

---

## 附录

### A. 配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| encoder_name | deberta-v3-base | 预训练编码器 |
| hidden_size | 768 | 隐层维度 |
| num_entity_labels | 7 | Entity 类别数 |
| num_attribute_labels | 6 | Attribute 类别数 |
| va_hidden_size | 256 | V-A 回归器隐层 |
| label_embedding_dim | 64 | Rethink 标签嵌入维度 |
| num_rethink_iterations | 2 | Rethink 迭代次数 |
| dropout | 0.1 | Dropout 率 |

### B. 模型参数量

| 模块 | 参数量 |
|------|--------|
| DeBERTa-v3-base | ~86M |
| SpanExtractor | ~300K |
| PairClassifier | ~2.4M |
| VARegressor | ~400K |
| FeatureLevelRethink | ~7M |
| **Total** | **~96M** |
