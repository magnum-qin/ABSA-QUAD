# README_AI.md — AI 快速理解指南

> 本文件供 AI 助手快速建立项目全貌认知，包含架构、数据流、已实现功能和创新方向。
> 人类文档请见 [README.md](README.md) 和 [docs/](docs/)。
> KD 结论必须以 [docs/CONCLUSIONS.md](docs/CONCLUSIONS.md) 为准；历史 dev
> 记录不得与 official test leaderboard 直接比较。
> TALLIP novelty 必须以
> [docs/TALLIP_VS_ICONIP_V20.md](docs/TALLIP_VS_ICONIP_V20.md) 为准：学生架构、
> language-specific recipe 和 five-seed ensemble 属于 ICONIP V20，不得重复
> 声称为 TALLIP 新贡献。

---

## 0. 任务定义

**项目**: SemEval-2026 Task 3 — Dimensional Aspect Sentiment Quad Prediction (DimASQP)

**核心任务**: 从文本中提取情感四元组，将离散极性替换为连续 Valence-Arousal (VA) 值：

```
Input:  "The food was great but service was slow."
Output: [
  (food, FOOD#QUALITY, great,   8.0#5.5),   # (Aspect, Category, Opinion, V#A)
  (service, SERVICE#GENERAL, slow, 3.5#4.0)
]
```

- **Valence (V)**: 1(极负) → 5(中性) → 9(极正)
- **Arousal (A)**: 1(极平静) → 5(中等) → 9(极激动)
- **Category**: `ENTITY#ATTRIBUTE` 格式，Entity={RESTAURANT,FOOD,DRINKS,AMBIENCE,SERVICE,LOCATION,NULL}，Attribute={GENERAL,PRICES,QUALITY,STYLE_OPTIONS,MISCELLANEOUS,NULL}

**主评估指标**: Continuous F1 (cF1)
```
cTP = 1 - dist(VA_pred, VA_gold) / D_max       # D_max = sqrt(128) ≈ 11.31
cF1 = 2 × cPrecision × cRecall / (cPrecision + cRecall)
```
分类三元组 (Aspect, Category, Opinion) 匹配时，VA 误差越小则 cTP 越高（最高=1）。

---

## 1. 项目结构与关键文件

```
ABSA-QUAD/
├── src/
│   ├── models/
│   │   ├── dimabsa_model.py     # 主模型：DimASQPModel（入口）
│   │   ├── va_module.py         # VA回归头：VARegressor, VAEmbedding, VAConsistencyVerifier
│   │   └── feature_rethink.py  # Rethink模块：FeatureLevelRethink, LabelEmbeddingModule, ConsistencyLoss
│   ├── data/
│   │   └── dataset.py           # 数据集：DimASQPDataset, collate_fn, create_dataloader
│   └── evaluation/
│       └── metrics.py           # 评估：DimASQPEvaluator（cF1/ExactF1/RMSE）
├── scripts/
│   ├── train.py                 # 训练脚本（含EMA, FGM, Trainer类）
│   └── inference.py             # 推理脚本（生成 SemEval 格式 JSONL）
├── configs/
│   ├── config_baseline.yaml     # 无 Rethink/FGM 基线
│   ├── config_rethink.yaml      # V1: Rethink
│   ├── config_rethink_v2.yaml   # V2: Rethink + 强 FGM
│   └── config_rethink_v3.yaml   # V3: Rethink + 温和 FGM + EMA（当前最优）
└── data/subtask_3/eng/
    ├── eng_restaurant_train_alltasks.jsonl  # 训练集 (~2284 样本)
    ├── eng_restaurant_dev_task3.jsonl       # 验证集 (~200 样本)
    └── eng_restaurant_test_task3.jsonl      # 本地任务输入副本；released gold 独立审计
```

---

## 2. 模型架构

### 2.1 总体结构（两阶段）

```
Input Text
    │
    ▼
[DeBERTa-v3-base encoder]  ← 骨干模型，输出 hidden_states [B, L, 768]
    │
    ├──────────────────────────────────────────────────────┐
    │           Stage 1: Draft Prediction                  │
    │                                                      │
    │  SpanExtractor (BIO) → 提取 Aspect/Opinion spans     │
    │  ↓                                                   │
    │  _get_span_representations → 笛卡尔积组合 A×O pairs  │
    │  ↓                                                   │
    │  PairClassifier → pair_valid / entity / attribute    │
    │  ↓                                                   │
    │  VARegressor → valence, arousal ∈ [1,9]              │
    │                                                      │
    └──────────────────────────────────────────────────────┘
              │  (bio_probs, pair_probs, entity_probs, attr_probs, V, A)
              ▼
[FeatureLevelRethink]  ← 核心创新模块
    │
    ├── LabelEmbeddingModule: 将 Stage1 预测概率转为 soft embeddings
    ├── 与 hidden_states 融合
    ├── MultiheadAttention (8 heads) 精炼
    └── Gated Fusion: gate × refined + (1-gate) × original
              │
              ▼
    ┌──────────────────────────────────────────────────────┐
    │           Stage 2: Refined Prediction                │
    │                                                      │
    │  SpanExtractor_stage2 → 重新提取 spans               │
    │  PairClassifier_stage2 → 重新分类                    │
    │  VARegressor_stage2 → 重新预测 VA                    │
    │                                                      │
    └──────────────────────────────────────────────────────┘
              │
              ▼
       Final Quadruplets
```

### 2.2 关键模块说明

#### SpanExtractor (`dimabsa_model.py:18`)
- 输入: `[B, L, 768]`
- 输出: BIO logits `[B, L, 5]`（标签: O, B-ASP, I-ASP, B-OPN, I-OPN）
- `extract_spans()` 方法将 BIO 预测转为 span 列表 `[(start, end, type), ...]`

#### PairClassifier (`dimabsa_model.py:108`)
- 输入: aspect_repr + opinion_repr，各 `[B, num_pairs, 768]`（拼接后 1536 维）
- 输出: pair_valid logits (2类) + entity logits (7类) + attribute logits (6类)

#### VARegressor (`va_module.py:12`)
- 输入: pair_repr `[B, num_pairs, 1536]`
- 共享层: `Linear(1536→256) → GELU → Dropout → LayerNorm`
- 每个 head（valence/arousal 各一）: `Linear(256→128) → GELU → Dropout → Linear(128→1)`
  （**两层 MLP，不是单层** —— 文档历史版本曾误写成单层）
- 输出激活: `sigmoid(x) * 8 + 1`，范围约束到 [1, 9]（不使用 clamp）
- 损失: Huber Loss (delta=1.0)

#### VAEmbedding (`va_module.py:168`)
- 将连续 VA 值转为 embedding（混合离散+连续）
- 离散化为 18 个 bin（0.5 间隔），用于 Rethink 模块的软嵌入输入

#### FeatureLevelRethink (`feature_rethink.py:135`)
- 核心: LabelEmbeddingModule → token_fusion → Self-Attention → FFN → Gated Fusion
- LabelEmbedding 包含 5 个分支：bio / pair_valid / entity / attribute（4 个 `nn.Embedding` 软嵌入）+ va（`Linear(2→64)` 投影）
- **生效路径**: `Concat([H, bio_emb])` → `Linear(832→768)` → MHA(8 heads) + LN+Residual → FFN(768→3072→768) + LN+Residual → Gated `g·H' + (1-g)·H`
- **未生效分支** ⚠: `pair_fusion` 在 `forward()` 中计算并返回 `fused_pair`，但上层 `DimASQPModel` 未使用 —— pair 级别的 entity/attribute/pair_valid/va soft-emb 实际上没进入 stage-2 推理路径

#### ConsistencyLoss (`feature_rethink.py:257`)
- `max(0, loss_stage2 - loss_stage1 + margin)` — 鼓励 Stage2 比 Stage1 更好

### 2.3 Loss 计算 (`dimabsa_model.py:554`)

```python
total_loss = bio_loss            # CrossEntropy + label_smoothing=0.1
           + entity_loss         # CrossEntropy + label_smoothing=0.1 (Hungarian匹配后)
           + attr_loss           # CrossEntropy + label_smoothing=0.1 (Hungarian匹配后)
           + va_loss             # Huber Loss (Hungarian匹配后)
           + 0.1 * consistency_loss  # Stage2 > Stage1 约束
```

**Hungarian Matching** (`dimabsa_model.py:452`):
- Cost matrix 计算：`2×span_cost + 1×entity_cost + 1×attr_cost + 0.5×va_cost`
- `span_cost` 基于 IoU，`entity/attr_cost` 基于负对数似然
- 使用 `scipy.optimize.linear_sum_assignment` 求最优匹配

---

## 3. 数据流水线

### 3.1 数据格式

**输入 JSONL**:
```json
{"ID": "R001", "Text": "The food was great.", "Quadruplet": [
  {"Aspect": "food", "Category": "FOOD#QUALITY", "Opinion": "great", "VA": "7.50#6.00"}
]}
```

### 3.2 数据处理 (`dataset.py`)

```
JSONL 文件
    │
    ▼
DimASQPDataset._process_item()
    ├── 解析 VA 字符串 → (valence, arousal) float
    ├── 解析 Category → (entity_id, attribute_id)
    │
    ▼
DimASQPDataset.__getitem__()
    ├── Tokenize (DeBERTa tokenizer, max_length=256, return_offsets_mapping=True)
    ├── _find_span_positions() — 字符→token 位置转换，处理重复 span（用 used_positions 集合去重）
    ├── _create_bio_labels() — 生成 BIO 序列标签
    └── 返回: {input_ids, attention_mask, offset_mapping, bio_labels, pair_labels, ...}
```

**重要**: `offset_mapping` 用于推理时将 token 位置反映射回原始文本字符串（`_tokens_to_text` 方法）。

---

## 4. 训练流程

### 4.1 训练技巧（均已实现）

| 技巧 | 实现位置 | 配置项 |
|------|---------|--------|
| 分层学习率 | `train.py:134` | `encoder_lr=1.5e-5`, `head_lr=7.5e-5` |
| Linear Warmup | `train.py:148` | `warmup_ratio=0.1` |
| FGM 对抗训练 | `train.py:88` (FGM类) | `fgm.enabled`, `fgm.epsilon=0.1` |
| EMA | `train.py:41` (EMA类) | `ema.enabled`, `ema.decay=0.999` |
| Label Smoothing | `dimabsa_model.py:570` | 硬编码 0.1 |
| Gradient Clipping | `train.py:238` | `max_grad_norm=1.0` |
| Early Stopping | `train.py:405` | `patience=8` |

### 4.2 EMA 使用方式
- 训练时每步后 `ema.update()` 更新影子参数
- 验证时 `ema.apply_shadow()` 切换到 EMA 参数，验证完 `ema.restore()` 还原

### 4.3 最优配置 (`config_rethink_v3.yaml`)
```yaml
encoder_lr: 1.5e-5
head_lr:    7.5e-5
batch_size: 8
num_epochs: 30
fgm.epsilon: 0.1
ema.decay:   0.999
rethink.num_iterations: 2
```

---

## 5. 当前实验结果

### 5.1 版本演进

| 版本 | cF1 | Exact F1 | VA RMSE | 关键变化 |
|------|-----|----------|---------|---------|
| Baseline | 0.2935 | 0.3252 | 1.3200 | 无 Rethink/FGM |
| +Rethink | 0.4064 | 0.4490 | 1.3125 | 启用 Feature Rethink |
| +Span Fix | 0.3836 | 0.4195 | 1.2208 | 修复重复 span 匹配 |
| **+Hungarian** | **0.4348** | **0.4748** | **1.1406** | 匈牙利最优匹配训练 |

### 5.2 消融实验结论
- Feature Rethink: **+11.3% cF1**
- Hungarian Matching: **+5.1% cF1**
- FGM (epsilon=0.1): **稳定提升，epsilon>0.3 训练不稳定**
- Rethink 迭代次数 2 最优（3 次略有下降）

### 5.3 错误分析（Dev Set）

| 错误类型 | 占比 | 根因 |
|----------|------|------|
| Span 边界错误 | 35% | "fried rice"提取为"rice"，"pretty good"→"good" |
| Category 分类错误 | 25% | FOOD#QUALITY vs FOOD#GENERAL 混淆 |
| 漏检 FN | 20% | NULL aspect（隐式方面）无法识别 |
| 多检 FP | 15% | 非情感词被误判为 opinion |
| VA 极端值偏移 | 5% | "disgusting"→V预测偏中性 |

> 注：上表为早期单模型分析。当前提交已升级为**per-dataset 专用 encoder + 5-seed 双层 ensemble + LLM 知识蒸馏**，权威结果见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) 与 README.md 结果表。

### 5.4 LLM 知识蒸馏 (KD, V23) — 最终证据闭环

**思路**：LLM（云端 DeepSeek 或本地 ollama qwen3.6）对 in-domain 无标注文本打 silver 四元组标签（121/44/14… 类约束 + few-shot + 严格校验：category 合法 / span 逐字子串 / VA∈[1,9]）→ 小模型在 gold+silver 上训练 → **推理时完全不用 LLM**（区别于失败的 LLM-fusion / post-correction）。

**关键脚本**：
- `scripts/llm_silver_label.py`（DeepSeek 云端，`thinking:disabled` 省 ~20× output token）
- `scripts/llm_silver_label_ollama.py`（本地 ollama，`think:false` 抑制推理 channel；None→"NULL" 归一化）
- `scripts/gen_native_ollama.py`（无 in-domain 语料时本地 LLM 原生生成池，中/日/英…）
- `scripts/run_v23_*.sh`（各数据集 KD 全流程）

**最终证据**：五个预注册 KD−baseline official-test 对照中，eng_laptop、
rus、tat、ukr 的 95% CI 下界大于 0；zho_restaurant 点估计为负且 CI 跨 0。
因此只能写“4/5 对照得到统计支持”，不能写“5/8 KD 成功”或 SOTA。
rus 最终采用 4k 生成批次 capped silver；同一 8k silver 的 cap=3000
paired mean 低于 full-8k，说明 4k 优势不能归因于单纯减量。

**最终解释（非普适因果定律）**：当 silver 提供 gold/base 缺失且标注较准确的
互补覆盖时，KD 更可能有效。直接 fidelity 中 DeepSeek 点估计 8/8 更高，
但仅三个域获得统计支持；Tatar matched downstream 无可靠排名反转。pool 来源
无普适排序，同池 gen8k 减量没有改善，现有证据也不支持统一 cap 或甜区。
统一口径见 [docs/CONCLUSIONS.md](docs/CONCLUSIONS.md) 和 §5.29.5–5.29.7；
与 ICONIP V20 的贡献边界见
[docs/TALLIP_VS_ICONIP_V20.md](docs/TALLIP_VS_ICONIP_V20.md)。

---

## 6. 已实现功能清单 [DONE]

### 核心模型
- [x] **DeBERTa-v3-base 编码器**：AutoModel 加载，强制 float32 避免梯度溢出
- [x] **BIO Span Extractor**：双线性层 + GELU，5类标签 (O/B-ASP/I-ASP/B-OPN/I-OPN)
- [x] **Cartesian Product Pair Builder**：笛卡尔积枚举所有 Aspect×Opinion 候选对
- [x] **PairClassifier**：validity + entity(7) + attribute(6) 三分支分类
- [x] **VARegressor**：共享特征层 + 独立 V/A head，sigmoid 缩放至 [1,9]
- [x] **VAEmbedding**：混合离散(18 bin)+连续投影的 VA 嵌入
- [x] **VAConsistencyVerifier**：opinion 表示 → 隐含 VA 预测（定义了但未接入主流程）
- [x] **FeatureLevelRethink**：LabelEmbedding + Self-Attention + FFN + Gated Fusion
- [x] **两阶段独立预测头**：Stage1/Stage2 各有独立 SpanExtractor/PairClassifier/VARegressor
- [x] **ConsistencyLoss**：margin loss 约束 Stage2 优于 Stage1

### 训练技术
- [x] **Hungarian Matching**：基于 IoU+分类+VA cost 的最优 pair 对齐
- [x] **Label Smoothing**：BIO/entity/attribute loss 均使用 smoothing=0.1
- [x] **Huber Loss**：VA 回归使用 delta=1.0 的 Huber loss
- [x] **FGM 对抗训练**：词向量 embedding 层的 Fast Gradient Method
- [x] **EMA**：decay=0.999，评估时使用影子参数
- [x] **分层学习率**：encoder lr 与 head lr 独立设置
- [x] **Linear Warmup + Linear Decay**
- [x] **Gradient Clipping** (max_norm=1.0)
- [x] **Early Stopping**（基于 cF1，patience=8）

### 数据与评估
- [x] **重复 Span 去重匹配**：used_positions 集合跟踪已匹配字符位置
- [x] **offset_mapping 反映射**：token 位置 → 原始文本字符串
- [x] **cF1 评估器**：完整实现 SemEval-2026 官方 cF1 计算
- [x] **Exact F1、VA RMSE/MAE** 评估
- [x] **批量推理脚本**：生成 SemEval 官方格式 JSONL

---

## 7. 未实现的创新方向 [TODO]

以下方向均有明确的研究价值，按优先级排序：

### 高优先级（工程可行，预期收益明确）

#### TODO-1: CRF 层约束 BIO 标注
- **问题**: 当前 BIO 预测独立，可能出现 I-ASP 在 B-OPN 后面等非法序列
- **方案**: 在 `SpanExtractor` 输出后加 Linear-CRF 层
- **预期**: Span 边界错误（占 35%）减少，cF1 +2~3%
- **文件**: `src/models/dimabsa_model.py:33`，需引入 `torchcrf` 库

#### TODO-2: NULL Aspect 专项处理
- **问题**: 隐式方面（如"It's pricey"中无明确 aspect 词）漏检率高，占 FN 的 20%
- **方案**: 在 BIO 之外增加一个句子级分类器，检测是否存在 NULL aspect；
  为 NULL aspect 生成固定的 [CLS] token representation
- **文件**: `src/models/dimabsa_model.py`，新增 `NullAspectDetector` 模块
- **预期**: cF1 +2~4%

#### TODO-3: Category-Conditioned VA Prediction（论文级）
- **问题**: 所有 Category 共用同一 VARegressor，但不同 Category 的 VA 分布规律差异显著
  （如 RESTAURANT#PRICES 的 V 通常低，FOOD#QUALITY 的 V 通常高）
- **方案**:
  1. 为每个 Category (7×6=42类) 学习 VA prototype embedding
  2. VA 预测 = prototype + offset 回归（而非从零预测绝对值）
  3. `VARegressor` 接收 `(pair_repr, category_id)` 输入
- **文件**: `src/models/va_module.py:12`，修改 `VARegressor.forward()` 签名
- **预期**: VA RMSE -0.1~0.2，cF1 +3~5%
- **参考**: Prototype Network 思路

#### TODO-4: 测试时集成（模型融合）
- **方案**: 训练多个不同随机种子的模型，推理时对 VA 取均值，对 span/category 投票
- **文件**: `scripts/inference.py`，新增 ensemble 逻辑
- **预期**: cF1 +1~2%，零额外训练成本

### 中优先级（方法创新，需要设计）

#### TODO-5: VA 对比学习正则化
- **问题**: VA 预测偏向中性值（极端情感预测不准）
- **方案**: 构建 triplet (anchor, positive, negative)，其中 positive 为 VA 相近样本，
  negative 为 VA 极性相反样本；在 VA 空间添加 Triplet Loss 或 InfoNCE
- **文件**: 新增 `src/models/va_contrastive.py`，在 `_compute_loss` 中加入
- **参考**: `docs/EXPERIMENTS.md:315` 已提及此方向

#### TODO-6: V-A 联合分布预测（论文核心贡献）
- **问题**: 当前独立预测 V 和 A，忽略了两者的相关性（如负面情感往往 V 低且 A 高）
- **方案**: 输出二元高斯分布参数 (μ_V, μ_A, σ_V, σ_A, ρ)，
  训练目标改为负对数似然（NLL）替代 Huber Loss
- **文件**: `src/models/va_module.py:12`，新增 `BivarGaussianVARegressor` 类
- **价值**: 领域内尚无此方向论文，理论和实验均有创新性

#### TODO-7: 激活 VAConsistencyVerifier
- **现状**: `VAConsistencyVerifier` 类已定义（`va_module.py:110`）但未接入主流程
- **方案**: 在 `DimASQPModel.forward()` 中使用它计算 opinion 隐含的 VA，
  作为对 VA 预测的软约束（consistency score 低时加大惩罚）
- **文件**: `src/models/dimabsa_model.py:376`，在 VA 回归后调用

#### TODO-8: Rethink 多轮迭代（当前固定为2轮）
- **现状**: `num_rethink_iterations` 配置项存在但代码只跑1次 Rethink
- **方案**: 在 `DimASQPModel.forward()` 中添加循环，将 Stage2 输出作为下一轮 Stage1 输入
- **文件**: `src/models/dimabsa_model.py:391`

### 低优先级（竞赛优势方向）

#### TODO-9: 跨语言迁移（SemEval-2026 多语言赛道）
- SemEval-2026 提供 11 种语言数据（含日语、俄语、豪萨语等低资源语言）
- 当前仅处理英语餐厅数据
- **方案**: 将 encoder 换为 `microsoft/mdeberta-v3-base`，零样本迁移至其他语言
- **文件**: `configs/` 新增多语言配置，`data/` 适配多语言数据路径

#### TODO-10: 数据增强
- **方案**: EDA (Easy Data Augmentation)、回译、LLM 生成增强样本
- 当前训练集仅 ~2284 样本，数据增强收益较高

---

## 8. 代码中的已知问题与注意事项

| 位置 | 问题描述 | 状态 |
|------|---------|------|
| `dimabsa_model.py` | `num_rethink_iterations` 未实现循环，实际只执行1次 | ✅ 已修复：forward 加 for 循环 |
| `dimabsa_model.py` | `pair_loss` 始终为0，pair validity head 无训练信号 | ✅ 已修复：Hungarian 匹配后计算二分类 loss + 无 gold 样本负信号 |
| `dimabsa_model.py` | `self.span_extractor.extract_spans` 在 Stage2 中调用 Stage1 方法 | ✅ 已修复：改为 `span_extractor_stage2` |
| `dimabsa_model.py` | `loss_weights` 配置未生效，consistency 硬编码 0.1 | ✅ 已修复：通过构造函数传入并应用 |
| `train.py` | FGM 在 warmup 阶段运行，导致早期训练不稳定 | ✅ 已修复：加 `global_step >= num_warmup_steps` 保护 |
| `train.py` | `save_top_k` 配置未实现 | ✅ 已修复：min-heap 保存最优 k 个 checkpoint |
| `dataset.py` | span 匹配顺序依赖 bug（opinion 不感知 aspect 位置，贪心分配） | ⚠️ 待修复：需要基于 aspect-opinion 距离的匹配 |
| `va_module.py:110` | `VAConsistencyVerifier` 类定义但从未被调用 | ⚠️ 待接入主流程 |
| `dimabsa_model.py` | Consistency Loss 只覆盖 BIO，不覆盖 entity/attribute/VA | ⚠️ 设计遗漏 |
| `dimabsa_model.py` | `num_rethink_iterations=0` 时 NameError 崩溃 | ✅ 已修复：构造函数加 assert >= 1 |
| `inference.py` | `load_model` 未传 `loss_weights`，模型结构不一致 | ✅ 已修复：同步参数 |
| `train.py` | Early stopping 中 cF1=0 时回退到 val_loss，可能在初期保存多余 checkpoint | 低优先级 |

---

## 9. 快速上手

```bash
# 环境
conda activate absa
pip install -r requirements.txt

# 训练最优配置
python scripts/train.py --config configs/config_rethink_v3.yaml

# 推理
python scripts/inference.py \
    --checkpoint checkpoints/rethink_v3/checkpoint_best.pt \
    --config configs/config_rethink_v3.yaml \
    --test_file data/subtask_3/eng/eng_restaurant_test_task3.jsonl \
    --output predictions.jsonl

# 评估（有标注的验证集）
python -c "
from src.evaluation.metrics import evaluate_predictions
m = evaluate_predictions('predictions.jsonl', 'data/subtask_3/eng/eng_restaurant_dev_task3.jsonl')
print(m)
"
```

---

## 10. 依赖关系图（模块间调用）

```
train.py
  └── Trainer
        ├── DimASQPModel (dimabsa_model.py)
        │     ├── encoder: DeBERTa-v3 (HuggingFace)
        │     ├── SpanExtractor          (dimabsa_model.py)
        │     ├── PairClassifier         (dimabsa_model.py)
        │     ├── VARegressor            (va_module.py)
        │     ├── FeatureLevelRethink    (feature_rethink.py)
        │     │     └── LabelEmbeddingModule
        │     │           └── VAEmbedding (va_module.py) ← 间接
        │     └── ConsistencyLoss        (feature_rethink.py)
        ├── DimASQPDataset (dataset.py)
        └── DimASQPEvaluator (metrics.py)
```

---

*最后更新: 2026-02-23 | 当前最优 cF1: 0.4348 (Rethink V3 + Hungarian)*
