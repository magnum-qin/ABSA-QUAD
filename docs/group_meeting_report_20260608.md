# 组会报告 — DimASQP (SemEval-2026 Task 3) 进展

日期：2026-06-08

## 1. 一句话总结

用 **440M 小模型 + LLM 知识蒸馏**，在 eng_laptop（最稀疏数据集）上 **dev cF1 0.4141 反超 27B LLM 队伍 nchellwig (0.4006)**；并验证了 KD 的适用边界 —— 只对数据稀缺数据集有效。

## 2. 8 数据集当前最佳（dev cF1）

| 数据集 | 配置 | dev cF1 | Test SOTA (1st) | 相对 SOTA |
|---|---|---|---|---|
| eng_restaurant | V16 deberta-v3-large | 0.6949 | Takoyaki 0.6514 | **+4.35** 🥇 |
| eng_laptop | **V23 KD** | **0.4141** | Takoyaki 0.4227 | −0.86（超 2nd nchellwig 0.4006）|
| jpn_hotel | V16 deberta-v2-jp (no_aug) | 0.4500 | PALI 0.4252 | **+2.48** 🥇 |
| rus_restaurant | V15 xlm-r-large | 0.5267 | PAI 0.5599 | −3.32 |
| tat_restaurant | V15 xlm-r-large | 0.4821 | Takoyaki 0.4736 | **+0.85** 🥇 |
| ukr_restaurant | V13b mdeberta | 0.5068 | PAI 0.5437 | −3.69 |
| zho_laptop | V16 chinese-roberta | 0.4487 | NYCU 0.4824 | −3.37 |
| zho_restaurant | V16 chinese-roberta | 0.5669 | NYCU 0.5521 | **+1.48** 🥇 |
| **平均** | per-lang mix | **0.5113** | — | — |

dev 上 **4/8 数据集反超各自 test 第 1 名**（caveat：dev≠test，最终需 test 验证）。

## 3. 本期核心工作：LLM 知识蒸馏 (KD)

### 3.1 动机
eng_laptop 长期是最弱短板（落后 SOTA ~12pp）。瓶颈 = 240 个 (entity#attribute) 组合 × 仅 4076 样本的极端稀疏，小模型学不到 LLM 那样的稀疏类先验。

### 3.2 方法（关键：先验进权重，推理不碰 LLM）
1. 采集无标注笔记本评论 8,578 句（Amazon + test 文本）
2. DeepSeek-v4-pro 生成 silver 四元组标签（121 类约束 + 校验）→ 8,541 样本 / 26,723 quads
3. deberta-v3-base 在 raw + BT + silver (23,590 行) 上训练，5-seed ensemble
4. **推理时无 LLM**（区别于此前失败的 LLM-fusion/post-correction）

### 3.3 结果
| 方法 | eng_laptop dev cF1 |
|---|---|
| 历史最好努力 (V16/V19/V20) | 0.3035–0.3088 |
| V5 复现 (deberta-base + 4-pivot BT) | 0.3254 |
| **V23 LLM 蒸馏** | **0.4141** (+8.87pp) |

单 seed 就从 ~0.30 跳到 ~0.39（+8.8pp），dev 为干净留出。

## 4. 关键工程发现

### 4.1 关闭 reasoning 省 19-29× token
DeepSeek-v4-pro 默认推理模式每句吐 ~2200 输出 token（大部分是隐藏 CoT），实际 JSON 只需 ~76。加 `extra_body={"thinking":{"type":"disabled"}}` → 输出降 19-29×、消除 402 余额耗尽。
- 同句 head-to-head：thinking ON 仅多 ~13% recall，有效率两者 ~98-100%，精确重合仅 ~37%（说明 LLM 抽取方差大，解释 LLM 队伍为何用 k=10 self-consistency）。
- silver 质量两模式等价 → **OFF 性价比完胜**。

### 4.2 KD 不是普适增强（适用边界）
| 数据集 | gold 量 | KD 效果 |
|---|---|---|
| eng_laptop | 4,076（稀缺）| **+10pp** ✅ |
| zho_laptop | 5,534（充足）+ 强中文 encoder | **−1.3pp** ❌（进行中）|
| rus | 1,766（最稀缺）| 训练中（最可能受益）|

**结论**：KD 是「数据稀缺数据集的特效药」，不是普适方法。gold 充足 + encoder 强时，silver 噪声反而稀释信号。

## 5. 方法学总结（论文骨架）

1. **DimASQP 架构**：Encoder + Span(BIO) + Pair + VA 回归 + Rethink 两阶段
2. **Per-language native encoder**（§5.23）：jpn=deberta-v2-jp, zh=chinese-roberta, eng=deberta-v3, rus/tat=xlm-r —— 比单一多语 encoder +5pp 量级
3. **5-seed ensemble + vote/topk sweep**（§7.6）：+4.76pp 平均
4. **数据增强的边界**：augmentation/Rethink 边际效益随 encoder 增强而衰减（§7.6.2/7.6.3）；KD 仅对稀缺数据集有效（§5.29）
5. **效率论点**：450M 小模型对标 14-27B LLM，4/8 数据集 dev 反超 SOTA

## 6. 失败实验（discussion 价值）
Cross-encoder ensemble、Per-category vote、MeCab v2、LLM fusion、Confidence-weighted ensemble、UML、6-pivot BT 过度增强、paraphrase on strong encoder —— 均已记录（§5.20, §5.25, §5.29）。

## 7. 下一步
- 等 zho_laptop KD 5-seed ensemble 确认（预期 < V16 0.4487）
- rus KD（数据最稀缺，KD 真正试金石）结果
- ukr KD 决策（无外部语料）
- test 集最终提交 + 论文撰写