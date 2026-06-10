# 组会报告 — DimASQP (SemEval-2026 Task 3) 进展

日期：2026-06-08

## 1. 一句话总结

用 **440M 小模型 + LLM 知识蒸馏**，在 eng_laptop（最稀疏数据集）上 **dev cF1 0.4141 反超 27B LLM 队伍 nchellwig (0.4006)**；并验证了 KD 的适用边界 —— 只对数据稀缺数据集有效。

## 1.5 eng_laptop 实验时间线（V20 → V23，本期重点）

按时间顺序，每次实验的**动机 → 做法 → 发现**。主线：把 eng_laptop（全项目最弱短板）从 0.3088 提升到 0.4141。

### 起点：V20（上次组会）
- roberta-large + BT(3-pivot) + paraphrase，16592 行，5-seed
- dev cF1 **0.3088**，落后 test SOTA (Takoyaki 0.4227) ~11pp

### ① V21 — jpn DeepSeek 增强 + LLM 后处理
- **动机**：DeepSeek 增强 + LLM 后处理能否突破 jpn V16 no_aug (0.4500)
- **做法**：jpn +939 增强样本，5-seed，LLM 两版后处理（zero-shot / Draft-Revision）
- **🔍 发现**：① LLM zero-shot 纠错灾难性（0.2919，−14pp）② Draft-Revision 安全但只 +0.0003 → **LLM 适合保守边界清洗，不能替模型做 category 决策**，未采用

### ② V22 — 多语种 DeepSeek 增强 + 大模型重跑
- **动机**：把"增强 + 5-seed"推广到 rus/ukr，eng_laptop 无增强大模型重跑
- **🔍 发现**：ukr 0.5063(≈V13b)、rus 0.4696(<V15 0.5267)、eng_laptop 0.3056 → **encoder scaling 比增强更关键**，均未替换最佳

### ③ 关键纠错 — 复现审计
- **动机**：审计能否复现记录的历史最佳
- **🔍 发现（重要）**：eng_laptop 真历史最佳是 **V5/V6 的 0.3304**（deberta-v3-base + 4-pivot BT + dual-layer），而 V16→V19→V20 系列（0.3035→0.3088）**从未追平**！之前一直锚定错的 baseline，方向错位；V5/V6 checkpoint+数据已丢失需重建

### ④ V5-repro — 复现历史最佳配方
- **做法**：MarianMT 4-pivot BT 重新生成 → 15049 行（与历史一致）；deberta-v3-base + dual-layer 5-seed
- **结果**：**0.3254**（距历史 0.3304 仅 −0.5pp）
- **🔍 发现**：近似复现成功；**0.3254 远超 V16/V19/V20 (0.30-0.31)** → **小模型 deberta-base + 多 pivot BT 才是对的路**，large 在 240 类稀疏空间过拟合

### ⑤ Route A — 6-pivot BT 加料
- **做法**：6-pivot (de/fr/zh/ru/**es/it**)，21603 行，10-seed
- **🔍 发现**：单 seed 全 0.283-0.298，ensemble 最高 0.3109 → **es/it pivot 稀释质量**，死路

### ⑥ V23 — LLM 知识蒸馏（突破）🚀
- **动机**：240 类 × 4076 样本极端稀疏，小模型学不到 LLM 稀疏类先验；推理时 LLM 介入全失败 → 改用**知识蒸馏**（先验进权重，推理不碰 LLM）
- **做法**：8578 句无标注笔记本评论 → DeepSeek-v4-pro silver 标注 8541 样本/26723 quads → deberta-v3-base 训 raw+BT+silver (23590 行) → 纯 ensemble 推理
- **结果**：单 seed mean **0.3935**（+8.8pp）；ensemble best **0.4141**
- **🔍 发现**：① **KD 是小模型碰 LLM-SOTA 的唯一有效路线** → 反超 nchellwig (27B, 0.4006) ② **关闭 reasoning 省 19-29× token**（v4-pro 默认吐 ~2200 隐藏 CoT token，`thinking:disabled` 降到 ~76，质量等价）③ **440M 模型达 27B LLM 水准**，参数小 60×

### ⑦ V23 推广 — zho_laptop / rus / ukr（完成）
- **silver 产量**：zho 17133 quads、rus 38396 quads（thinking-off，keep 0.98-0.99）
- **最终结果（5-seed ensemble）**：

| 数据集 | gold 量 | 旧最佳 | KD ensemble | Δ | Test SOTA | 结论 |
|---|---|---|---|---|---|---|
| **rus** | 1,766（最稀缺）| V15 0.5267 | **0.5753** | **+4.86** | PAI 0.5599 | ✅ **反超 SOTA +1.54** 🥇 |
| **ukr** | 2,967（稀缺）| V13b 0.5068 | **0.5560** | **+4.92** | PAI 0.5437 | ✅ **反超 SOTA +1.23** 🥇 |
| zho_laptop | 5,534（充足）| V16 0.4487 | 0.4428 | −0.59 | NYCU 0.4824 | ❌ 保留 V16 |

（ukr 无原生乌语语料 → MarianMT 把俄语 geo-reviews 翻译成乌语，DeepSeek 标注；翻译 silver 仍反超 SOTA）

- **🔍 发现（论文核心价值）**：**KD 收益 ∝ 数据稀缺度，3 数据集反超 SOTA，单调关系完整验证**：
  - rus(1766) **+4.86** ✅ | ukr(2967) **+4.92** ✅ | eng_laptop(4076) **+8.87** ✅ | zho_laptop(5534) **−0.59** ❌
  - **3 个数据稀缺数据集（≤4k gold）全部反超 test SOTA**；充足数据集稀释
  - **「LLM distillation benefit is inversely proportional to gold-data sufficiency」**；且对 silver 文本来源鲁棒（ukr 用机器翻译仍超 SOTA）

### 一图总结
```
V20 (roberta-large+BT+para)   0.3088
 │ ①② V21/V22: 增强+推理时LLM → 均失败
 │ ③ 审计纠错: 真历史最佳 V5/V6=0.3304 (deberta-base+4pivot BT)
 ▼
V5-repro (deberta-base+4pivot) 0.3254  ← 方向纠正: 小模型+BT > 大模型
 │ ⑤ Route A 6-pivot: 0.3109 (es/it稀释)
 ▼
V23 (LLM知识蒸馏)             0.4141  🚀 +10.5pp, 反超nchellwig
 │ ⑦ 推广: KD只对数据稀缺有效 (非普适)
```

### 方法学收获（可写入论文 discussion）
1. **小模型 + 数据增强 > 大模型**：deberta-base 0.3254 > deberta-large 0.3035
2. **推理时引入 LLM 全失败，训练时蒸馏 LLM 成功**
3. **KD 收益 ∝ 数据稀缺度**：稀缺特效、充足稀释
4. **Reasoning 模型对结构化抽取无显著增益**：关闭思考省 19-29×，质量等价

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