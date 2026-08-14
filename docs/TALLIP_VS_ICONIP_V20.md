# TALLIP 与 ICONIP V20 的贡献边界

更新日期：2026-08-14。

本文档用于约束 TALLIP 稿件的 novelty、相关工作和 prior-publication
声明。ICONIP V20 的依据是仓库中的冻结稿
[`paper_iconip_current.tex`](paper_iconip_current.tex)；TALLIP 数值证据以
[`CONCLUSIONS.md`](CONCLUSIONS.md) 和
[`tallip_core_tables.md`](audits/tallip_kd/tallip_core_tables.md) 为准。

## 一句话区别

**ICONIP V20 研究如何构建并按数据集选择稳定的多语言 DimASQP 学生系统；
TALLIP 固定并复用该学生系统，研究离线 LLM silver 在什么条件下有效，以及
这些结论能否通过 clean-inductive 审计、paired 统计检验和一次性 official
test 复核。**

因此，TALLIP 的新意不是再次提出 LEADER-DimASQP，也不是把 V20 的配置表
扩大，而是把研究对象从“学生架构与配置选择”转向“LLM 监督的有效条件与
证据可信度”。

## 逐项差异

| 维度 | ICONIP V20 | TALLIP |
|---|---|---|
| 核心问题 | 怎样得到稳定、可按语言/数据集选择的 DimASQP 系统 | 固定学生后，LLM 标注的 silver 何时、为何能改善学生 |
| 方法中心 | LEADER-DimASQP：Rethink、Hungarian matching、VA regression 与多种子推理 | 离线结构化 KD：teacher 对无标注文本生成四元组 silver，推理仍只用学生 |
| 主要变量 | encoder、transfer、paraphrase、COMET、top-k、ensemble 与训练技巧 | teacher、pool 来源、silver 数量/批次、域互补性与泄漏清理 |
| LLM 的角色 | 增强或候选融合属于配置研究，LLM candidate fusion 是已报告负结果之一 | DeepSeek/Qwen 是离线 silver teacher，并直接接受 label-hidden gold-dev fidelity 评测 |
| 学生系统 | 论文的主要技术贡献 | 从 V20 继承并作为受控实验载体，不作为 TALLIP 新贡献 |
| 数据范围 | 同一项 SemEval 任务的 8 个数据集 | 仍为同一 8 个数据集；数据集数量本身不是新增贡献 |
| 主要证据 | dev 上的系统/配置比较与 five-seed ensemble | clean-inductive five-seed、8×2 teacher fidelity、五域曝光匹配/同文本自训练、直接互补性/error analysis、固定任务 macro bootstrap、paired bootstrap 与多重校正、冻结 official test |
| 数据完整性 | 按当时比赛开发流程报告 | 统一文本泄漏审计，清理 3 个 KD 输入并补训 clean baseline/KD |
| 主要结论 | 无普适学生配置；语言专用 encoder 与多种子推理最稳健 | KD 收益有条件；teacher/pool 效果依域而变；不存在普适 silver cap |
| 交付物 | V20 系统与配置研究 | 原始 teacher 响应、删除明细、SHA256、协议锁和六张冻结证据表 |

## ICONIP V20 已经拥有的贡献

以下内容在 TALLIP 中只能作为 prior work 或继承设置，不能再次声称为新贡献：

1. LEADER-DimASQP 学生架构，包括 Rethink、Hungarian matching 和连续 VA
   regression。
2. validation-driven 的 language-specific encoder 与 per-dataset recipe 选择。
3. transfer、paraphrase、COMET、encoder scale、native encoder 和 top-k 的系统
   配置研究。
4. five-seed ensemble，以及 CRF、过量 COMET、cross-encoder fusion、
   confidence voting、uncertainty loss 和 LLM candidate fusion 等负结果。
5. V20 的 dev 结果及其相对当时 dev baseline 的改善。

如果 TALLIP 需要解释这些组件，应明确写成 “we build on our previously
introduced student” 并引用 V20；V20 的架构消融不应占据 TALLIP 主结果篇幅。

## TALLIP 真正新增的贡献

1. **离线结构化 LLM-to-student KD 问题设定。** 在固定 V20 学生和推理流程
   的前提下，系统研究 LLM 标注的 in-domain silver，而非提出新的学生架构。
2. **teacher 质量的直接与下游双重测量。** 完成 8 数据集 × 2 teachers 的
   label-hidden gold-dev fidelity，并用 Tatar matched-pool five-seed 对照检查
   teacher 排名是否只是学生训练扰动。
3. **teacher、pool 与 quantity 的边界研究。** 纳入跨 teacher、rus 的
   size-matched pool、zho 的观察性 pool 对照，以及同一 gen8k 池的 cap3000
   停止实验，区分来源、数量与生成批次质量。
4. **inductive-safe 与统计闭环。** 审计 gold/silver/distill/dev/test，清理并
   重训受影响的 eng/ukr/zho；固定五个 seeds、ensemble 阈值与 10,000 次
   paired bootstrap。
5. **非 LLM 控制与机制边界。** 在 official-test 结果冻结后，仅使用 dev，
   在五个 adopted 数据集补充等训练曝光的 repeated-gold 与相同 silver 文本的
   student self-training 五种子控制，以区分 LLM 标注收益、重复曝光和学生
   伪标签；并直接测量 silver 有效性、覆盖、分布与 repaired/harmed errors。
6. **一次性 official-test 复核与可复现冻结。** test gold 只用于冻结后的
   post-competition local evaluation，不用于模型、阈值、teacher、pool 或
   增强选择；保存预测与输入哈希并冻结核心表。主要对照按预先定义的 family
   报告 Holm/BH 校正；资源表区分已验证覆盖、可计时调用和不可恢复的历史
   token/cost 字段。

这些是**经验与评测方法贡献**，不是新的网络模块。最稳妥的论文主张是：
在多语言结构化情感任务中，LLM silver 的收益取决于标注质量、覆盖和与 gold
的互补性；当前数据不支持普适 teacher、pool 排序或 silver 数量最优点。

## 复用项：必须披露，但不计入新意

- 同一 DimASQP 任务、8 个数据集和官方 cF1；
- LEADER-DimASQP、Rethink、Hungarian matching 和 VA head；
- language-specific encoders、FGM、EMA、训练轮数等学生配方；
- five-seed ensemble、vote/top-k 推理及 dev-only 选择原则；
- V20 no-KD 系统作为 matched baseline 或非 KD 数据集的冻结主系统。

复用这些内容是控制变量所必需的。TALLIP 应把它们压缩为“inherited student
and training protocol”小节，并把篇幅集中到 KD 设计、审计、统计与发现。

## TALLIP 核心贡献建议文本（英文）

可在不改动事实边界的前提下润色：

1. *Building on our previously introduced LEADER-DimASQP student, we formulate an
   offline structured knowledge-distillation protocol in which LLMs annotate
   in-domain unlabeled text and are not used at inference time.*
2. *We disentangle teacher, pool, and silver-data effects through direct 8-by-2
   teacher-fidelity evaluation, matched downstream controls, pool comparisons,
   a same-pool quantity intervention, and five-domain dev-only exposure-matched
   and same-text student-self-training controls, together with direct
   complementarity and error-transition measurements.*
3. *We provide an inductive-safe evaluation by auditing train/dev/test overlap,
   retraining all affected five-seed systems, and reporting paired bootstrap
   intervals plus family-wise multiplicity diagnostics against matched controls.*
4. *A dev-locked, one-shot evaluation on the released official test set supports
   conditional KD gains in four adopted domains, while showing that neither a
   universal teacher/pool ordering nor a universal silver-size optimum is
   warranted.*

这里的 “four adopted domains” 是针对五个预注册 KD 对照中的四个，不是
“4/8 个数据集达到 SOTA”。八个主系统都没有超过 official leaderboard best。

## Prior-publication / cover-letter 建议文本（英文）

> This manuscript builds on our earlier ICONIP V20 study of LEADER-DimASQP. The
> earlier work introduced the student architecture, language-specific recipe
> selection, and multi-seed inference, which are treated here as inherited
> components rather than new contributions. The present manuscript asks a
> different question: when LLM-labeled silver data improves that fixed student.
> It adds direct two-teacher fidelity evaluation across eight datasets,
> teacher/pool/quantity controls, five-domain exposure-matched and same-text
> student-self-training controls, direct complementarity/error measurements,
> leakage-audited clean-inductive retraining, multiplicity-aware paired
> statistical inference, and a dev-locked post-competition evaluation on
> the released official test set. None of these post-V20 KD analyses or official
> test results appeared in the earlier manuscript.

投稿时仍应按目标期刊的最新政策披露并引用先前论文；本段不替代出版社的
具体投稿要求。

## 贡献—证据映射

| TALLIP 主张 | 直接证据 | 允许的强度 |
|---|---|---|
| teacher fidelity 具有域差异 | 核心表 1；Tatar matched control（表 4） | DeepSeek 整体更稳健，部分域差异不确定；不声称可靠排名反转 |
| clean-inductive KD 的 dev 效果 | 核心表 2–3 | eng/rus supported；tat/ukr/zho dev evidence 不确定 |
| pool 没有普适排序 | 核心表 4 | rus 为受控证据，zho 为观察性证据；不作跨域因果排序 |
| 更少 silver 并非 rus gen4k 优势原因 | 核心表 4 | 同池 cap3000 未改善；不推出普适最佳数量 |
| LLM 标注收益不只是训练曝光 | 核心表 5 | 单项 eng/rus/ukr supported；五域 Holm 后 eng/rus 保持 |
| LLM teacher 相对学生伪标签的增益 | 核心表 5 | 五域同文本对照中仅 eng/rus supported 且 Holm 后保持；tat/ukr/zho 不确定 |
| 信息互补性的直接描述性证据 | 核心表 5c；完整 error-slice artifact | 指标跨域非单调；支持多因素解释，不作因果规律 |
| 多重比较与资源边界 | 核心表 5d–5e | 分族 Holm/BH；历史 token/账单缺失，不虚构货币成本 |
| official test 上的条件性 KD 收益 | 核心表 6 | 五个预注册单项对照中四个 supported；Holm 后 eng/tat/ukr 保持；不声称 SOTA |

核心表编号以冻结生成物为准，不应手工复制后继续修改数值。

## 禁止的重叠或越界表述

- “TALLIP proposes LEADER-DimASQP / Rethink / Hungarian matching”；
- 把 language-specific encoder、five-seed ensemble 或 V20 augmentation sweep
  再列为 TALLIP 新贡献；
- 仅因仍覆盖相同 8 个数据集就声称 TALLIP 的实验范围新增；
- 用 TALLIP official-test 分数减去 ICONIP dev 分数宣称提升；
- 把 post-competition 本地评测写成新的官方提交或 test SOTA；
- 把 “信息互补性”写成已独立操纵并证明的普适因果机制；
- 声称存在普适最优 teacher、pool、silver cap 或“silver 越多越好”；
- 把 zho_restaurant 写成得到统计支持的 KD 成功案例。

## 写作编排建议

TALLIP 主文应将继承的 V20 学生架构压缩为一个带明确引用的小节。主结果围绕
六张冻结表展开：teacher fidelity、clean five-seed、paired dev、受控消融、
post-lock 非 LLM 控制和 official test。V20 的架构/配置消融只用于说明固定学生的来源，必要时放附录；
否则稿件容易被理解为同一方法的重复配置报告，而不是一个独立的 KD 研究。
