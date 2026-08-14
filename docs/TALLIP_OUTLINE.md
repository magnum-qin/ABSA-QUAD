# TALLIP 论文大纲：可信的多语言结构化 LLM 蒸馏

状态：写作大纲 v1，2026-08-14。本文档不属于论文 LaTeX，后续英文稿应以冻结
实验产物自动导出的数值为准。

事实与措辞的优先级如下：

1. [`CONCLUSIONS.md`](CONCLUSIONS.md)：最终结论口径；
2. [`tallip_core_tables.md`](audits/tallip_kd/tallip_core_tables.md)：冻结数值；
3. [`TALLIP_VS_ICONIP_V20.md`](TALLIP_VS_ICONIP_V20.md)：与先前工作的边界；
4. [`EXPERIMENTS.md`](EXPERIMENTS.md)：实验过程和历史背景。

引用核心结果前必须运行：

```bash
python scripts/kd/build_tallip_report.py --verify
```

## 1. 论文定位

### 推荐标题

**When Does LLM-Labeled Silver Data Help? An Inductive-Safe Study of Offline
Structured Distillation for Multilingual Dimensional Aspect Sentiment Quad
Prediction**

备选标题：

- **Beyond More Training Data: Auditing LLM-to-Student Distillation for
  Multilingual Dimensional Aspect Sentiment Quad Prediction**
- **Teacher Fidelity Is Not Enough: A Controlled Study of LLM-Labeled Silver
  Data for Multilingual Structured Sentiment Analysis**

推荐标题使用 “structured distillation” 而不是暗示可访问 teacher logits 的经典
KD。正文第一次出现时应明确：本文蒸馏的是 LLM 生成的结构化四元组标签，teacher
只在离线标注阶段出现，学生推理时不调用 LLM。

### 一句话中心论点

在固定的多语言 DimASQP 学生系统上，LLM silver 在五个预注册域的平均效果可靠
优于原始基线、等训练曝光和相同文本的学生伪标签，但单域效果明显异质；teacher
fidelity、pool 来源或 silver 数量均不能单独给出普适排序，因此可信结论必须依赖
inductive-safe 清洗、matched controls 和冻结后的 paired inference。

### 不属于本文的论点

- 本文不重新提出 LEADER-DimASQP、Rethink、Hungarian matching、语言专用
  encoder 或 five-seed ensemble；这些属于 ICONIP V20。
- 本文不宣称八个数据集上的 test SOTA；八个冻结主系统均未超过相应 leaderboard
  best。
- 本文不证明“信息互补性”是普适因果定律，也不声称存在普适 teacher、pool 或
  silver cap。

## 2. 研究问题与回答

| RQ | 研究问题 | 主要证据 | 论文可给出的回答 |
|---|---|---|---|
| RQ1 | 直接 teacher fidelity 是否随数据集变化，能否由下游学生分数替代？ | 8×2 label-hidden dev；Tatar 同池五种子 | DeepSeek 点估计 8/8 更高，但仅 3/8 得到统计支持；Tatar 下游差异不确定，学生分数不能替代直接 fidelity |
| RQ2 | 清除 dev/test 文本重合后，LLM silver 是否仍改善固定学生？ | clean-inductive 五种子、ensemble、paired bootstrap | 五域等权 macro KD−baseline 为 +0.0600，95% CI [+0.0291,+0.0911]；单域仅 eng/rus 的 dev CI 排除 0 |
| RQ3 | 收益是否只是更多训练行或学生自身伪标签造成？ | repeated-gold 与 same-text self-training，各五域×五种子 | macro 对两个控制均为正；family-wise 校正后，LLM 标签的额外价值在 eng/rus 最可靠，其他域不确定 |
| RQ4 | teacher、pool、quantity 和互补性是否存在普适规律？ | teacher/pool/quantity 消融；覆盖、JSD、error transitions、slices | 不存在可支持的普适排序或数量最优点；质量、覆盖和适配共同作用是描述性解释 |
| RQ5 | dev-locked 发现能否在 official test 上维持？ | 一次性赛后本地 test；冻结预测；paired bootstrap | 五个预注册 KD 对照中四个 raw CI 支持，Holm 后 eng/tat/ukr 保持；zho 不确定，不能作 test SOTA 声明 |

## 3. 整体叙事

论文应按以下逻辑推进：

1. LLM 标注无标签文本看似适合低资源结构化情感任务，但“学生变好”混合了
   teacher 质量、文本来源、训练曝光、学生自训练、泄漏和选择偏差。
2. 固定 V20 学生后，将研究对象限定为离线结构化 silver supervision。
3. 先直接测 teacher，再做 clean-inductive 下游比较，然后逐层加入 repeated-gold、
   same-text self-training、pool/quantity 和 error-level 证据。
4. 使用固定 seeds、阈值、paired bootstrap、多重比较和 official-test protocol lock
   约束结论强度。
5. 最终发现不是“某 teacher 或某数量最好”，而是：平均收益存在，单域可靠性和
   teacher 的额外价值具有明显条件性。

## 4. 摘要结构

建议采用五句式或五段式摘要，最后再写：

1. **背景与缺口：** LLM silver 常用于低资源 NLP，但结构化、多语言任务中的收益
   容易与数据泄漏、额外曝光和 teacher–student 适配混淆。
2. **研究设计：** 固定已发表的 DimASQP 学生，在 8 个数据集上直接比较两类
   teacher，并在 5 个采用 KD 的数据集上执行 clean-inductive 五种子实验和两类
   matched controls。
3. **统计方法：** 使用按样本 paired bootstrap、分族 Holm/BH、多数据集等权
   macro bootstrap，以及读取 test gold 前冻结的 official-test 协议。
4. **核心结果：** 报告三项 macro 效果；指出 same-text 条件下只有 eng/rus 的
   teacher 标签优势在多重校正后保持；说明 official test 的 4/5 raw、3/5 Holm
   结果。
5. **结论：** LLM silver 的平均价值可成立，但直接 fidelity、pool 来源和 silver
   数量都不是跨域充分判据，可信评估必须同时报告控制实验与不确定性。

摘要中不放历史 transductive 分数，不使用 “SOTA”、 “universally” 或 “optimal
silver size”。

## 5. 正文章节大纲

### 1 Introduction

建议用六个段落完成：

1. 介绍 DimASQP 同时预测 aspect、opinion、category 与连续 valence–arousal，说明
   多语言和低资源条件下结构标签昂贵。
2. 引出离线 LLM 标注的吸引力：可扩大监督、推理时无 LLM 成本，但标签质量和
   覆盖随语言及领域变化。
3. 指出现有下游对比的四个主要混杂：teacher fidelity 未直接测量、训练曝光未
   匹配、silver 文本与 split 可能重合、test 或 dev 可能被反复用于选择。
4. 概述本文的受控证据链：8×2 fidelity、clean-inductive 五种子、两个非 LLM
   控制、teacher/pool/quantity 消融和直接 error analysis。
5. 用最少数字预告结论：三项五域 macro CI 为正，但可靠的单域 teacher-source
   优势集中在 eng/rus；official test 支持条件性而非普适收益。
6. 给出四项贡献，不把 V20 学生组件列入贡献。

建议贡献文本：

1. We formulate an offline structured LLM-to-student distillation protocol for
   multilingual DimASQP while keeping the previously introduced student and
   inference procedure fixed.
2. We directly evaluate two teachers on eight label-hidden development sets and
   disentangle teacher, pool, quantity, exposure, and same-text pseudo-label
   effects through matched controls.
3. We provide an inductive-safe and multiplicity-aware evaluation with five
   seeds, paired bootstrap inference, deletion-level provenance, and frozen
   decision rules.
4. We show reliable average gains but substantial domain heterogeneity, and
   identify the limits of teacher fidelity, pool source, and silver quantity as
   universal selection rules.

### 2 Related Work

#### 2.1 Aspect Sentiment Quad Prediction and Dimensional ABSA

- 定义离散 polarity ASQP 与连续 VA DimASQP 的区别。
- 只概述与结构化抽取、集合预测和连续情感有关的工作。
- 在本小节末引用 ICONIP V20，明确学生架构来自先前工作。

#### 2.2 Knowledge Distillation, Pseudo-Labeling, and Data Augmentation

- 区分 logits-based KD、response/label distillation、self-training 和普通文本增强。
- 说明本文需要 repeated-gold 与 same-text self-training 两个控制的原因。

#### 2.3 LLM Annotation for Multilingual and Low-Resource NLP

- 讨论 LLM 作为离线 annotator 的成本与可部署性优势。
- 聚焦跨语言/domain fidelity、结构化输出有效率、类别与 span 覆盖。
- 不将低资源自动等同于更高 KD 增益。

#### 2.4 Reliable Evaluation of Silver Supervision

- 讨论 contamination、inductive evaluation、paired resampling、多重比较和 test
  selection bias。
- 用这一小节自然引出本文的 audit-and-freeze protocol。

### 3 Task and Offline Structured Distillation

#### 3.1 DimASQP Task

- 定义输出集合 `(aspect, category, opinion, valence, arousal)`。
- 给出 NULL span、类别预测和 VA 范围。
- 定义官方 Task-3 cF1；辅助 fidelity 指标放实验设置或附录。

#### 3.2 Inherited Student

- 用一段话和一张紧凑示意图说明 V20 学生输入输出。
- Rethink、Hungarian matching、FGM、EMA、encoder 和 five-seed vote 只作为固定
  配置说明，并引用先前论文。
- 不重复 V20 的架构消融或配置演进表。

#### 3.3 LLM-to-Student Silver Pipeline

- 无标签文本 → teacher structured response → schema validation → silver rows →
  gold+silver student training → student-only inference。
- 写清 teacher 不提供 logits，本文属于 structured label/response distillation。
- 说明训练数据的顺序、gold:silver 配方和每数据集冻结阈值保持不变。

#### 3.4 Variables and Alternative Explanations

- teacher：DeepSeek V4 Pro 与 Qwen3.6-35B；Gemma 仅作历史背景。
- pool：原生生成、真实公开语料、翻译构建。
- quantity：独立批次与同池 cap 必须分开解释。
- alternative explanations：更多 update、学生伪标签、泄漏和随机 seed。

### 4 Experimental Design

#### 4.1 Datasets and Evaluation Scope

- 列出 8 个数据集及语言、domain、gold train/dev/test 规模。
- 区分 8 个 teacher-fidelity 数据集与 5 个 adopted-KD 数据集。
- 解释其余 3 个数据集为何保留冻结非 KD 主系统，不把“采用 KD”说成全八域。

#### 4.2 Contamination Audit and Inductive Cleaning

- 说明 Unicode NFKC、casefold 与空白折叠规范化，以及
  exact/normalized/source-ID 三类重合。
- 报告受影响训练集清洗：eng 23,590→22,595；ukr 9,105→8,500；
  zho-real 10,911→9,932。
- rus-gen4k 与 tat-qwen 为零文本重合，重新审计但不重训。
- 指向删除明细、输入 SHA256 和零交集自动检查。

#### 4.3 Direct Teacher-Fidelity Protocol

- 从 gold dev 删除 `Quadruplet`，few-shot 和类别集合只来自 train。
- 8×2 teacher；固定 prompt/version/temperature；保存原始响应和解析状态。
- 主指标 cF1；辅助指标包括 span/category F1、VA error、NULL/rare/invalid/error。
- paired bootstrap seed=20260713，10,000 次。

#### 4.4 Student Training and Frozen Inference

- seeds：42/123/456/789/1024。
- 唯一实验变化是 supervision 数据；学生配置、epoch、优化器和 best-checkpoint
  规则继承并固定。
- 列出各数据集冻结 vote/top-k 阈值；正文可压缩，完整配置放附录。
- 明确 inference 使用 `checkpoint_best.pt`。

#### 4.5 Matched Controls

- **Repeated gold：** 匹配目标训练行数和 student recipe，检验训练曝光解释。
- **Same-text self-training：** 使用与 LLM silver 完全相同的文本和行数，仅将标签
  换成 student ensemble 伪标签，检验 LLM teacher 的额外标签价值。
- **Tatar teacher match：** 相同 gold、silver 文本、行顺序和五 seeds，仅换 teacher。

#### 4.6 Statistical Inference and Multiplicity

- 单数据集：按样本 paired bootstrap，报告 ΔcF1、95% CI 和 two-sided p。
- 多数据集：固定五域、每域等权 macro bootstrap，不按样本量加权。
- 多重比较：在每个预定义 family 内分别做 Holm FWER 与 BH FDR。
- CI 跨 0 写 “inconclusive”，不能写 “no effect” 或 “equivalent”。

#### 4.7 Official-Test Lock and Reproducibility

- 系统、阈值、checkpoint、输入/预测哈希在读取 test gold 前冻结。
- official test 仅作一次性赛后本地 evaluation，不反向选择任何实验。
- 说明 registry、原始响应、audit manifest、bootstrap seed 和核心表锁。

### 5 Results

#### 5.1 RQ1: Direct Teacher Fidelity Is Domain-Dependent

- 主表使用冻结核心表 1。
- 先给总体模式：DeepSeek point estimate 8/8 更高。
- 再收紧统计解释：只有 eng_restaurant、zho_laptop、zho_restaurant 支持差异。
- 用 Tatar matched-pool 对照说明 DeepSeek−Qwen ensemble +0.0038、CI 跨 0；先前
  Qwen 学生优势可能是扰动或配方混杂，而非可靠排名反转。

#### 5.2 RQ2: Clean-Inductive KD Has a Reliable Average but Heterogeneous Effect

- 报告冻结核心表 2 的五 seed mean±SD 与 ensemble。
- 核心表 3 展示 clean KD−baseline 和 clean−transductive。
- 主要总结使用五域等权 macro +0.0600 [+0.0291,+0.0911]。
- 单域层面只将 eng/rus dev 写为 supported；tat/ukr/zho 为方向性或不确定。
- clean−transductive 均不支持可靠差异，不能据此声称泄漏没有风险；清洗是协议
  正确性要求，而非以分数变化决定是否必要。

#### 5.3 RQ3: Gains Are Not Explained by Exposure Alone, but Teacher Value Varies

- 主结果来自冻结核心表 5 和 5d。
- KD−repeated-gold macro +0.0588 [+0.0318,+0.0853]。
- KD−same-text self-training macro +0.0621 [+0.0351,+0.0888]。
- self-training−repeated-gold macro −0.0033 [−0.0226,+0.0170]，不确定。
- 分数据集与 Holm 结果必须并列：eng/rus 对两个关键替代解释最稳健；ukr 的
  exposure 单项支持但 family-wise 不保持；tat/zho 不建立普适 teacher 优势。

#### 5.4 RQ4: No Universal Teacher, Pool, or Quantity Rule

- teacher 结论回指 §5.1，不重复声称排名反转。
- rus 同 teacher、cap=3000：generated > translated > real，只允许域内结论。
- zho real−generated +2.3pp 是观察性结果，不能与 rus 拼成因果排序。
- rus 同一 gen8k 池 cap3000 比 full 两 seed mean 低 1.14pp，预注册 gate 失败；
  这排除了“gen4k 只是因为 silver 更少”，但不是完整 dose-response。
- translation/native 对照存在数量与构建方式混杂，只放补充材料或描述性讨论。

#### 5.5 Direct Complementarity and Error Evidence

- 报告 validity、span/category coverage、JSD、repaired/harmed 和 error slices。
- 对照 eng 与 zho：eng 覆盖增量和 KD gain 最大；zho fidelity 最高但 gain 最小，
  且 span coverage 下降。
- 三个 exploratory Spearman 的 CI 都很宽；它们用于拒绝单指标预测，不用于建立
  moderator。
- 句长、单/多 quadruplet、rare 与 implicit slices 强调跨域异质性。

#### 5.6 RQ5: One-Shot Official-Test Evaluation

- 主表使用冻结核心表 6。
- 先报告全部 8 个冻结主系统，明确只有 5 个是预注册 KD 对照。
- raw paired CI：eng/rus/tat/ukr supported，zho inconclusive negative point。
- family-wise Holm：eng/tat/ukr 保持，rus 不保持。
- 写明无系统超过 leaderboard best；本节标题和正文均不使用 “submission” 或
  “SOTA”。

### 6 Discussion

#### 6.1 What Makes Silver Supervision Useful?

- 平均增益和单域异质性可以同时成立。
- fidelity 是必要的质量信号之一，但不足以预测下游收益。
- coverage、gold 的缺口、类别分布和 teacher–student 适配共同构成更合理解释。
- 当前证据与信息互补性一致，但没有独立操纵互补性，因而不作因果宣称。

#### 6.2 Implications for Multilingual and Low-Resource NLP

给出审慎的实践流程：

1. 先用 label-hidden gold 测 teacher fidelity；
2. 审计 train/silver 与 dev/test 的规范化文本重合；
3. 用多 seeds 和冻结 dev 规则训练；
4. 至少加入 exposure-matched control；条件允许时加入 same-text self-training；
5. 只在域内选择 pool/teacher，不移植普适排序；
6. test 只用于一次性最终复核。

#### 6.3 Efficiency and Deployment

- LLM 仅离线标注，student inference 的 LLM 调用数为 0。
- 报告 post-lock 训练 wall time、best checkpoint 存储和 model-forward throughput。
- historical token usage 与账单未保存，明确为 unavailable，不估算货币成本。
- 不把不同 encoder 或 NVMe/workspace 日志作受控速度比较。

#### 6.4 Relation to ICONIP V20

- 用一段显式说明 V20 提出学生和配置选择；本文固定它来研究 silver supervision。
- 新证据包括 fidelity、clean-inductive、matched controls、统计校正和 official-test
  lock。
- 正文引用先前论文；cover letter 使用
  [`TALLIP_VS_ICONIP_V20.md`](TALLIP_VS_ICONIP_V20.md) 中的披露文本。

### 7 Limitations and Ethical Considerations

- 只研究一个结构化情感任务，跨任务外推尚未验证。
- adopted-KD moderator 分析只有 5 个域，探索性相关 CI 很宽。
- 只有两个主要 teacher；模型版本、服务端更新和 prompt 可能影响复现。
- pool 对照并非全部完全受控；zho 与 ukr 的部分比较只具描述性。
- official-test multiplicity 是 post-hoc robustness analysis，不能改变冻结选择。
- 未保留历史 token/billing，无法给出完整经济成本或碳成本。
- silver 标签可能继承 LLM 的语言、地域和类别偏差；下游部署仍需按域审计。
- official test 是赛后本地评测，不是新的竞赛提交。

### 8 Conclusion

用三层结论收束：

1. 五个固定域的平均 KD 收益得到统计支持，并超过等曝光与同文本学生伪标签。
2. 单域可靠性和 LLM teacher 的额外价值具有显著异质性，最稳健证据集中在
   eng/rus，official test 也不支持把所有 adopted 域写成成功。
3. 因此，未来工作应优先改进可审计的域内选择与监督互补性测量，而不是继续寻找
   普适 teacher、pool 或 silver 数量甜区。

## 6. 表格与图形布局

冻结核心表仍以 [`tallip_core_tables.md`](audits/tallip_kd/tallip_core_tables.md)
中的编号和数值为准。论文排版时可以合并或移至附录，但不得手工维护第二套数值。

### 主文表格

| 论文位置 | 内容 | 冻结来源 |
|---|---|---|
| Table 1 | 五个 adopted KD 的 teacher、pool、base/silver 行数和冻结阈值 | registry、核心表 2/5c |
| Table 2 | 8 个数据集、语言/domain、gold split 和实验角色 | `tallip.yaml` 与 split 文件 |
| Table 3 | 8×2 direct teacher fidelity | 核心表 1 |
| Table 4 | clean-inductive 五种子、ensemble 与 KD−baseline | 核心表 2–3 |
| Table 5 | 三类 macro 对照及 raw/Holm 域级摘要 | 核心表 5d |
| Table 6 | teacher/pool/quantity 消融及允许解释 | 核心表 4 |
| Table 7 | complementarity 与 error transitions | 核心表 5c |
| Table 8 | 一次性 official-test 主系统和五个预注册对照 | 核心表 6 |

resource audit、完整 seed 分数、辅助 fidelity 指标、error slices 和所有 raw/Holm/BH
p 值放附录。若篇幅紧张，Table 5 可压缩为三列，完整 evidence-grade 表放附录。

### 建议图形

1. **Figure 1 — Paired effect forest plot（已生成）.** 按数据集分面展示 KD−baseline、
   KD−repeated-gold、KD−same-text self-training 的 ΔcF1 与 95% CI，并单独显示
   五域 macro。它是“平均有效、单域异质”的主视觉证据；由
   `scripts/kd/plot_tallip_figures.py` 从冻结统计自动生成。
2. **Figure 2 — Audit-aware offline distillation protocol（可选）.** 展示
   gold/unlabeled 输入、两种 teacher、schema validation、overlap filter、三个
   训练分支（LLM/repeat/self）、dev lock 和 one-shot test。该图只在版面允许时
   保留，不重复 V20 网络图。
3. **Appendix Figure A1 — Fidelity versus downstream gain（可选）.** 展示 n=5 的探索性
   关系并在图内标出 CI/`descriptive only`；不要画回归趋势线暗示可靠 moderator。

## 7. 附录结构

- **Appendix A:** 完整 prompt、few-shot 选择、schema 和 parser。
- **Appendix B:** contamination audit、删除原因、行数与 SHA256。
- **Appendix C:** 五个 seeds、固定阈值、checkpoint 与 inference manifest。
- **Appendix D:** 完整 teacher fidelity 辅助指标和有效/空/错误率。
- **Appendix E:** 所有 paired bootstrap、Holm/BH 和 macro bootstrap 结果。
- **Appendix F:** pool/quantity/translation 对照与停止规则。
- **Appendix G:** complementarity、error transition 和全部 slices。
- **Appendix H:** 资源、存储、吞吐和缺失 token/cost 字段。
- **Appendix I:** 可复现命令、registry 与 artifact hash。

## 8. Claim–Evidence 写作锁

| 拟写结论 | 最强允许措辞 | 不允许的延伸 |
|---|---|---|
| 五域平均 KD 效果 | reliable positive macro-average effect on the fixed five-domain set | KD works for every language/domain |
| teacher fidelity | DeepSeek is more consistent overall; three domain differences are supported | DeepSeek universally dominates Qwen |
| Tatar teacher | no reliable ranking difference under matched text and seeds | Qwen is better because an earlier student score was higher |
| exposure | gains exceed matched exposure on eng/rus; ukr has single-comparison support | more rows never explain KD gains |
| same-text teacher value | LLM labels add reliable value on eng/rus | LLM labels always beat self-training |
| pool | no universal ordering; rus has a strong within-domain ordering | generated/real pools are universally best |
| quantity | lower cap does not explain rus gen4k advantage | 3k or 4k is an optimal silver size |
| complementarity | patterns are consistent with a multi-factor explanation | complementarity is a proven causal moderator |
| official test | four raw paired contrasts supported; three remain after Holm | 4/8 SOTA or five adopted domains all succeed |

## 9. 推荐写作顺序

1. 先写 §3–4（任务、KD protocol、实验设计），固定术语和控制变量。
2. 从冻结表生成 §5，不从 `EXPERIMENTS.md` 的历史过程手工抄数。
3. 写 §6–7，把结果限制、prior-work 边界和部署含义一次讲清。
4. 再写 Introduction 与 Related Work，使引言承诺与实际证据完全对应。
5. 最后写 Abstract、Conclusion、标题和 cover letter。
6. 插图完成后做一次全文 claim audit，并再次运行核心表 `--verify`。

当前不需要继续训练新的 teacher、pool、cap 或 augmentation 网格。只有在发现数据、
评估或复现错误，或审稿人提出明确问题时，才重新开启实验。
