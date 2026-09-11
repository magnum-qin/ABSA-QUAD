# ICSAI 第一阶段：固定实验矩阵与执行协议

设计日期：2026-09-07。目标会议 9 月 15 日截止，内部提交目标 9 月 14 日北京时间 18:00。**本文件锁定下一阶段设计，不代表已启动或完成实验，也不承诺所有训练能在截止前完成。**

机器可读权威设计：[experiment_matrix.json](audits/icsai_phase1/experiment_matrix.json)。输入：[gold_input_manifest.json](audits/icsai_phase1/gold_input_manifest.json)。全部设计/审计 hash：[phase1.lock.json](audits/icsai_phase1/phase1.lock.json)。后续修改必须写时间、理由、已见结果范围及受影响实验，不能覆盖旧冻结记录后仍称预先固定。

## 清单二：固定实验矩阵

主研究问题：固定 encoder、gold 输入和训练曝光后，BIO 反馈是否优于单阶段和同结构无反馈精炼？统一配置跨八域的结果作为辅助系统证据。

| 编号 | 数据集 | 方法 | seeds | 训练次数 | 状态 |
|---|---|---|---|---:|---|
| A-core | eng_restaurant、zho_restaurant、tat_restaurant | 单阶段，无 Rethink | 42/123/456 | 9 | 已锁定，未运行 |
| B-core | 同上三域 | 当前 BIO-only Rethink，两轮共享 stage-2 heads | 42/123/456 | 9 | 已锁定，未运行 |
| C-core | 同上三域 | 与 B 同网络，但每轮 soft BIO embedding 乘 0 后再进入 token_fusion | 42/123/456 | 9 | 已锁定，控制实现待第二阶段完成 |
| A-extension | eng_laptop、jpn_hotel、rus_restaurant、ukr_restaurant、zho_laptop | 与 A-core 完全相同的统一训练配置 | 42/123/456 | 15 | 已锁定，核心组后执行 |
| H | 八域原稿系统及保留替代物 | 来源分级的历史观察性结果 | 历史五 seeds | 不自动重训 | 来源详见证据清单，不能当受控组 |
| E | 已完成的新三种子模型、可溯源的旧五种子模型 | 单模型/固定集成/成本 | 固定成员 | 0 次新增训练 | 完成对应模型后评估 |
| Q | 实际纳入论文的增强数据及预测 | 标签审计、结构/VA/显隐式误差切片 | 不适用 | 0 次新增训练 | 分析工作，不扩展模型搜索 |

**核心 27 次 + 扩展 15 次 = 42 次训练。** 所有 42 个唯一 run ID、输出目录、seed 和 train 文件已显式写入 JSON。A-core 直接计入八域统一 A 基线，不重复训练。3 seeds 是本轮固定预算，不再根据显著性决定增加哪些 seed。

选择三域的理由在看新消融结果前固定：英文作为资源充分的对照，中文检验不同分词/书写体系，Tatar 检验低资源语言。已有历史结果暴露不能被抹去，所以不声称这是完全不知历史表现的抽样。

**C 的精确定义**：保留 B 全部参数、soft embedding 计算、token_fusion、attention/FFN/gate、两轮循环和损失；仅将每轮 BIO embedding 置零。它比“随意加一层 MLP”更明确。B/C 总分配参数量相同，但 C 的零反馈分支没有有效梯度；同时报告总参数与实际参与梯度的参数，不称有效容量完全相同。C 不能靠现有 YAML 中不存在的字段假装启用，须先实现并验证干预。

## 共同控制条件

- encoder/tokenizer：`microsoft/mdeberta-v3-base`，缓存 revision **a0484667b22365f84929a935b5e50a51f71f159d**。各域分别训练，不是共享权重多语言模型。
- 数据：八域已冻结 clean gold 副本；无增强、silver、跨域任务 checkpoint。保持剩余训练行顺序及批次顺序，记录 shuffle seed。八域有序 entity/attribute label map 已逐项写入 JSON，并核实覆盖全部新 train 类别；只从同域已有配置借用标签本体，不继承其 KD 数据或训练设置。不得由 test 标签推断。
- 初始化：同 seed 的 A/B/C 共有参数逐字相同，B/C 完整参数逐字相同；保存训练前 fingerprint。不能只设置随机种子后假定相同。
- 30 epochs、关闭 early stopping（patience=31）、batch=8、accumulation=1、max length=256、float32；同数据集各组训练更新数相同。固定训练曝光不等于固定 FLOPs，后者单独计时。
- encoder LR=1.5e-5，heads LR=7.5e-5；AdamW weight decay=0.01；linear warmup=0.1；clip=1；dropout=0.1；FGM epsilon=0.1，统一沿用 trainer warmup guard；EMA decay=0.999。
- CRF/UML/category-conditioned VA 关闭。固定损失权重 span/pair/entity/attribute/VA/consistency=2/1/1/1/1/0.5，VA 内部 V/A 权重=1/2。A 无 consistency，B/C 一致；这一区别属于整体两阶段方法，B−C 更直接检验反馈内容。
- 每 epoch 评估 dev；选最高 dev cF1 的 EMA checkpoint，完全同分选最早 epoch；只用 best，不做 top-3 平均。使用的 dev scorer 版本必须记录，最终主表用锁定官方 evaluator；不得混写为同一度量实现已经验证等价。
- 单模型架构指标为主。推理 pair threshold=0.3，span topk=0，dual-layer=false；新三种子集成固定成员 42/123/456，vote=2。旧五种子阈值只作为历史记录，不能套到新受控组。
- 输出独立存放于 `checkpoints/icsai_phase1/`、`predictions/icsai_phase1/`；不覆盖原 checkpoint、submission 或 TALLIP 冻结文件。

## 统计、成本和指标规则

1. 报告每 seed cF1、均值、样本 SD、B−A 和 B−C 配对 seed 差值。六个主要对照（三域×两差值）使用 Holm 校正；未校正 95% CI 与校正后 p 分开呈现。
2. 配对样本 bootstrap 10,000 次，seed=20260907；每次同一数据集所有 seeds/方法用相同重采样 ID，分别重算 cF1 后平均 seed 差值。这个 CI 衡量样本不确定性，不能替代 seed 方差；3 seeds 也不能可靠估计整个调参流程的随机性。
3. 固定三域宏平均只代表这三个域，不推广成八域普遍增益。扩展只有 A 组，不提供其他五域 B 的证据。统一 A 与历史 H 的差值同时包含编码器、增强、架构、训练版本等差异，不作单变量因果归因。
4. 对每次输入/推理检查 ID 对齐、每 seed 每 triple 最多一票。若现有实现产生重复 key，先修复并记录实现变更，不能在看到 test 得分后改投票规则。
5. 成本在相同 GPU、batch、dtype、序列设置下报告端到端吞吐、峰值显存、模型参数与训练更新数/耗时；顺序加载 ensemble 时总参数不等于同时驻留显存。既有 forward-only 时间只能注明范围复用。
6. 补离散三元组 F1、matched V/A MAE（同时报告匹配覆盖率）、显式/隐式跨度数量。边界宽松诊断采用字符跨度 IoU≥0.5、相同 category、一对一匹配，NULL 只匹配 NULL；字符串无法唯一定位的重复提及单列歧义覆盖率，不默默猜跨度。诊断指标不替代官方 cF1。
7. 增强审计：对实际用于论文的 language×augmentation-type 分层，每层固定随机 seed=20260907 抽 50 对，不足则全量；两位具备目标语言能力的标注者独立判断 span/pair/category/VA 保留程度，盲于模型性能，报告一致率和仲裁规则。人力不足则如实缺失，不能用同一生成模型自评替代人工证据。

## test 暴露与锁定时序

本项目已在 7 月完成赛后 official-test 评估，当前 test 结果已知。新结果必须标记为 **post-competition supplementary evaluation under a prospectively fixed protocol after prior test exposure**。只用 train/dev 选模型；所有预定模型、checkpoint、推理参数和预测 hash 在新一轮 test 评分之前冻结；每个预定结果都报告，包括负结果。

不能重新调用已知 test 去找“最佳 encoder/vote/seed”，也不能把已经暴露的 test 通过换目录或加 hash 称为新独立盲测。采用新 untouched dataset 能加强独立性，但不在本轮冻结范围。

## 截止日前的执行与停止规则

- **9 月 7 日**：完成本阶段交付。第一阶段没有启动训练。
- **9 月 8 日起**：实现 C、验证初始化/梯度/投票和数据 gate，然后按数据集×方法×seed 轮转执行核心 27 次；记录所有尝试和失败，不先跑完“最有希望”的方法再决定其余组。
- 核心完成后开展扩展 15 次；分析、来源核对和正文写作可在训练时推进。
- **9 月 13 日北京时间 18:00** 为内部结果整理节点。未完成的组如实标 pending，不按成绩删组，不用已完成的有利 seeds 替代三 seed 预定分析；不能计算缺少预定单元的八域宏平均。仅因时间/资源做变更时记录 amendment 和当时已见结果。
- **9 月 14 日北京时间 18:00** 为内部提交目标，不是官网截止小时。不能保证 42 次训练都能完成；不足时收缩论文主张和表格，不伪造完成状态。

## 复核与交接

审计/清单来源：[ICSAI_PHASE1_EVIDENCE.md](ICSAI_PHASE1_EVIDENCE.md)。后续训练的实现 gate 属于第二阶段任务，并非本阶段已经通过。

```
python scripts/icsai/verify_phase1.py
# 换机器恢复被 gitignore 的 clean gold 输入，保留原协议不变：
python scripts/icsai/freeze_protocol.py --restore-inputs
# 动态检查可单独复跑，但会更新 probe 报告；如环境改变需保留原报告并记录差异：
/home/g24tka12/miniforge3/envs/absa/bin/python scripts/icsai/probe_rethink_path.py
```

前两个读取/准备脚本无需 GPU；动态 probe 用合成 encoder，不联网。验证失败时定位差异，不能简单重建 lock 掩盖来源变化。
