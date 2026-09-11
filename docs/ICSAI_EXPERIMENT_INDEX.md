# ICSAI 实验与稿件记录索引

## 2026-09-12 修订版

八项审阅修改的对应位置见 [ICSAI_REVISION_V3.md](ICSAI_REVISION_V3.md)。当前文稿为 [v3 正文](paper_icsai_v3.pdf)、[补充材料](paper_icsai_v3_supplement.pdf)，[复现入口](ICSAI_REPRODUCIBILITY_V3.md)区分随包表格重建与需本地资产的推理复现。新增 27 个固定检查点的阶段诊断和日语酒店 NULL 分层，均未追加训练。以下第二稿记录作为历史快照保留。

## 已完成

- 42次训练：三域A/B/C各3seed共27次，另五域A各3seed共15次。
- 官方test评分交叉核验：42个单模型、14个固定三seed集成，共56份预测。
- 六项预定对照及Holm校正；八域A单模型/集成描述性汇总。
- 60次重复成本测量、9条独立模型加载记录。
- 3个可追溯案例、6页第二稿、7张表、1张图、11篇引用。

## 入口

- 设计和输入来源：`audits/icsai_phase1/`
- 实现/初始化/测试证据：`audits/icsai_phase2/`
- 训练队列：`audits/icsai_core_training/`、`audits/icsai_extension_training/`
- 官方结果：`audits/icsai_core_test/`、`audits/icsai_extension_test/`
- 成本与案例：`audits/icsai_cost/`、`audits/icsai_cases/`
- 逐次运行元数据：`audits/icsai_archive/runs/`（42个run，每个含30轮epochs、status、初始化指纹、解析后配置）
- 压缩原始日志：`audits/icsai_archive/logs/`；解压后SHA-256见`log_manifest.json`。
- 论文：[第二稿PDF](paper_icsai_v2.pdf)、[源码包](paper_icsai_v2_source.zip)、[验证记录](audits/icsai_manuscript/validation.json)。

## 存储边界

仓库包含代码、配置、论文、结果JSON、输入/模型/预测哈希、逐轮记录和压缩日志。完整模型权重、数据集及逐样本预测仍在本地，由现有.gitignore排除；不把GitHub记录归档称为全部权重备份。恢复实验需要取得对应数据和模型。带有本地绝对路径的配置是溯源记录，在其他机器运行时需按说明配置路径。

历史test已暴露，本文是赛后固定方案补充评估。完整实验记录不意味着盲测或投稿已完成。
