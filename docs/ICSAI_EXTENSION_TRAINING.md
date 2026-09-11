# ICSAI 五域单阶段扩展训练

2026-09-09 启动，执行 Phase-1 已冻结的全部15项 uniform_extension 实验。数据集为 eng_laptop、jpn_hotel、rus_restaurant、ukr_restaurant、zho_laptop；每域 A 单阶段模型，seed 42/123/456，gold-only，每次30轮。无新超参数、增强或迁移初始化选择。

运行入口 `scripts/icsai/extension_queue.py`。复用已验证训练、推理与审计实现，原核心队列和冻结实现文件未修改。队列顺序按 seed，再按冻结设计的数据集顺序。

任务会自动完成15次训练、逐项核验30轮/实际更新步数/最早并列最佳 epoch/最佳及最后 checkpoint/EMA，再导出15份 dev 预测与五域均值、样本SD。已有核心训练中的三个 A 数据集加上本批五域，构成八域单阶段训练覆盖；本批完成不等于八域 test 评估完成。扩展 test 仍须冻结模型和预测后统一评分。

主要记录：

- `docs/audits/icsai_extension_training/status.json`：当前任务与进程 PID；必须与实际进程一起检查。
- `queue_manifest.json`：固定队列、设计和实现哈希、运行环境。
- `queue.log`、各 run 的 train/predict-dev 日志：详细运行记录。
- `progress.json`、`completed_runs.csv`、`dev_summary.json`：已完成项与训练时间估算。
- `checkpoints/icsai_phase1/{run_id}/epochs.jsonl`：当前 run 的逐轮进度。

总计450轮、131130次 optimizer 更新。按此前九次 A 核心训练的累计耗时/步数估算约12小时，仅是启动时粗估，未覆盖新域序列长度差异。完成首项后队列会使用实测耗时自动更新估算。遇到失败停止并记录错误，不自动覆盖或重复训练。
