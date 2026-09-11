# ICSAI 核心 27 次训练：执行与监控

启动日期：2026-09-07。执行已获用户授权。第一阶段实验矩阵、三个 seeds、三个数据集及 A/B/C 定义不变。

## 已启动的队列

固定次序：**seed → 方法 → 数据集**，每个 seed 内依次 A/B/C，每个方法内依次英文餐厅、中文餐厅、Tatar 餐厅；共 27 次。确切顺序和代码 hash 在 [queue_manifest.json](audits/icsai_core_training/queue_manifest.json)，首个任务为 `eng_restaurant_A_s42`。

单张 RTX 4090 串行运行，固定 OMP/MKL 线程数为 4，禁用 tokenizer 额外并行，使用本地预训练权重，不联网下载。队列有进程文件锁，每个 run 有排他 claim；同一任务不会被自动重复启动。

每次 run 固定 30 epochs。完成后自动核对：

- 30 条完整 epoch 记录、每轮更新数、总更新数；
- best checkpoint 对应 dev cF1 最佳且同分最早的 epoch；latest 为 epoch 30；
- checkpoint 的 run/arm 标识和 EMA 权重；
- 已完成的同数据集同 seed 各方法样本排列 hash 完全一致。

核对通过后恢复 best EMA 模型，导出 **dev 预测**，检查 ID 顺序与单 seed 唯一三元组。随后进入下一个任务。每个方法收齐三 seeds 后自动汇总 dev 均值、样本 SD。任何子进程失败或校验失败都会停止整个队列并记录原因，不跳过失败方法挑选好结果。

## 当前状态与日志

```bash
python scripts/icsai/core_status.py
```

该命令同时核查 `/proc` 中的实际 queue/child 命令行，避免仅靠旧状态文件误判仍在训练。

- [status.json](audits/icsai_core_training/status.json)：当前任务、阶段、进程号、完成数。
- [events.jsonl](audits/icsai_core_training/events.jsonl)：各次开始、完成和错误事件。
- `docs/audits/icsai_core_training/{run_id}.train.log`：完整训练日志。
- `checkpoints/icsai_phase1/{run_id}/epochs.jsonl`：每轮 dev 指标、实际更新数、顺序 hash、耗时。
- 第一个完整 run 完成后生成 `progress.json`，给出按实际更新耗时估计的剩余训练时间；它不保证不同方法/语言的运行成本相同。
- 完成记录写入 `completed_runs.csv` 和 `dev_summary.json`。这些属于 dev 结果，不是 test 成绩。

队列独立于对话进程运行；后台程序负责推进和失败停止，不代表助手在对话结束后持续人工观察。再次查看时以实际进程和日志为准。

## 边界

此队列**只执行核心 27 次训练与 dev 导出**，不会启动另外 15 次扩展，也不会评分 test。核心实验完成后再按预定流程做配对统计、冻结所选模型及预测，然后开展已定义的 test 对照。

若机器重启、训练被杀或资源不足，不要删除 claim 并重新执行。先核查当前进程与 checkpoint，再实施带记录的恢复；当前入口不自动恢复未完成的 run。没有完整数据的组不输出三 seed 或八域宏平均结论。
