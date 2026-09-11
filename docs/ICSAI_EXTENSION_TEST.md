# ICSAI 五域扩展评估与八域汇总

入口：`scripts/icsai/extension_test.py`，2026-09-10 启动于持久执行会话。

任务自动依次执行：

1. 对全部15份 dev 预测检查 ID、重复三元组、VA 范围；重新计算本地评分并对齐训练选模分数；逐份官方 evaluator 交叉核验，不重新选 checkpoint。
2. 核验既有实现冻结，计算全部15个最佳 checkpoint 的完整 SHA-256，冻结模型、五域 test 输入与固定推理方案。
3. 核对每个 checkpoint 哈希，生成15份 test 单模型预测和5份固定三 seed、vote=2 集成；全部20份预测冻结后才读取 test 标签并统一评分。
4. 逐份与固定版本官方 CLI 对照，输出各 seed、离散 F1、匹配 V/A MAE和覆盖率。核对已完成核心三域的预测锁，合并成24个 A 单模型和8个固定集成的八域结果表。等权八域宏平均仅作描述，不新增显著性检验。

进度与日志位于 `docs/audits/icsai_extension_test/status.json` 和 `pipeline.log`。状态必须结合 PID 存活检查。任务遇错停止并写入 failed，不覆盖已有结果；重启须核对已完成阶段。

产物：`dev_statistics.json`、`protocol_lock.json`、`predictions.lock.json`、`test_statistics.json`、`eight_domain_summary.json` 和 `eight_domain_summary.md`。最终状态 complete 才表示全部完成。

本轮属于历史 test 暴露后的固定补充评估，不是盲测；不得据此重新选择模型、阈值或 seed。五域只评估 A，不能把原三域 Rethink 消融外推为八域结论。
