# ICSAI 核心 test 自动评估

启动日期：2026-09-09。核心训练已完成27/27；全部 dev 预测已核验。本任务执行原冻结方案中的三数据集、A/B/C、三个 seed 的 test 对照，不根据新 test 分数进行选模或调参。

## 运行入口与阶段

入口：`scripts/icsai/core_test.py`。任务运行于持久执行会话；首次后台启动在等待阶段退出，已核对未产生推理产物后重新启动并保留启动记录。PID 记录在 `docs/audits/icsai_core_test/pipeline_claim.json`。当前阶段以 `status.json` 为准，同时检查对应 PID 是否仍存在；旧状态文件不能单独证明任务仍活跃。

1. 等待 `protocol_lock.json` 完成：包括全部27个最佳 EMA checkpoint 的完整文件 SHA-256。
2. 逐个核对 checkpoint 哈希，使用 `load_control_checkpoint` 加载模型并保持 C 的 BIO 反馈置零。生成27份固定 test 预测。
3. 校验所有预测的 ID、每 seed 每三元组最多一票和 VA 范围，生成全部9个三 seed、vote=2集成。
4. 写出 `predictions.lock.json`，然后才读取官方 test 标签并评分。全部36份预测逐份与固定版本官方 CLI 交叉核验。
5. 报告单模型每 seed、均值与样本SD、六项配对样本 bootstrap（10000次，seed 20260907）及 Holm p；集成仅作次要描述结果。输出离散抽取F1、匹配 V/A MAE和覆盖率、显式/隐式样本数量。

进度：`docs/audits/icsai_core_test/status.json`、`prediction_progress.json`。

日志：`docs/audits/icsai_core_test/pipeline.log`；模型冻结日志在 `docs/audits/icsai_core_analysis/test_freeze.log`。

结果：成功后生成 `docs/audits/icsai_core_test/test_statistics.md` 和 `test_statistics.json`，最终状态为 `complete`。运行失败会写入 `failed` 与错误详情，不能把已生成的部分预测当成完整结果。脚本拒绝覆盖已有任务及预测；重启前须核对失败阶段。

## 已完成的流程检查

`preflight.log` 记录 vote=2、VA 算术平均、单票排除、重复/ID/无效VA拒绝检查。C 模型恢复与因果干预已由 Phase-2 的模型测试和 GPU smoke 验证；本流程复用其加载入口。

## 解释限制

历史 test 已暴露，本轮是固定方案的赛后补充评估，不能称为盲测。置信区间仅反映固定训练 seed 条件下的样本不确定性；不能证明训练程序整体显著性或三域之外的泛化。推理耗时为包含首批的一次测量，模型/分词器加载单独记录；显存为推理阶段峰值分配量。这些仅作为描述性资源记录，不能替代后续重复成本基准。
