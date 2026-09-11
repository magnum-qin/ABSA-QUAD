# ICSAI：核心训练完成后的执行顺序

更新：2026-09-09。27/27 核心训练完成；27 份 dev 预测已重新评分，并逐份与官方 evaluator 交叉验证。

## 当前证据

A 为单阶段模型，B 为 BIO 反馈 Rethink，C 为同结构但将 BIO 反馈置零。官方 dev 评分下，B−A 在英文餐饮为 +1.80 个百分点、中文餐饮为 −1.61、鞑靼语餐饮为 −2.14。六项 B−A/B−C 对照经 Holm 校正均未达到 0.05 显著性水平。不能将不显著解释成等效，也不能声称反馈普遍有效。

英文 dev 的官方 scorer 与训练时本地 scorer 存在差异；保留原本地 dev 选出的 checkpoint，不按官方重评分重新选择。完整统计见 [dev_statistics.md](audits/icsai_core_analysis/dev_statistics.md)。

## 后续顺序

1. 冻结全部 27 个 best checkpoint 的完整文件 SHA-256、输入、推理参数与统计方法。冻结脚本为 `python -m scripts.icsai.freeze_core_test`，成功后生成 `docs/audits/icsai_core_test/protocol_lock.json`；日志为 `docs/audits/icsai_core_analysis/test_freeze.log`。
2. 按锁定方案生成全部 27 份 test 预测；必须使用保留 C 干预的 `load_control_checkpoint`。先检查样本 ID、重复三元组与 VA 范围，再冻结预测文件哈希，然后统一评分。不得根据 test 结果选择 seed、阈值、模型或数据集。
3. 报告每 seed、均值与样本 SD、六项配对 bootstrap 对照及 Holm p；附离散 F1、匹配 VA MAE/覆盖率和显式/隐式样本数量。九个固定三 seed 投票集成作为次要结果，测量推理耗时与显存以回应成本质疑。
4. 完成其余五域 A 模型的 15 次既定训练，补齐单模型八域覆盖。不能据此把三域反馈消融外推到全部八域。
5. 将证据写入 ICSAI 稿件：准确说明 BIO 反馈路径、负结果、评分口径、历史模型溯源缺口及增强数据局限。新实验明确标为已有 test 暴露后的赛后补充评估，不能宣称盲测。完成格式、引用和编译检查，按既定内部时间提交。

当前新 test 预测与评分尚未执行；不要把模型冻结完成等同于 test 实验完成。增强保真度和成本实验仍需补充可复核证据。
