# ICSAI 第三稿：八项修改对应表

本轮依据作者“完成你所建议的8点”的指示修订 v2。产出 [v3 正文](paper_icsai_v3.pdf)、[补充材料](paper_icsai_v3_supplement.pdf)和 [复现说明](ICSAI_REPRODUCIBILITY_V3.md)。这是 ARS 指导下的有界 LaTeX 修订与实证诊断，不宣称完成了独立盲审、会场标准绑定或 ARS 全流水线机械认证。旧 v2 原样保留。

| # | 修改 | 最终位置与证据 |
|---|---|---|
| 1 | 摘要不再把 B–A 差异直接归因于反馈；加入 B–C 三个差值及校正结论 | 正文 Abstract；与原 core test 六项 contrasts 对应。B–C 同时影响 forward content 和 gradient 的限制也进入摘要。 |
| 2 | 说明原系统出处、版本、旧主张与新实验区别 | Introduction 的实现说明、补充 S1 的代码和实验锁；2026-09-15 按 Suzuki 老师意见删除未发表 ICONIP 稿件的正文引用及参考文献条目，历史源文件仅保留作内部归档。 |
| 3 | 明确 B/C 的直接 BIO 监督只在最终输出，初稿 loss 为 detached margin 参照 | Method / Assignment and Training Objective 末段；补充 S2；对照原模型 `_compute_loss` 和 forward 输出覆盖逻辑。 |
| 4 | 说明 C 切断 embedding、probability、初稿头反馈任务梯度及融合输入列数据梯度 | Method / BIO Feedback and Its Control；Results 的 B–C 表述；补充 S2 解释 margin gate、weight decay/EMA 边界。 |
| 5 | 完成全部三域、三 seed、A/B/C 共 27 检查点的开发集阶段诊断 | Results / Intermediate BIO Diagnostics、主表 V；补充 S3 和阶段明细表；`stage_summary.json` 与 27 组 tag traces。全部最终输出精确复现原开发集预测，无训练或重新选模。 |
| 6 | 补入 IMN 与 IMKTN，区分迭代消息传递、routing 和本稿单 BIO 通道 | Related Work / Representation Refinement and Attribution；新增两条正式引用；补充 S6；arXiv MCP 和 ACL 元数据核验记录。 |
| 7 | 修正 NULL 机制，并计算日语酒店两类分层 | Method / Single-Stage Extraction；Results / Implicit-Element Coverage；补充 S4 和分层表。152/800 句含隐式元素，175/1443 个 gold quad 隐式；3 个 seed 和固定集成共 16 份分层评分经原官方脚本核验。 |
| 8 | 提供逐 seed 差值及协议、代码、数据、模型、预测、评分器证据入口 | 正文 Companion Materials and Versioning；补充 S1/S5/S6；`ICSAI_REPRODUCIBILITY_V3.md`；源码包与逐文件 manifest。说明随包表格重建与需本地资产的完整复现之区别。 |

## 与最初审阅意见的校正

NULL 支持的原审阅意见不够完整。数据预处理实际上给 NULL aspect/opinion 分配 CLS 位置的 BIO 标注。修订保留“没有额外 learned NULL candidate”的事实，但不再暗示完全没有监督的 NULL 通道。真正的边界是单层 BIO 对同一个 CLS 上两种类型的冲突，aspect 优先；特殊 token 的零偏移也使 NULL 输出不等于正确隐式识别。

新阶段结果同样以实际证据为准：B 的三域初稿平均 span F1 为 0.102、0.082、0.050，最终为 0.856、0.729、0.707。低初稿 decoded F1 并不证明 soft 表示毫无信息，也不能解释 Tatar test 下降的原因。C 的第二阶段头仍通过最终调用得到监督。没有把这些观察改写成反馈普遍无效的结论。

## 验收边界

- 原始 132 个受保护文件的 SHA-256 保持一致，包括 v2、旧表格、模型实现和核心/扩展评分记录。
- 27 组阶段 trace 重新计算指标及 21 个汇总行；阶段推理时逐项核对了最终开发集 quads。
- 56 份原 test 预测的独立 cF1 重算最大误差约 `7.8e-16`；六项 Holm 和 18 个配对 seed 差值一致。
- 日语酒店 16 份分层评分与固定官方脚本一致。
- 新增诊断的五项边界测试通过；正文与补充材料编译及逐页视觉检查单独记录。
- 源码包的资产清单与隔离重建结果见 `docs/audits/icsai_revision_v3/archive_validation.json`；不把包中未提供的权重/数据/原 quad 预测称为完整备份。

正式投稿前仍由作者确认会场格式与作者信息。本轮仅完成这八项修改，没有额外训练、发布、投稿或代替作者作出版声明。

## 2026-09-15 导师意见修订

删除未发表 ICONIP 稿件的正式引用，并移除补充材料中以该稿件作为论文依据的表述；参考文献现为 13 条。贡献段采用 “Our main contributions are as follows:” 及 (1)、(2)、(3) 编号。实验数值不变。
