# ICSAI 稿件初稿（2026-09-10）

2026-09-12 更新：当前修订版为 [v3 正文](paper_icsai_v3.pdf)与[补充材料](paper_icsai_v3_supplement.pdf)。八项修改见 [对应表](ICSAI_REVISION_V3.md)，[复现说明](ICSAI_REPRODUCIBILITY_V3.md)及 [源码包](paper_icsai_v3_source.zip)另附。旧稿及以下历史记录保持保留。

主文件：[paper_icsai_draft.tex](paper_icsai_draft.tex)；预览：[paper_icsai_draft.pdf](paper_icsai_draft.pdf)。原 ICONIP tex 保留。当前第二稿6页，是供老师审阅的版本，不是最终投稿版本。

## 已完成

英文摘要、引言、简要相关工作、方法、实验设置、已有结果讨论及结论。作者顺序、单位和邮箱沿用原稿。标题暂定为 Revisiting BIO Feedback for Multilingual Dimensional Aspect Sentiment Quadruple Extraction，围绕已完成的固定对照组织论证。

初稿五张表涵盖数据规模、八域 A 单模型与三 seed 集成、三域 A/B/C 消融、六项 Holm 校正检验、抽取及匹配 V/A 误差。表格通过 `scripts/icsai/build_paper_tables.py` 从结果 JSON 生成，重新验证56份 test 预测哈希及结果来源锁；未手工选择有利 seed。

方法与代码逐项核对：仅 BIO 为有效反馈；pair 表示为两段池化向量拼接；Hungarian 代价不含 pair-validity，VA 匹配为归一化 L1；回归训练为 Huber；consistency 仅约束 BIO 标量损失。明确 C 分配参数相同但有效梯度不等价。另明确实际候选构建没有额外 learned NULL span，不能声称完整支持隐式候选生成。

已编译并逐页查看PDF；最终日志无越界盒、未定义引用。验证记录位于 `audits/icsai_manuscript/validation.json`，页面截图在同目录。少量 underfull 排版警告不影响可读性。

## 证据边界及后续

- 本轮42次训练全部完成；五域扩展 test 已完成，八域表已纳入。
- 不声称 Rethink 普遍有效，不混用旧开发集排名和新测试集成绩；显著下降及不显著结果完整保留。
- 明确历史 test 已暴露，不能称盲测；bootstrap 只涵盖固定训练 seed 下的样本不确定性。
- 9月11日工作包已完成：新增第六张表，报告60次重复推理成本测量。只支持固定英文工作负载下的成本描述，不外推生产或全语言性能。
- 当前论文以 gold-only 对照为中心，未主张旧增强数据的保真度或收益。若后续恢复该贡献，需另补保真度审计及相应证据。
- 最终投稿前需加强相关工作与贡献定位、完善结果分析，检查作者信息和最终格式。当前不把内部 A 对照称为外部 SOTA 基线。

## 格式与引用来源

采用会议[投稿页](https://www.icsai.com.cn/submission.html)提供的 LaTeX 模板中 IEEEtran.cls（从已存档 conference_latex.zip 提取）。官网说明初投不限页数，注册包含6页；本版未压缩到人为页数限制。

核对来源：[ACOS](https://aclanthology.org/2021.acl-long.29/)、[ASQP](https://aclanthology.org/2021.emnlp-main.726/)、[DimABSA](https://arxiv.org/abs/2601.23022)、[共享任务概述](https://arxiv.org/abs/2604.07066)、[DeBERTa官方实现](https://github.com/microsoft/DeBERTa)、[Hungarian原文](https://onlinelibrary.wiley.com/doi/10.1002/nav.3800020109)。修正原 ASQP 引用的作者列表。

## 重建

在仓库根目录运行 `/home/g24tka12/miniforge3/envs/absa/bin/python scripts/icsai/build_paper_tables.py`。

随后在 docs 目录运行 `/home/g24tka12/miniforge3/envs/icsai-tex/bin/tectonic --keep-logs paper_icsai_draft.tex`。

LaTeX 编译环境独立安装在 icsai-tex 中。主文件依赖同目录 IEEEtran.cls、icsai_tables 中7个 tex 表格及 icsai_figures/feedback.tex（TikZ）。

成本表由 `scripts/icsai/integrate_cost.py` 在测量完成后生成；原始记录与范围限制见 `audits/icsai_cost/`。

## 第二稿完成

第二稿快照为 [paper_icsai_v2.pdf](paper_icsai_v2.pdf)，源码包为 [paper_icsai_v2_source.zip](paper_icsai_v2_source.zip)。新增5篇相关工作、固定规则3个案例及方法示意图，核验记录见 `audits/icsai_manuscript/review_0912.json`。下一阶段需要老师实际反馈；未发送稿件或执行投稿。
