# ICSAI v3：复现与证据入口

本版落实八项审稿修改。正文为 [paper_icsai_v3.pdf](paper_icsai_v3.pdf)，方法、逐 seed 与分层细节见 [补充材料](paper_icsai_v3_supplement.pdf)。原始 v2、七张结果表、模型实现与测试结果保持原样。本轮没有追加训练；新增分析是审稿后的描述性诊断，不加入原六项检验。

## 文稿与证据

| 项目 | 入口 |
|---|---|
| 八项修改及验收位置 | [ICSAI_REVISION_V3.md](ICSAI_REVISION_V3.md) |
| 原系统来源与版本 | [paper_iconip_current.tex](paper_iconip_current.tex)，按早期系统稿件引用；不推断最终出版状态 |
| 实验矩阵与数据身份 | [experiment_matrix.json](audits/icsai_phase1/experiment_matrix.json)、[gold_input_manifest.json](audits/icsai_phase1/gold_input_manifest.json) |
| 实现与预检锁 | [implementation.lock.json](audits/icsai_phase2/implementation.lock.json)、[preflight.json](audits/icsai_phase2/preflight.json) |
| 核心模型、预测与评分 | [protocol_lock.json](audits/icsai_core_test/protocol_lock.json)、[predictions.lock.json](audits/icsai_core_test/predictions.lock.json)、[test_statistics.json](audits/icsai_core_test/test_statistics.json) |
| 扩展模型、预测与评分 | [protocol_lock.json](audits/icsai_extension_test/protocol_lock.json)、[predictions.lock.json](audits/icsai_extension_test/predictions.lock.json)、[test_statistics.json](audits/icsai_extension_test/test_statistics.json) |
| 每次训练配置及逐 epoch 记录 | [icsai_archive/runs/](audits/icsai_archive/runs/)：42 次训练 |
| 阶段诊断协议与汇总 | [stage_protocol.json](audits/icsai_revision_v3/stage_protocol.json)、[stage_summary.json](audits/icsai_revision_v3/stage_summary.json) |
| 每个检查点的指标与标签序列 | [stages/](audits/icsai_revision_v3/stages/)：27 个 JSON 与对应 JSONL |
| 日语酒店两种 NULL 分层 | [implicit_results.json](audits/icsai_revision_v3/implicit_results.json)：逐 seed、集成、cF1、分类 F1 与计数 |
| 六项原始逐 seed 对照 | [seed_contrasts.json](audits/icsai_revision_v3/seed_contrasts.json) |
| arXiv MCP 文献核验 | [literature_verification.json](audits/icsai_revision_v3/literature_verification.json) |
| 文件身份与环境 | [artifact_manifest.json](audits/icsai_revision_v3/artifact_manifest.json)、[environment.json](audits/icsai_revision_v3/environment.json) |
| 数值与 PDF 验收结果 | [validation.json](audits/icsai_revision_v3/validation.json) |

源码包的隔离重建验收见工作区 `audits/icsai_revision_v3/archive_validation.json`；该记录包含 ZIP 哈希，因此不放入 ZIP，避免自引用。

## 模式一：从随包记录重建新表格

源码包保留仓库相对路径。解压后在根目录运行，Python 3.10 或更高版本即可，不需要 GPU、PyTorch、原始数据或模型权重：

```bash
python3 scripts/icsai/build_revision_tables.py
python3 scripts/icsai/verify_revision.py
python3 -m unittest discover -s tests -p test_icsai_revision_diagnostics.py
```

生成 `docs/icsai_tables_v3/` 中四张表；脚本从每句标签序列重算阶段指标并核验来源哈希。原始七张表随包保留；其原始生成脚本依赖本地原始预测和数据，不能将缺少这些资产的包称为全流程复现。

安装 Tectonic 后，在 `docs/` 编译：

```bash
tectonic --keep-logs paper_icsai_v3.tex
tectonic --keep-logs paper_icsai_v3_supplement.tex
```

当前工作区使用 `/home/g24tka12/miniforge3/envs/icsai-tex/bin/tectonic`。首次构建可能下载 TeX 包。`IEEEtran.cls`、原始与新增表格、方法图均在源码包中。`paper_iconip_current.tex` 仅作原系统溯源，未把原稿的单独编译依赖宣称为本包支持范围。

## 模式二：在本地完整资产上复核评分和原始预测

在具有完整原始数据与预测的工作区运行：

```bash
/home/g24tka12/miniforge3/envs/absa/bin/python scripts/icsai/verify_revision.py --local-assets
```

它核验原始 56 份 test 预测哈希、按独立连续匹配公式重算 cF1、检查原六项 Holm 校正，并对日语酒店 16 份分层评分调用固定官方评分器。它还核验新阶段诊断对应的 27 份原始开发集预测身份。不进行训练或选模。

官方数据源为 [DimABSA2026](https://github.com/DimABSA/DimABSA2026)，本实验官方 test gold 与评分器版本是 `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`。评分器路径为该版本下 `evaluation_script/metrics_subtask_1_2_3.py`，SHA-256 是 `75eee32e50c5ebbd0000ed810b818954abedb9899f56987227c813667e3f9dbe`。本地预期根目录为 `data/official_test_gold/DimABSA2026/<上述版本>/`。训练/开发集的预期路径和 SHA-256 见输入 manifest；仅取得同名文件并不足以满足身份检查。

原预测存放在 `predictions/icsai_core_test/`、`predictions/icsai_extension_test/`，核心开发集预测在 `predictions/icsai_phase1/<run_id>/dev.jsonl`。文件身份以预测锁和阶段协议为准。

## 模式三：从原检查点重新执行阶段诊断

这需要约 27 个原始检查点、原训练/开发集文件、锁定模型和原始开发集预测。使用 `environment.yml` 了解基础依赖，实际诊断版本见 `environment.json`。原环境中的 PyTorch/Transformers 版本和 tokenizer 行为必须保持一致；不要为了消除 warning 改动 tokenizer 设置后仍称作原预测复现。

编码器为 `microsoft/mdeberta-v3-base`，revision 为 `a0484667b22365f84929a935b5e50a51f71f159d`。原检查点位置与 SHA-256 见 core protocol lock；恢复时使用 `src.models.icsai_control.load_control_checkpoint`，包含 EMA，不能用通用加载器丢失 A/B/C 干预设置。

在原机器的根目录运行：

```bash
/home/g24tka12/miniforge3/envs/absa/bin/python scripts/icsai/replay_revision_stages.py --output-dir docs/audits/icsai_revision_v3_replay --device cuda
```

输出目录必须不存在，防止将随包旧结果的检查误报为新推理。新结果与原结果分开；最终 quad 输出必须逐项等于原始开发集预测。所有 checkpoint 会先核验哈希；没有优化器步骤。原 `revision_diagnostics.py stages` 命令会复用并验证已有完整记录，不能把该命令的快速结束称为重新执行了全部推理。

跨机器恢复存在明确限制：checkpoint 配置保存了原编码器绝对路径。`HF_HUB_CACHE` 只控制配置工厂查找，不能自动重写 checkpoint 内路径。在其他机器必须提供该绝对路径下的相同只读模型快照，或另行实现并记录仅重定位路径的加载适配；本轮未验证后一种跨机器方案。缺少任一锁定资产时，完整推理复现不可执行，模式一仍可运行。

## 包含与不包含

`paper_icsai_v3_source.zip` 是本地审阅与源码证据包，包含两份新 PDF/TeX、原始 v2、原系统 TeX、脚本、配置、阶段标签序列、原始汇总及审计记录，另有逐文件 SHA-256 清单。完整模型权重、原始数据集和原始 quad 预测不随包分发，不能从哈希还原这些资产。本轮未发布、上传或投稿，也不将文献核验记为作者已亲自阅读。
