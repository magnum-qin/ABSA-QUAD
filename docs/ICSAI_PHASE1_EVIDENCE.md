# ICSAI 第一阶段：范围、八系统来源与证据清单

核实日期：2026-09-07。第一阶段仅审计、准备输入和锁定设计；没有启动训练、调用 LLM 标注、重新评分 test、提交论文或发送邮件。

## 1. 目标会议与范围

用户已确认目标为 **9 月 15 日截止**的会议，即 **2026 12th International Conference on Systems and Informatics (ICSAI 2026)**，2026-12-19 至 12-21，中国太仓。

- 官网：<https://www.icsai.com.cn/>，当前首页投稿截止日期为 **2026-09-15**。
- 投稿说明：<https://www.icsai.com.cn/submission.html>。英文 PDF；使用官网模板；**初稿没有页数上限**，录用后一次注册覆盖 **6 页**，超页另收费。因此以 6 页为排版目标，不能写“初稿最多 6 页”。
- 投稿入口：<https://www.icsai.com.cn/submission/openconf.php>。本阶段只查入口，未登录、未提交。
- 模板：<https://www.icsai.com.cn/download/latex.zip>，已下载核实含 `IEEEtran.cls`、`bare_conf.tex`。
- 首页没有明确截止小时或时区；不假定 AoE。内部提交目标锁定为 **9 月 14 日北京时间 18:00**，最终外部截止仍以会议系统为准。
- 网页解析器把日期显示成删除线，但 HTML 为 `<s style="text-decoration:none">`，不能据此认定日期失效。协办方 7 月新闻稿的 8 月 31 日为旧日期；本次以当前官网及用户确认的 9 月 15 日为依据。
- 官网页面和模板本地快照、SHA256：[conference_sources.json](audits/icsai_phase1/conference_sources.json)。网页后续可能更新，提交前复查日期即可。

论文范围锁定为 **LEADER 学生系统、BIO 反馈精炼的受控证据、统一配置与历史定制策略、集成成本及误差分析**。本轮不接入全 pair/VA 反馈、不新增 teacher/pool/KD 实验。KD 保留后续稿件主线；如复用 TALLIP 已有表，届时更新两篇论文的贡献边界。

## 2. 作为基准的 ICONIP 稿件

仓库稿：[`paper_iconip_current.tex`](paper_iconip_current.tex)，主表八域 dev 宏平均 0.5008，与所提供 review 对应。文件 SHA256 及当前 Git HEAD 已写入 [inventory.json](audits/icsai_phase1/inventory.json)。**作者已在本对话确认 docs 中的 tex 为 ICONIP 稿件，后续以该文件为投稿版本基准，不再要求另行提供 PDF。**

当前 `submission/final_manifest.json` 是后续 V23/KD 冻结，**不是 ICONIP 原八系统清单**。旧 `scripts/audit_best_reproduction.py` 与 `scripts/run_final_best_submission_4090.sh` 可提供历史路径，但也有阈值/文件漂移，须与实际文件交叉核对。未运行这些会覆盖 submission 的脚本。

## 3. 八个原稿最终系统

共同历史 seeds：**42、123、456、789、1024**。下表 checkpoint 前缀均位于 `checkpoints/`，应加 `{seed}/checkpoint_best.pt`。**八组原前缀下的 40 个 best checkpoint 当前全部缺失**；原脚本引用的八个 `/tmp` 配置也全部缺失。这是当前本地文件事实，不表示其他机器/备份没有。

`vote/topk/dual` 是历史脚本参考值，非全部已验证事实。原始 score、当前文件 hash、行数、各 seed 存在性、候选预测逐项清单见 [inventory.json](audits/icsai_phase1/inventory.json)。

| 数据集 | 原稿 dev cF1 | 编码器参考 | 历史 train 文件（`data/subtask_3/` 下） | 期望/当前行数 | 历史 checkpoint 前缀 | vote / topk / dual |
|---|---:|---|---|---:|---|---|
| eng_restaurant | 0.6949 | microsoft/deberta-v3-large | eng/eng_restaurant_train_alltasks.jsonl | 2284/2284 | v16_native_eng_restaurant_seed | 4 / 0 / 否 |
| eng_laptop | 0.3304 | microsoft/deberta-v3-base | eng/eng_laptop_train_alltasks_augmented.jsonl | **15049/12632** | v5_eng_laptop_seed | 3 / 0 / 是 |
| jpn_hotel | 0.4500 | ku-nlp/deberta-v2-large-japanese | jpn/jpn_hotel_train_alltasks.jsonl | 1600/1600 | v16_ablation_jpn_no_aug_seed | 4 / 0 / 否 |
| rus_restaurant | 0.5267 | xlm-roberta-large | rus/rus_restaurant_train_v13.jsonl | 1766/1766 | v15_xlmr_rus_seed | 4 / 0.10 / 否 |
| tat_restaurant | 0.4821 | xlm-roberta-large | tat/tat_restaurant_train_v13.jsonl | 1599/1599 | v15_xlmr_tat_seed | 3 / 0.20 / 否 |
| ukr_restaurant | 0.5068 | microsoft/mdeberta-v3-base | ukr/ukr_restaurant_train_v13b.jsonl | 2967/2967 | v13b_ukr_seed | **脚本 3，分数匹配文件 4** / 0 / 否 |
| zho_restaurant | 0.5669 | hfl/chinese-roberta-wwm-ext-large | zho/zho_restaurant_train_alltasks.jsonl | 6050/6050 | v16_native_zho_restaurant_seed | 4 / 0 / 否 |
| zho_laptop | 0.4487 | hfl/chinese-roberta-wwm-ext-large | zho/zho_laptop_train_v13c.jsonl | 5534/5534 | v16_native_zho_laptop_seed | 4 / 0.10 / 否 |

行数一致不能证明历史训练输入逐字相同；现在计算的 hash 仅锁定当前文件。历史预训练/transfer 来源不完全可恢复，不能因为叫 V5/V15 就断言实际加载成功。新实验全部从同一预训练编码器冷启动，不继承这些不确定路径。

### 保留配置与替代 checkpoint

- eng_restaurant：`configs/pipeline_eng_restaurant_v16.yaml` 可读，不能替代删除的 `/tmp` 原快照及缺失权重。
- eng_laptop：`configs/pipeline_eng_laptop_v5repro.yaml` 属于**后续重建**。保留 `v5repro_eng_laptop_seed{seed}`、`v5repro_eng_laptop_inductive_seed{seed}`，不能称为原 V5 checkpoint；6-pivot 系列也不是原 4-pivot 系统。
- jpn：`configs/pipeline_jpn_v16.yaml` 指向 **2893 行增强版**训练文件，不是原稿 no-augmentation 配置。
- rus：`configs/pipeline_rus_v15.yaml` 可读，原权重缺失。
- tat：`configs/pipeline_tat_v15.yaml` 在本地缺失；编码器由稿件/历史实验记录支持，不能宣称精确恢复训练配置。
- ukr：`configs/pipeline_ukr_v13b.yaml` 可读；保留的是 **inductive 重训**五 seeds，不是原 V13b。
- zho_restaurant：该名称的独立 V16 yaml 缺失，但五个 `final_zho_restaurant_v16_seed{seed}` 权重仍在，可从内嵌 config 恢复；它们是后续 final 重训。
- zho_laptop：`configs/pipeline_zho_laptop_v16.yaml` 及五个 `final_zho_laptop_v16_seed{seed}` 权重仍在，属于后续 final 重训。

实际读取 retained checkpoint 的配置记录见 [retained_checkpoint_metadata.json](audits/icsai_phase1/retained_checkpoint_metadata.json)；不加载 GPU，不运行推理。checkpoint 内容未做全量 SHA256，因此这份记录不是模型权重锁。

### 已有 dev/test 预测能复用到什么程度

以下 dev 分数是**用当前本地 scorer 对已有 dev 预测做诊断复核**，不是重新选系统，也不是官方 test 得分。同分/文件名均不能独立证明生成来源。

| 数据集 | dev 证据 | 已有非 KD test 候选（`submission/final/` 下） | 复用边界 |
|---|---|---|---|
| eng_restaurant | v16_missing_sweep/v16_eng_restaurant_vote4_topk0.0.jsonl 当前为 **0.699893**，不是 0.6949 | eng_restaurant_test.jsonl | 与 7 月 official-test 协议预测 hash 一致；可作为已冻结 V16 输出，不能证明它来自稿中 0.6949 的确切状态 |
| eng_laptop | 未发现复核为 0.3304 的候选 | eng_laptop_test_v20_backup.jsonl；另有 clean-v5repro official-test 预测 | **V20/clean-v5repro 均不能冒充原 V5/V6**；原 0.3304 对应 test 证据缺口 |
| jpn_hotel | v16_sweep/v16_jpn_vote4_topk0.0.jsonl 为 **0.446590**，对应增强设置分数而非 no_aug 0.4500 | jpn_hotel_test.jsonl | hash 匹配旧协议 no_aug 名称，但权重/精确配置不在；不得把 0.4466 的种子表解释成 no_aug |
| rus_restaurant | v15_sweep/v15_rus_vote4_topk0.10.jsonl → 0.526737 | rus_restaurant_test_v15_backup.jsonl | dev 数值、命名、旧协议 test hash 支持；生成权重缺失，只能复用预测，不能直接补单 seed 推理 |
| tat_restaurant | v15_sweep/v15_tat_vote3_topk0.20.jsonl → 0.482072 | tat_restaurant_test_v15_backup.jsonl | 同上 |
| ukr_restaurant | v13b_ukr_vote4.jsonl → **0.506848**；vote3 → 0.495212 | ukr_restaurant_test_v13b_backup.jsonl | **阈值矛盾未消除**，此 backup 不在 7 月清洗版 test 锁中；暂列待核实，不能拼成严格同配置主表 |
| zho_restaurant | final_dev_verify/zho_restaurant_dev_verify.jsonl → 0.566944 | zho_restaurant_test_v16_backup.jsonl | 可复用后续 final 重训预测/权重；test hash 匹配旧协议，不改写为原权重 |
| zho_laptop | final_dev_verify/zho_laptop_dev_verify.jsonl → 0.448682 | zho_laptop_test.jsonl | 同上 |

所有这些 test 候选均已检查 ID 集合及行数与无标签 test 输入一致；**本阶段没有读取 released test gold 或重新评分这些预测**。未发现某文件的范围仅限本地扫描的 `predictions/` 与 `submission/final/`，不等于全世界没有备份。

## 4. 清单一：已有证据可以复用什么、必须补什么

| 证据/问题 | 可以复用 | 还必须补 | 优先级 |
|---|---|---|---|
| 原稿与 reviewer 对应关系 | 作者确认的 docs/paper_iconip_current.tex、评审所引用数值、文件 hash | 已确认，无需另行提供稿件 | 已完成 |
| 八域主结果 | 六个非 KD test 文件与旧 official-test 锁 hash 相符；另有 ukr 未锁 backup | 将原稿 best、已冻结输出、clean 重训分开；补 eng_laptop 来源、ukr 阈值；无法恢复时明确新重训，不造原八域宏平均 | P0 |
| 原始权重/配置 | 后续 zho final、eng/ukr 重训与对应配置 | 原八组权重缺失；先查个人备份，有则核验，无则采用新协议重训 | P0，备份可选 |
| Rethink 架构 | 当前源代码、与 ICONIP 参考提交相同的代码、动态干预/梯度结果 | 改写 BIO-only 描述；按固定 A/B/C 设计完成受控实验，不能声称 VA/category 反馈有效 | P0 |
| 种子与集成 | 原文已有 best seed/mean，英文 Rethink 甚至已有 5-seed 历史表；部分 ensemble 预测 | 单 seed 预测来源、SD、paired CI、dev 最佳单模型 test；历史表可作描述，不能替代受控新实验 | P0 |
| 统一配方 | 同一 mDeBERTa snapshot 可用，八域 clean gold 输入已准备 | A 组八域分别训练；核心三域 A/B/C；不叫“统一权重多语言模型” | P0/P1 |
| 数据增强 | 旧配方/部分增强文件；KD 审计脚本可借鉴结构 | 对实际 paraphrase/translation 作独立标签审计；KD fidelity 不替代增强保真 | P1 |
| 评估指标与资源 | 官方 evaluator commit、既有 test 审计、局部成本结果 | 离散提取 F1、matched VA/coverage、显隐式和边界切片；LEADER 端到端成本 | P0 分析 |
| dev 选择曝光 | 本地预测文件清单和旧脚本/日志 | 汇总真正尝试的训练配置与解码设置，去掉副本；当前 dev 候选文件数不是配置尝试数，不能直接写进论文 | P0 |
| 外部比较 | 可定位原 shared-task 系统论文和官方测试协议 | 逐外部行核实 split/数据/评估范围；跨 split 排名删除 | P0 写作 |

## 5. 已有 test 暴露与新输入

现有 `official_test/protocol_lock.json` 记录 2026-07-18 冻结，官方数据 commit 为 `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`；当前仓库已有 released gold 和 test 结果。本对话此前也已读到结果。因此 **9 月新增实验属于看过 test 结果之后预先固定设计的赛后补充评估**，不能再宣称新 blind/hidden test，也不能用新冻结追溯消除旧调参乐观偏差。7 月历史锁的时间顺序是记录支持的历史声明，不是本阶段重新见证的首次读取事件。

新实验仅用 gold train，规范化为 NFKC+casefold+whitespace 后，删除与 dev 或无标签 test 文本完全重合的训练行，保留其他行顺序/重复权重。没有读取 test 标签；这种清洗不能排除语义近重复，也不证明预训练模型没见过基准。

| 数据集 | 原始 gold 行数 | 新协议行数 | 去掉原因 |
|---|---:|---:|---|
| eng_restaurant | 2284 | 2284 | 无 |
| eng_laptop | 4076 | 4075 | 1 条 dev 重合 |
| jpn_hotel | 1600 | 1597 | 1 条 dev、2 条 test 输入重合 |
| rus_restaurant | 1240 | 1240 | 无 |
| tat_restaurant | 1240 | 1240 | 无 |
| ukr_restaurant | 1240 | 1239 | 1 条 dev 重合 |
| zho_restaurant | 6050 | 6050 | 无 |
| zho_laptop | 3490 | 3490 | 无 |

清洗副本在 `data/icsai_phase1/gold_clean/`，原数据未改。行号、删除原因及全部输入 hash 见 [gold_input_manifest.json](audits/icsai_phase1/gold_input_manifest.json)。这些数据文件受现有 gitignore 管理；换机器需运行准备脚本并核对 hash，而不是只复制文档。

## 6. 需要人工介入的位置

1. **目标会议已由用户确认，当前无需再次确认。** 截止小时官网未给出，按提前一天的内部目标安排；若需要用到最后几个小时，再向老师/会议方核实。
2. **ICONIP 稿件已由作者确认位于 docs，稿件确认事项已完成。** 原 checkpoint 备份如有可提供位置；没有也能继续新协议，但不能宣称恢复原权重。备份不是继续工作的前置条件。
3. **ICONIP 已撤回（作者在本对话确认，记录日期 2026-09-07）。** 撤回状态确认事项已完成，不再作为改投前的待办；本代理未代办撤回。
4. 增强标签的人工作业需要具备目标语言能力的标注者；本阶段只是设计审计，不假造人工结论。

第一阶段的交付是**明确事实和缺口并锁定下一阶段设计**，不是把缺失资产全部恢复或把实验跑完。固定实验矩阵见 [ICSAI_PHASE1_PROTOCOL.md](ICSAI_PHASE1_PROTOCOL.md)，架构证据见 [rethink_audit.md](audits/icsai_phase1/rethink_audit.md)。
