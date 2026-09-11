# ICSAI 下一步：A/B/C 对照实现与训练入口

本阶段对应“实现并验证 C 组无 BIO 反馈对照”。第一阶段冻结设计与历史源文件保持不变；当前实现状态在本文件及 `docs/audits/icsai_phase2/` 单独记录，不覆盖历史审计或预注册矩阵。

## 实现

- `src/models/icsai_control.py`：通过明确的 LabelEmbedding 子类，先计算原始 soft BIO embedding，再在 C 组乘 0；B 返回原始 embedding。其余 token fusion、attention、FFN、gate、两次循环、预测头和损失沿用原模型。
- B/C state-dict key、tensor shape、总参数量相同；替换模块时恢复原权重并隔离初始化 RNG，不增加随机数消耗。A/B/C 共有参数通过实际字节 hash 核对。
- C 的开关属于 checkpoint 的 `config.icsai.arm`，不是权重张量。**必须使用本实验的 checkpoint loader / dev 推理入口**恢复它，旧 `scripts/inference.py` 不理解该控制开关。
- `scripts/icsai/control_runtime.py`：独立于模型 RNG 的逐 epoch 排列；严格按 dev cF1 选 checkpoint，完全同分保留最早 epoch，包括 cF1=0；固定 30 epochs，不采用原入口的 val_loss 平局回退或 early stopping。
- 固定输入、label map、seed、超参及输出路径全部从第一阶段 JSON 生成，42 个配置在 `configs/icsai_phase2/`。训练配置没有 test_file，训练命令不接受测试集路径或额外调参参数。
- checkpoint 保存仍采用既有 Trainer 的 best/latest 原子保存、EMA 和 top-k 实现，配置 top-k=1；每 epoch 写 `epochs.jsonl`，记录实际更新数、样本顺序 hash 和 dev 结果。

## 验证范围

1. 八项合成输入测试覆盖：B 与旧模型输出逐字相同；A/B/C 初始化一致性；C 屏蔽 BIO 内容但保留有效精炼梯度；三个方法都能进行真实 loss backward/optimizer step；C checkpoint/EMA 恢复；逐 epoch 排列不受随机数消耗影响；零分同分时选最早 epoch且不早停；重复投票与 ID 错位拒绝；全部 42 配置与设计一致（部分检查归入同一测试）。结果在 `unit_tests.log`。
2. 使用固定 mDeBERTa 预训练 snapshot，实际构造三个 seed × A/B/C 共九个模型，核对共有参数 hash 与 B/C 完整参数 hash。三核心域有序 7/6 label map 完全相同，先验证该事实，再用英文域作为网络构造代表。其他两个域正式运行时仍逐个核对实际初始化，不能只靠 seed 名称宣称一致。
3. GPU 冒烟检查：每个方法在前 8 条 clean 英文训练样本上做两个独立优化器更新；第二次手动进入 FGM 分支，检查有限 loss/梯度、EMA 路径及预测唯一性。**这不是正式实验，不参与论文分数，不保存训练 checkpoint，也不评分 dev/test**。初始权重、梯度统计、峰值显存及限制见 `gpu_smoke.json`。
4. 实现锁及 `verify_control.py` 校验上述证据、实现文件和配置 hash；正式 train 命令必须先通过该 gate。第一阶段 `verify_phase1.py` 仍可原样运行。

首次 GPU 冒烟尝试在加载模型前被来源检查拒绝：训练入口在初始化审计后增加了重置 RNG 的操作，代码 hash 已变化。已保留 `gpu_smoke_attempt1_stale_gate.log`，重新执行初始化审计后再运行 GPU 检查；没有忽略 gate 或进入训练后补改记录。

## 检查结果与运行边界

- 八项测试通过；9 个真实预训练初始化通过；A/B/C GPU 冒烟均通过。
- A 总参数 **282,528,790**；B/C 各 **297,125,356**。三个 seed 的 A/B/C 共有参数 hash 相同，B/C 全部权重 hash 相同。
- RTX 4090 上该小样本检查的峰值分配显存分别约 9.41 / 10.08 / 10.07 GiB；仅作为实现能运行的证据，不代替论文效率测量。
- Transformers 4.57.6 在本地 mDeBERTa snapshot 上触发 Mistral regex 警告。检查源码和 backend 后确认本模型是 DebertaV2TokenizerFast/Unigram；未应用 Mistral 专用修改。默认分词与显式 `fix_mistral_regex=False` 在八域各前 8 条 train 文本（共 64 条）的 IDs/offsets 完全相同，见 `tokenizer_check.json`。
- 另一个 SentencePiece byte-fallback 警告是实际兼容性边界：fast tokenizer 对部分罕见字符可能产生 UNK，与 slow tokenizer 不完全等价。这次保持原 fast tokenizer 与 offset 流程，未更换分词器；64 条检查不证明所有语言文本都没有该问题。

## 使用入口

环境：`/home/g24tka12/miniforge3/envs/absa/bin/python`。从项目根目录运行：

```bash
python scripts/icsai/verify_phase1.py
python scripts/icsai/verify_control.py

# 一次只执行矩阵中一个完整 run；不接受 epoch/阈值覆盖。
/home/g24tka12/miniforge3/envs/absa/bin/python scripts/icsai/run_control.py train --run-id eng_restaurant_A_s42

# 完成该 run 后恢复其 best EMA checkpoint，仅导出 dev 预测。
/home/g24tka12/miniforge3/envs/absa/bin/python scripts/icsai/run_control.py predict-dev --run-id eng_restaurant_A_s42
```

实际 run 输出仍使用第一阶段预定路径 `checkpoints/icsai_phase1/{run_id}/`，配置文件所在的 phase2 名称不改变实验 ID。

`run_claim.json` 通过排他创建阻止同一 run 被重复启动或覆盖；运行状态另写 `status.json`。若中断，不能删除 claim 后盲目重跑，应检查具体进程、epoch checkpoint 和日志再实施恢复。当前入口不自动恢复、不调度全部 27 次训练。

## 已知边界

- 单个英文 minibatch 的 GPU 检查不能证明全部域/全部长度都不会 OOM，也不能作为完整训练耗时或论文效率数据。
- C 的总分配参数量与 B 相同，但 BIO embedding 的梯度为零；报告的单 minibatch 非零梯度参数量依赖数据，不等于跨数据集固定的“有效模型容量”。
- 同 seed 的 dropout 轨迹不要求 A 与 B 完全相同；配对条件是共有初始参数和数据排列相同，B/C 除干预外执行相同网络。
- 正式核心训练尚需单独启动和监控。此实现阶段不声称核心 27 次已完成，也不产生新的 test 性能证据。
