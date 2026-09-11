# Rethink 实际路径审计

日期：2026-09-07。只核对源码和 CPU 合成输入，不训练、不下载预训练权重。

## 直接结论

当前实现是 **BIO soft embedding 反馈的两轮 token refinement**。pair validity、entity、attribute、VA 虽然被嵌入并融合为 `fused_pair`，但调用者通过 `refined_hidden, _ = ...` 丢弃第二个返回值；没有进入第二阶段预测。这与原 tex 所写“BIO、category、VA 全部反馈到 hidden states”不一致。

## 静态证据

- `src/models/feature_rethink.py:225–254`：BIO → token_fusion → self-attention → FFN → gated residual → refined_hidden；pair labels → pair_fusion → fused_pair，为相互独立的返回路径。
- `src/models/dimabsa_model.py:496–567`：`range(self.num_rethink_iterations)` 实际循环，stage-2 重新抽 span，用 refined_hidden 的 span pooling 建新 pair；不是使用 fused_pair。两轮共享同一组 stage-2 heads。
- `src/models/dimabsa_model.py:817–858`：consistency 仅比较 BIO loss，stage-1 loss detach；不是四元组/VA 一致性损失。最终 pair/category/VA 监督作用于最终输出，不能据此声称 stage-1 pair/VA 被反馈训练。
- `src/models/dimabsa_model.py:178–231`：PairClassifier 仅拼接 aspect 与 opinion representation；原 tex 的“逐元素乘积与距离特征”也未见有效实现。
- `src/models/dimabsa_model.py:585–661`：Hungarian cost 是 `2*(2-IoUasp-IoUopn) + entity_NLL + attribute_NLL + 0.5*((abs(Verr)+abs(Aerr))/8)`，没有 validity 或 consistency 项；VA cost 是归一化 L1，不是 Huber。训练 VA loss 用 Huber，不应混写匹配 cost 与训练 loss。
- `scripts/inference.py:266–294`：集成以未经归一化的精确 `(Aspect,Category,Opinion)` 字符串为 key，达到 vote threshold 后对该 key 的 VA 算术平均并裁剪。不同类别各自成 key，无互斥仲裁；无独立平局选择器。循环按输出 quad 累加，未在此处确保一个 seed 每 key 最多一票；运行前应验证单模型输出是否有重复。缺席 seed 不参与 VA 平均。
- 当前模型文件与仓库 ICONIP 参考稿提交 `beb1670` 中相应模型文件 `git diff` 为空；说明该参考提交已有相同路径。**不能仅凭此证明所有历史 checkpoint 都是由该代码产生**。
- `README_AI.md` 关于“代码只跑一轮”的旧 TODO 已过时；当前确实按配置循环两轮。本次没有修改历史 README 或原稿，避免将审计变成无痕改写。

## 动态干预与梯度证据

运行：`/home/g24tka12/miniforge3/envs/absa/bin/python scripts/icsai/probe_rethink_path.py`。

使用真实 DimASQPModel/FeatureLevelRethink，替换为 16 维合成 encoder；固定一个非空 aspect/opinion pair；eval 模式关闭 dropout。输出六类最终张量全部比较。

| 干预 | 最大绝对输出变化 |
|---|---:|
| 改写全部 pair/类别/VA 反馈输入 | 0 |
| 把 fused_pair 返回值替换为大常数 | 0 |
| 改写 BIO soft probabilities | 0.3217778802 |

forward hook 观测两次 Rethink 调用。对最终输出构造平方和并反向传播，BIO embedding/token_fusion 有梯度，pair/entity/attribute embedding、VA projector、pair_fusion 的梯度均为 None。

完整数值、版本与源码 hash：[rethink_probe.json](rethink_probe.json)。该实验验证计算依赖，不测性能；固定 span 和合成 encoder 的范围已说明，训练 loss 的结论另由静态源码支持。

## 后续稿件允许的表述

“Draft BIO probabilities are embedded and fused into token states, which are refined with self-attention, a feed-forward block and gated residual fusion. Shared second-stage heads re-extract spans and predict categories and valence–arousal over two refinement iterations.”

不再声称 full-quad/VA-aware feedback 已生效，不把 Hungarian 算法本身称为新算法，不以未计时数据声称 lightweight 的成本优势。下一阶段主实验保持此 BIO-only 实现，另用 zero-BIO 控制隔离预测反馈；接入 pair/VA 属于新架构，排除在本轮冻结设计之外。
