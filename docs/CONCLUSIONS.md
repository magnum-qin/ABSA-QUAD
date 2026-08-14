# TALLIP KD 最终结论口径

更新日期：2026-08-14。

本文档是 KD 结论的单一权威入口。历史实验日志可以保留当时的假设和
dev 结果，但论文、README 和汇报不得用 dev 分数与官方 test leaderboard
作横向比较，也不得把观察性结果写成普适因果规律。

TALLIP 的 novelty 与 ICONIP V20 的边界见
[`TALLIP_VS_ICONIP_V20.md`](TALLIP_VS_ICONIP_V20.md)：V20 已拥有学生架构、
language-specific recipe 和多种子推理；TALLIP 只把离线 LLM silver KD、
teacher/pool/quantity 边界、clean-inductive 与统计/test 闭环作为新增贡献。

## 证据层级

1. released official test 上的冻结系统与 paired bootstrap；
2. clean-inductive dev 上的五种子结果与 paired bootstrap；
3. post-lock dev-only 的 exposure-matched repeated-gold 与 same-text
   self-training 五种子对照；
4. 8×2 teacher fidelity 和 Tatar matched downstream control；
5. teacher、pool 与 quantity 的受控消融；
6. 早期 transductive 或跨配置 dev 对比，仅作历史/描述性证据。

official test 是赛后一次性本地评估：系统、预测哈希和阈值均在读取 gold
前冻结，未用 test 标签选择模型、teacher、pool、数据增强或阈值。

## 最终结果

| 数据集 | 冻结主系统 | official test cF1 |
|---|---|---:|
| eng_laptop | clean DeepSeek KD | 0.3532 |
| eng_restaurant | V16 baseline | 0.5904 |
| jpn_hotel | V16 no-augmentation | 0.4217 |
| rus_restaurant | DeepSeek gen4k KD | 0.5189 |
| tat_restaurant | Qwen KD | 0.4311 |
| ukr_restaurant | clean DeepSeek KD | 0.4608 |
| zho_laptop | V16 baseline | 0.4309 |
| zho_restaurant | clean DeepSeek-real KD | 0.4660 |

五个预注册 KD 对照中，eng_laptop、rus_restaurant、tat_restaurant 和
ukr_restaurant 的 KD−baseline 95% CI 下界大于 0；zho_restaurant 的点估计
为 −0.0137，95% CI [−0.0289, +0.0014]，证据不确定。因此允许写
“official test 在四个域支持 KD 的条件性收益”，不能写“5/8 KD 成功”。

八个主系统均未超过相应官方 leaderboard 最佳分数，不能声明 test SOTA。

## Post-lock 控制实验

为检验 LLM-KD 的 dev 收益是否只是更多训练行/更新步数造成，五个 adopted
KD 数据集均按目标训练行数构造 repeated-gold，并沿用相同学生配置、五个
seeds 和冻结阈值。LLM-KD−repeated-gold 在 eng_laptop、rus_restaurant、
ukr_restaurant 的 95% CI 下界大于 0；tat_restaurant 与 zho_restaurant 的
CI 跨 0。因此，当前证据支持“这三个域的收益不能仅由训练曝光解释”，不支持
把该结论外推到所有域。

五个 adopted 数据集均在与 LLM silver 完全相同的文本和行数上构造了学生
ensemble 伪标签。LLM-KD 相对 self-training 在 eng 和 rus 分别为 +0.0933
[+0.0476, +0.1407]、+0.1183 [+0.0531, +0.1854]，得到统计支持；tat、ukr、
zho 分别为 +0.0569 [−0.0203, +0.1402]、+0.0438
[−0.0231, +0.1069]、−0.0016 [−0.0294, +0.0256]，均为 inconclusive。
zho 的 self-training 又可靠优于 repeated-gold +0.0276
[+0.0079, +0.0463]；tat/ukr 的两个非 LLM control 之间也无可靠差异。
五域同文本对照因此将结论收敛为：只有 eng/rus 建立了 LLM teacher 的额外
标签价值，tat/ukr/zho 没有；不能把 teacher-source 优势外推为跨域规律。

分族多重比较校正进一步收紧表述。same-text family 的 eng/rus 在 Holm 与 BH
校正后仍显著，tat/ukr/zho 均不显著。Exposure family 中 eng/rus 保持 Holm
显著；ukr 的单项 95% CI 下界虽大于 0，但 Holm p=0.1542，因此只能说单项
分析支持，不能说在五域 family-wise 校正后仍得到支持。一次性 official-test
family 中 eng/tat/ukr 保持 Holm 显著，rus 不保持，zho 仍不确定；该分析是
post-hoc robustness check，绝不改变任何冻结 test 选择。

五域直接描述性分析还测量了 silver 标签有效性、类别分布 JSD、dev
aspect/opinion 类型覆盖、row-level repaired/harmed 和 error slices。eng 的
span 覆盖增量与 KD dev 增益均最大；zho 的直接 teacher fidelity 最高，却是
KD dev 增益最小且 span 覆盖下降。这些非单调模式支持“质量、覆盖、互补性
共同作用”的解释，但不能证明单一指标或普适因果机制。

固定五个 adopted 数据集并给予每个数据集相同权重的 macro bootstrap 进一步
给出整体效应：KD−baseline 为 +0.0600 [0.0291, 0.0911]，KD−repeated-gold
为 +0.0588 [0.0318, 0.0853]，KD−same-text self-training 为 +0.0621
[0.0351, 0.0888]，三者的 macro CI 均高于 0；self-training−repeated-gold
为 −0.0033 [−0.0226, 0.0170]，证据不确定。这支持“在预注册五域集合上的
平均 KD 收益”，但不表示五个域分别都显著，也不能外推到未观测语言或任务。

五域探索性 Spearman 分析显示，teacher fidelity 与 KD dev 增益 ρ=−0.70，
95% bootstrap CI [−1.00, 0.875]；span 覆盖增量与 KD 增益 ρ=+0.70，CI
[−0.875, 1.00]；category alignment 为 ρ=+0.10，CI [−1.00, 1.00]。
由于仅 n=5 且全部 CI 很宽，这些结果不能建立 moderator，只进一步说明不能
用 direct fidelity 或任何单一覆盖指标预测下游收益。句长、单/多四元组切片
同样呈跨域异质性，不支持统一的“长句更受益”等规律。

## 允许保留的六条结论

1. **KD 的收益具有条件性，且不能统一归因于训练曝光。** 单项 post-lock
   控制在 eng/rus/ukr 排除了“仅增加训练行数”这一替代解释，但 tat/zho
   证据不确定；五域 Holm 校正后只有 eng/rus 保持显著。
   结果与“silver 提供 gold/base 缺失且标注较准确的覆盖时更可能有效”一致。
   信息互补性仍是解释性机制，不是已独立操纵的普适因果定律。
2. **DeepSeek 整体更稳健，Qwen 在部分域接近。** 直接 teacher fidelity 的
   点估计为 DeepSeek 8/8 更高，但只有 eng_restaurant、zho_laptop 和
   zho_restaurant 的差异得到统计支持，其余五个域不确定。
3. **Tatar 不存在可靠的 teacher 排名反转。** matched dev 和 official test
   都是 DeepSeek 点估计略高但 CI 跨 0；学生最终分数的微小波动不能单独代表
   teacher 标注质量。
4. **pool 来源没有普适排序。** zho_restaurant 的描述性结果为 real 更好，
   rus 的 size-matched 对照为 generated > translated > real；跨数据集差异还
   同时受域匹配、teacher 标注质量和训练配方影响。
5. **rus gen4k 的优势不能归因于 silver 更少。** 同一 gen8k silver 上，
   cap3000 的两 seed paired mean 比 full 低 1.14pp，未通过预注册继续门槛；
   独立 gen4k 优势更符合生成批次质量/覆盖差异。
6. **没有普适的 silver 数量最优点。** 最佳 silver 取决于质量、覆盖、
   多样性和 gold:silver 权重；现有证据不支持统一 cap、甜区或 dose-response。

上述结果还支持一个边界条件：**LLM teacher 的额外价值具有域依赖性。**
eng/rus 的 LLM-KD 显著优于 same-text self-training，并在五域 Holm 校正后
保持；tat/ukr/zho 均未建立可靠优势。不得把“LLM 标签必然优于学生伪标签”
写成跨域规律。

## 已撤回或禁止的表述

- “dev 反超 test SOTA”或任何由跨 split 数字得到的 SOTA 结论；
- “低资源数据必然从 KD 受益”或“KD 增益与 gold 规模成反比”；
- “真实池普遍优于生成池”或相反的固定来源排序；
- “silver_cap=3000 是黄金甜区”或“silver 越少/越多越好”；
- “Tatar 上 Qwen 显著优于 DeepSeek”；
- “zho_restaurant 是 KD 成功案例”：official test 的 paired 证据不确定。
- “LLM teacher 在所有域都优于 self-training”：tat/ukr/zho 的 matched-text
  对照均未建立可靠优势。
- 将 Rethink、Hungarian matching、language-specific encoder 或 five-seed
  ensemble 再次列为 TALLIP 新贡献；这些属于 ICONIP V20 的既有工作。

数值、置信区间和协议细节见
[`docs/audits/tallip_kd/tallip_core_tables.md`](audits/tallip_kd/tallip_core_tables.md)
与
[`docs/audits/tallip_kd/official_test/official_test_table.md`](audits/tallip_kd/official_test/official_test_table.md)。
六张核心表及五域 post-lock 补充分析已由
[`tallip_core_tables.lock.json`](audits/tallip_kd/tallip_core_tables.lock.json)
做 SHA256 冻结；论文引用前必须通过
`python scripts/kd/build_tallip_report.py --verify`。
