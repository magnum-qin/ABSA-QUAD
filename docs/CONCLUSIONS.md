# TALLIP KD 最终结论口径

更新日期：2026-07-18。

本文档是 KD 结论的单一权威入口。历史实验日志可以保留当时的假设和
dev 结果，但论文、README 和汇报不得用 dev 分数与官方 test leaderboard
作横向比较，也不得把观察性结果写成普适因果规律。

## 证据层级

1. released official test 上的冻结系统与 paired bootstrap；
2. clean-inductive dev 上的五种子结果与 paired bootstrap；
3. 8×2 teacher fidelity 和 Tatar matched downstream control；
4. teacher、pool 与 quantity 的受控消融；
5. 早期 transductive 或跨配置 dev 对比，仅作历史/描述性证据。

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

## 允许保留的六条结论

1. **KD 的收益具有条件性。** 当前结果与“silver 提供 gold/base 缺失的、
   且标注较准确的覆盖时更可能有效”一致；信息互补性是解释性机制，尚未被
   独立操纵为普适因果定律。
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

## 已撤回或禁止的表述

- “dev 反超 test SOTA”或任何由跨 split 数字得到的 SOTA 结论；
- “低资源数据必然从 KD 受益”或“KD 增益与 gold 规模成反比”；
- “真实池普遍优于生成池”或相反的固定来源排序；
- “silver_cap=3000 是黄金甜区”或“silver 越少/越多越好”；
- “Tatar 上 Qwen 显著优于 DeepSeek”；
- “zho_restaurant 是 KD 成功案例”：official test 的 paired 证据不确定。

数值、置信区间和协议细节见
[`docs/audits/tallip_kd/tallip_core_tables.md`](audits/tallip_kd/tallip_core_tables.md)
与
[`docs/audits/tallip_kd/official_test/official_test_table.md`](audits/tallip_kd/official_test/official_test_table.md)。
五张核心表已由
[`tallip_core_tables.lock.json`](audits/tallip_kd/tallip_core_tables.lock.json)
做 SHA256 冻结；论文引用前必须通过
`python scripts/kd/build_tallip_report.py --verify`。
