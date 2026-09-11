# ICSAI 八域 A 单阶段模型 test 结果

Share with caveats: post-test-exposure fixed supplementary evaluation; no tuning or model selection on test.

| 数据集 | 三 seed cF1 均值±SD | 固定三 seed 集成 cF1 |
|---|---:|---:|
| eng_laptop | 0.2460 ± 0.0117 | 0.2667 |
| eng_restaurant | 0.5321 ± 0.0028 | 0.5575 |
| jpn_hotel | 0.2362 ± 0.0025 | 0.2514 |
| rus_restaurant | 0.3997 ± 0.0124 | 0.4326 |
| tat_restaurant | 0.3332 ± 0.0171 | 0.3682 |
| ukr_restaurant | 0.3713 ± 0.0377 | 0.4252 |
| zho_laptop | 0.3492 ± 0.0056 | 0.3794 |
| zho_restaurant | 0.3833 ± 0.0050 | 0.4091 |

八域等权宏平均：单模型 0.3564；集成 0.3862。仅作描述性汇总。

历史 test 已暴露，本轮不是盲测。保留原本地 dev 选定的 checkpoint。扩展五域仅评估 A，不能据此扩展三域 Rethink 消融结论。集成有额外推理成本；单次资源记录不能替代重复基准。

逐 seed、离散 F1、匹配 V/A MAE、匹配覆盖率与数据来源见 eight_domain_summary.json；本轮 dev 复核见 dev_statistics.json。
