# ICSAI 固定核心 test 对照

Share with caveats: post-test-exposure supplementary evaluation; fixed trained seeds; three domains only.

全部27个单模型与9个固定投票集成完成，逐份官方评分交叉核验通过。

| 数据集 | A cF1±SD | B cF1±SD | C cF1±SD |
|---|---:|---:|---:|
| eng_restaurant | 0.5321 ± 0.0028 | 0.5380 ± 0.0033 | 0.5324 ± 0.0156 |
| zho_restaurant | 0.3833 ± 0.0050 | 0.3737 ± 0.0037 | 0.3715 ± 0.0092 |
| tat_restaurant | 0.3332 ± 0.0171 | 0.3169 ± 0.0228 | 0.3285 ± 0.0067 |

| 数据集 | 对照 | 差值（百分点） | 未校正95% CI | Holm p |
|---|---|---:|---:|---:|
| eng_restaurant | B-A | +0.59 | [-0.12, +1.29] | 0.2014 |
| eng_restaurant | B-C | +0.56 | [-0.01, +1.13] | 0.1659 |
| zho_restaurant | B-A | -0.96 | [-1.71, -0.21] | 0.0730 |
| zho_restaurant | B-C | +0.22 | [-0.35, +0.80] | 0.4418 |
| tat_restaurant | B-A | -1.62 | [-2.72, -0.46] | 0.0222 |
| tat_restaurant | B-C | -1.16 | [-2.11, -0.19] | 0.0816 |

CI 仅表示在固定三个训练 seed 下的样本重采样不确定性，不覆盖整个训练过程；不显著不等于等效。历史 test 已暴露，不能称为盲测。三域结果不能外推到八域。

逐 seed、集成、离散抽取 F1、匹配 V/A MAE 与覆盖率、显式/隐式样本计数和单次推理资源记录见 [test_statistics.json](test_statistics.json)。计时包含首批，尚不能代替重复性能基准。
