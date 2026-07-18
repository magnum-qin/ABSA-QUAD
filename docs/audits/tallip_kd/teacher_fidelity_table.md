# TALLIP teacher fidelity (gold dev, labels hidden from teachers)

| Dataset | DeepSeek cF1 | Qwen cF1 | DS-Qwen | 95% CI | Decision |
|---|---:|---:|---:|---:|---|
| eng_laptop | 0.3575 | 0.3170 | +0.0404 | [-0.0066, +0.0879] | inconclusive |
| eng_restaurant | 0.5529 | 0.3638 | +0.1892 | [+0.1418, +0.2384] | supported |
| jpn_hotel | 0.2146 | 0.2004 | +0.0142 | [-0.0344, +0.0610] | inconclusive |
| rus_restaurant | 0.3904 | 0.3649 | +0.0255 | [-0.0494, +0.1039] | inconclusive |
| tat_restaurant | 0.4359 | 0.3954 | +0.0405 | [-0.0675, +0.1516] | inconclusive |
| ukr_restaurant | 0.3786 | 0.3598 | +0.0188 | [-0.0579, +0.0879] | inconclusive |
| zho_laptop | 0.2821 | 0.2374 | +0.0447 | [+0.0137, +0.0755] | supported |
| zho_restaurant | 0.4653 | 0.4060 | +0.0593 | [+0.0240, +0.0941] | supported |

Paired bootstrap: 10,000 resamples, seed=20260713.
