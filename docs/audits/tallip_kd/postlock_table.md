# Post-lock dev-only control results

All comparisons use the same gold dev examples, fixed pre-test thresholds, 10,000 paired bootstrap resamples, and seed 20260713. Official-test labels are forbidden.

| Dataset | Control | Five-seed mean±SD | Ensemble cF1 | LLM KD − control [95% CI] | Decision | Evidence grade / allowed claim |
|---|---|---:|---:|---:|---|---|
| eng_laptop | gold_repeat | 0.2971±0.0055 | 0.3154 | +0.0948 [+0.0494, +0.1426] | supported | High: LLM benefit exceeds exposure |
| eng_laptop | self_training | 0.2973±0.0067 | 0.3169 | +0.0933 [+0.0476, +0.1407] | supported | High: LLM teacher adds value over student pseudo-labels |
| rus_restaurant | gold_repeat | 0.4897±0.0099 | 0.5238 | +0.0924 [+0.0247, +0.1634] | supported | High: LLM benefit exceeds exposure |
| rus_restaurant | self_training | 0.4767±0.0171 | 0.4980 | +0.1183 [+0.0531, +0.1854] | supported | High: LLM teacher adds value over student pseudo-labels |
| tat_restaurant | gold_repeat | 0.4581±0.0181 | 0.5256 | +0.0071 [-0.0621, +0.0754] | inconclusive | High: no reliable LLM advantage over exposure |
| ukr_restaurant | gold_repeat | 0.4122±0.0187 | 0.4641 | +0.0736 [+0.0036, +0.1507] | supported | High: LLM benefit exceeds exposure |
| zho_restaurant | gold_repeat | 0.5084±0.0042 | 0.5478 | +0.0260 [-0.0028, +0.0543] | inconclusive | High: no reliable LLM advantage over exposure |
| zho_restaurant | self_training | 0.5505±0.0096 | 0.5754 | -0.0016 [-0.0294, +0.0256] | inconclusive | High: no reliable LLM advantage over student pseudo-labels |

## Self-training versus exposure-matched repeated gold

| Dataset | Self-training − repeated gold [95% CI] | Decision | Allowed claim |
|---|---:|---|---|
| eng_laptop | +0.0015 [-0.0174, +0.0215] | inconclusive | no reliable difference between the two non-LLM controls |
| rus_restaurant | -0.0258 [-0.0773, +0.0229] | inconclusive | no reliable difference between the two non-LLM controls |
| zho_restaurant | +0.0276 [+0.0079, +0.0463] | supported | student pseudo-labels improve over repeated exposure |
