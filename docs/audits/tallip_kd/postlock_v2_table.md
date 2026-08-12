# Post-lock dev-only control results

All comparisons use the same gold dev examples, fixed pre-test thresholds, 10,000 paired bootstrap resamples, and seed 20260713. Official-test labels are forbidden.

| Dataset | Control | Five-seed mean±SD | Ensemble cF1 | LLM KD − control [95% CI] | Decision | Evidence grade / allowed claim |
|---|---|---:|---:|---:|---|---|
| tat_restaurant | gold_repeat | 0.4581±0.0181 | 0.5256 | +0.0071 [-0.0621, +0.0754] | inconclusive | High: no reliable LLM advantage over exposure |
| tat_restaurant | self_training | 0.4268±0.0098 | 0.4758 | +0.0569 [-0.0203, +0.1402] | inconclusive | High: no reliable LLM advantage over student pseudo-labels |
| ukr_restaurant | gold_repeat | 0.4122±0.0187 | 0.4641 | +0.0736 [+0.0036, +0.1507] | supported | High: LLM benefit exceeds exposure |
| ukr_restaurant | self_training | 0.4525±0.0198 | 0.4940 | +0.0438 [-0.0231, +0.1069] | inconclusive | High: no reliable LLM advantage over student pseudo-labels |

## Self-training versus exposure-matched repeated gold

| Dataset | Self-training − repeated gold [95% CI] | Decision | Allowed claim |
|---|---:|---|---|
| tat_restaurant | -0.0498 [-0.1085, +0.0143] | inconclusive | no reliable difference between the two non-LLM controls |
| ukr_restaurant | +0.0298 [-0.0245, +0.0867] | inconclusive | no reliable difference between the two non-LLM controls |
