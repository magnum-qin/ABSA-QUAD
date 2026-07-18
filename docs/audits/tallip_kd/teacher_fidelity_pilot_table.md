# Paired 8x10 teacher-fidelity pilot

Secondary evidence: only the first 10 gold-dev examples per dataset are compared.

| Dataset | DeepSeek cF1 | Qwen cF1 | DS-Qwen | 95% CI | Decision |
|---|---:|---:|---:|---:|---|
| eng_laptop | 0.3534 | 0.3170 | +0.0364 | [-0.1465, +0.1886] | inconclusive |
| eng_restaurant | 0.5684 | 0.4025 | +0.1660 | [+0.0346, +0.3486] | supported |
| jpn_hotel | 0.1326 | 0.0461 | +0.0865 | [+0.0000, +0.2134] | inconclusive |
| rus_restaurant | 0.4901 | 0.3996 | +0.0905 | [-0.0446, +0.2201] | inconclusive |
| tat_restaurant | 0.4304 | 0.3963 | +0.0341 | [-0.3282, +0.2999] | inconclusive |
| ukr_restaurant | 0.3281 | 0.4098 | -0.0818 | [-0.2941, +0.0872] | inconclusive |
| zho_laptop | 0.2670 | 0.1775 | +0.0895 | [-0.0778, +0.2480] | inconclusive |
| zho_restaurant | 0.2958 | 0.3696 | -0.0738 | [-0.2477, +0.1170] | inconclusive |

Paired bootstrap: 10,000 resamples, seed=20260713; n=10 per dataset.
