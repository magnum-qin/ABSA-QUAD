# TALLIP clean-inductive KD results

| Dataset | System | Seed mean±sd | Ensemble cF1 | vs baseline Δ [95% CI] | Decision |
|---|---|---:|---:|---:|---|
| eng_laptop | clean | 0.3785±0.0107 | 0.4102 | +0.1058 [+0.0601, +0.1537] | supported |
| rus_restaurant | original | 0.5553±0.0099 | 0.6162 | +0.0895 [+0.0191, +0.1631] | supported |
| tat_restaurant | original | 0.4637±0.0147 | 0.5327 | +0.0506 [-0.0302, +0.1353] | inconclusive |
| ukr_restaurant | clean | 0.4851±0.0355 | 0.5377 | +0.0472 [-0.0466, +0.1449] | inconclusive |
| zho_restaurant | clean | 0.5400±0.0026 | 0.5738 | +0.0069 [-0.0204, +0.0349] | inconclusive |

Paired bootstrap: 10,000 resamples, seed=20260713.
