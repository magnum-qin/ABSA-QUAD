# Tatar matched-text teacher control

Same 1,240 gold rows and same 3,915 silver texts/order; only teacher labels differ.

| Teacher | Five-seed mean±SD | vote=3 ensemble cF1 |
|---|---:|---:|
| DeepSeek | 0.4708±0.0065 | 0.5365 |
| Qwen | 0.4637±0.0147 | 0.5327 |

DeepSeek−Qwen ensemble Δ=+0.0038, 95% CI [-0.0823, +0.0852], inconclusive.

Paired bootstrap: 10,000 resamples, seed=20260713.
