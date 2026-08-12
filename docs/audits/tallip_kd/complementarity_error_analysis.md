# Five-domain information complementarity and error analysis

Descriptive dev-only measurements; they do not establish a universal causal moderator.

| Dataset | Teacher fidelity | Silver rows | Dev span coverage Δ A/O | Category JSD Δ silver−base | Repaired / harmed rows | All-dev KD ΔcF1 |
|---|---:|---:|---:|---:|---:|---:|
| eng_laptop | 0.3575 | 7547 | +21 / +38 | -0.0141 | 71 / 44 | +0.1058 |
| rus_restaurant | 0.3904 | 3000 | -4 / +0 | +0.0503 | 20 / 19 | +0.0895 |
| tat_restaurant | 0.3954 | 3915 | -9 / +4 | +0.0571 | 16 / 20 | +0.0506 |
| ukr_restaurant | 0.3786 | 5534 | +4 / +9 | +0.0241 | 22 / 18 | +0.0472 |
| zho_restaurant | 0.4653 | 3882 | -30 / -56 | -0.0009 | 130 / 112 | +0.0069 |

Largest KD dev gain: eng_laptop; largest combined dev span-coverage gain: eng_laptop; highest direct teacher fidelity: zho_restaurant.

The measured fidelity, distribution, and lexical-coverage descriptors are non-monotonic across five domains. They support a multi-factor quality/coverage/complementarity interpretation, not a single causal score or universal ordering.

## Dev error slices

| Dataset | All rows / Δ | Multi-quad rows / Δ | Rare-category rows / Δ | Implicit rows / Δ |
|---|---:|---:|---:|---:|
| eng_laptop | 200 / +0.1058 | 93 / +0.1211 | 8 / +0.1575 | 0 / — |
| rus_restaurant | 48 / +0.0895 | 24 / +0.0900 | 0 / — | 0 / — |
| tat_restaurant | 48 / +0.0506 | 24 / +0.0519 | 0 / — | 0 / — |
| ukr_restaurant | 48 / +0.0472 | 24 / +0.0956 | 0 / — | 0 / — |
| zho_restaurant | 300 / +0.0069 | 202 / +0.0132 | 1 / +0.0000 | 0 / — |
