# TALLIP KD core evidence tables

Generated from frozen machine-readable artifacts. Tables 1–4 use development-set evidence; Table 5 is a one-shot post-competition evaluation on released official test labels.

## Table 1. Direct teacher fidelity on label-hidden gold dev

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

## Table 2. Clean KD five-seed and ensemble results

| Dataset | Clean/original | Seed scores | Mean±SD | Ensemble cF1 |
|---|---|---|---:|---:|
| eng_laptop | clean | 42:0.3826, 123:0.3610, 456:0.3900, 789:0.3799, 1024:0.3789 | 0.3785±0.0107 | 0.4102 |
| rus_restaurant | original | 42:0.5714, 123:0.5471, 456:0.5487, 789:0.5578, 1024:0.5513 | 0.5553±0.0099 | 0.6162 |
| tat_restaurant | original | 42:0.4798, 123:0.4538, 456:0.4681, 789:0.4733, 1024:0.4437 | 0.4637±0.0147 | 0.5327 |
| ukr_restaurant | clean | 42:0.4448, 123:0.4835, 456:0.4601, 789:0.5016, 1024:0.5353 | 0.4851±0.0355 | 0.5377 |
| zho_restaurant | clean | 42:0.5391, 123:0.5417, 456:0.5427, 789:0.5405, 1024:0.5361 | 0.5400±0.0026 | 0.5738 |

## Table 3. Paired clean comparisons

| Dataset | Contrast | ΔcF1 | 95% CI | Decision |
|---|---|---:|---:|---|
| eng_laptop | clean_minus_baseline | +0.1058 | [+0.0601, +0.1537] | supported |
| eng_laptop | clean_minus_transductive | -0.0038 | [-0.0444, +0.0351] | inconclusive |
| rus_restaurant | original_minus_baseline | +0.0895 | [+0.0191, +0.1631] | supported |
| tat_restaurant | original_minus_baseline | +0.0506 | [-0.0302, +0.1353] | inconclusive |
| ukr_restaurant | clean_minus_baseline | +0.0472 | [-0.0466, +0.1449] | inconclusive |
| ukr_restaurant | clean_minus_transductive | -0.0183 | [-0.0738, +0.0292] | inconclusive |
| zho_restaurant | clean_minus_baseline | +0.0069 | [-0.0204, +0.0349] | inconclusive |
| zho_restaurant | clean_minus_transductive | +0.0159 | [-0.0075, +0.0398] | inconclusive |

## Table 4. Ablation claims and evidence grade

| Axis | Controlled evidence | Result | Grade / allowed claim |
|---|---|---|---|
| Teacher | 8 datasets × 2 teachers on identical label-hidden gold dev | DeepSeek cF1 is higher in 8/8; supported on eng_restaurant, zho_laptop, zho_restaurant | High for overall robustness; no evidence of a reliable ranking reversal |
| Teacher downstream | tat, same 1,240 gold + same 3,915 silver texts/order, five paired seeds | DeepSeek 0.5365 vs Qwen 0.5327; Δ=+0.0038, CI [−0.0823,+0.0852] | High matched-control quality; ranking inconclusive |
| Pool source | rus real/trans/generated, same teacher and cap=3000 | generated > translated > real | High within rus; no universal source ranking |
| Pool source | zho real vs generated | real +2.3pp | Medium; observational support only |
| Silver quantity | rus same gen8k pool, paired seeds 42/123, cap3000 vs 7988 | −1.14pp; gate failed | Medium; rejects the planned follow-up, not a full dose-response |
| Translation vs native | ukr translated and native experiments differ in pool construction/quantity | mixed | Low; descriptive only |
| Information complementarity | synthesis across teacher, pool, and quantity results | plausible moderator | Interpretation, not a causal or universal optimum claim |

## Table 5. One-shot official Task-3 test evaluation

Post-competition local evaluation. Systems and prediction hashes were frozen from dev before gold retrieval.

| Dataset | Frozen main system | Official test cF1 |
|---|---|---:|
| eng_laptop | eng_laptop_clean_kd | 0.3532 |
| eng_restaurant | eng_restaurant_v16 | 0.5904 |
| jpn_hotel | jpn_hotel_v16_no_aug | 0.4217 |
| rus_restaurant | rus_restaurant_gen4k_kd | 0.5189 |
| tat_restaurant | tat_restaurant_qwen_kd | 0.4311 |
| ukr_restaurant | ukr_restaurant_clean_kd | 0.4608 |
| zho_laptop | zho_laptop_v16 | 0.4309 |
| zho_restaurant | zho_restaurant_clean_kd | 0.4660 |

| Dataset | Pre-registered contrast | ΔcF1 | 95% CI | Decision |
|---|---|---:|---:|---|
| eng_laptop | eng_laptop_clean_kd − eng_laptop_clean_baseline | +0.0822 | [+0.0629, +0.1021] | supported |
| rus_restaurant | rus_restaurant_gen4k_kd − rus_restaurant_v15_baseline | +0.0211 | [+0.0005, +0.0431] | supported |
| tat_restaurant | tat_restaurant_qwen_kd − tat_restaurant_v15_baseline | +0.0274 | [+0.0071, +0.0482] | supported |
| ukr_restaurant | ukr_restaurant_clean_kd − ukr_restaurant_clean_baseline | +0.0393 | [+0.0150, +0.0639] | supported |
| zho_restaurant | zho_restaurant_clean_kd − zho_restaurant_v16_baseline | -0.0137 | [-0.0289, +0.0014] | inconclusive |
| tat_restaurant | tat_restaurant_deepseek_matched − tat_restaurant_qwen_kd | +0.0109 | [-0.0069, +0.0293] | inconclusive |
