# Official Task-3 test results

Post-competition evaluation on the released official test labels; all system choices and prediction hashes were frozen before gold retrieval.

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

| Dataset | Comparison | ΔcF1 [95% CI] | Decision |
|---|---|---:|---|
| eng_laptop | eng_laptop_clean_kd − eng_laptop_clean_baseline | +0.0822 [+0.0629, +0.1021] | supported |
| rus_restaurant | rus_restaurant_gen4k_kd − rus_restaurant_v15_baseline | +0.0211 [+0.0005, +0.0431] | supported |
| tat_restaurant | tat_restaurant_qwen_kd − tat_restaurant_v15_baseline | +0.0274 [+0.0071, +0.0482] | supported |
| ukr_restaurant | ukr_restaurant_clean_kd − ukr_restaurant_clean_baseline | +0.0393 [+0.0150, +0.0639] | supported |
| zho_restaurant | zho_restaurant_clean_kd − zho_restaurant_v16_baseline | -0.0137 [-0.0289, +0.0014] | inconclusive |
| tat_restaurant | tat_restaurant_deepseek_matched − tat_restaurant_qwen_kd | +0.0109 [-0.0069, +0.0293] | inconclusive |

Paired bootstrap: 10,000 resamples, seed=20260713.
