# TALLIP KD core evidence tables

**Status: frozen on 2026-08-14.** Tables 1–5 use development-set evidence; Table 6 is a one-shot post-competition evaluation on released official test labels. Run `python scripts/kd/build_tallip_report.py --verify` before citation.

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

## Table 2. Inductive-safe KD five-seed and ensemble results

| Dataset | Training input | Seed scores | Mean±SD | Ensemble cF1 |
|---|---|---|---:|---:|
| eng_laptop | cleaned | 42:0.3826, 123:0.3610, 456:0.3900, 789:0.3799, 1024:0.3789 | 0.3785±0.0107 | 0.4102 |
| rus_restaurant | original (zero dev/test overlap) | 42:0.5714, 123:0.5471, 456:0.5487, 789:0.5578, 1024:0.5513 | 0.5553±0.0099 | 0.6162 |
| tat_restaurant | original (zero dev/test overlap) | 42:0.4798, 123:0.4538, 456:0.4681, 789:0.4733, 1024:0.4437 | 0.4637±0.0147 | 0.5327 |
| ukr_restaurant | cleaned | 42:0.4448, 123:0.4835, 456:0.4601, 789:0.5016, 1024:0.5353 | 0.4851±0.0355 | 0.5377 |
| zho_restaurant | cleaned | 42:0.5391, 123:0.5417, 456:0.5427, 789:0.5405, 1024:0.5361 | 0.5400±0.0026 | 0.5738 |

## Table 3. Paired dev comparisons under the frozen protocol

| Dataset | Contrast | ΔcF1 | 95% CI | Decision |
|---|---|---:|---:|---|
| eng_laptop | clean KD − baseline | +0.1058 | [+0.0601, +0.1537] | supported |
| eng_laptop | clean KD − transductive KD | -0.0038 | [-0.0444, +0.0351] | inconclusive |
| rus_restaurant | inductive-safe KD − baseline | +0.0895 | [+0.0191, +0.1631] | supported |
| tat_restaurant | inductive-safe KD − baseline | +0.0506 | [-0.0302, +0.1353] | inconclusive |
| ukr_restaurant | clean KD − baseline | +0.0472 | [-0.0466, +0.1449] | inconclusive |
| ukr_restaurant | clean KD − transductive KD | -0.0183 | [-0.0738, +0.0292] | inconclusive |
| zho_restaurant | clean KD − baseline | +0.0069 | [-0.0204, +0.0349] | inconclusive |
| zho_restaurant | clean KD − transductive KD | +0.0159 | [-0.0075, +0.0398] | inconclusive |

## Table 4. Ablation claims and evidence grade

| Axis | Controlled evidence | Result | Grade / allowed claim |
|---|---|---|---|
| Teacher | 8 datasets × 2 teachers on identical label-hidden gold dev | DeepSeek cF1 is higher in 8/8; supported on eng_restaurant, zho_laptop, zho_restaurant | High for overall robustness; no evidence of a reliable ranking reversal |
| Teacher downstream | tat, same 1,240 gold + same 3,915 silver texts/order, five paired seeds | DeepSeek 0.5365 vs Qwen 0.5327; Δ=+0.0038, CI [-0.0823,+0.0852] | High matched-control quality; ranking inconclusive |
| Pool source | rus real/trans/generated, same teacher and cap=3000 | generated (0.6162) > translated (0.5761) > real (0.5586) | High within rus; no universal source ranking |
| Pool source | zho real vs generated | real − generated = +2.3pp | Medium; observational support only |
| Silver quantity | rus same gen8k pool, paired seeds 42/123, cap3000 vs 7988 | -1.14pp; gate failed | Medium; rejects the planned follow-up, not a full dose-response |
| Translation vs native | ukr translated and native experiments differ in pool construction/quantity | mixed | Low; descriptive only |
| Training exposure | five datasets, same row exposure and student recipe; LLM silver replaced by repeated gold | supported on eng_laptop, rus_restaurant, ukr_restaurant; inconclusive on tat_restaurant, zho_restaurant | High within dataset; LLM advantage exceeds exposure on eng/rus/ukr only |
| Teacher source | five adopted datasets, identical pseudo-label text pool and row count; LLM labels vs student-ensemble labels | LLM KD supported on eng_laptop, rus_restaurant; reliable difference not established on tat_restaurant, ukr_restaurant, zho_restaurant | High within dataset; no universal cross-domain teacher-source claim |
| Information complementarity | five adopted domains: measured silver validity, category/lexical coverage, distribution divergence, dev slices, and row-level repaired/harmed errors | descriptive patterns are consistent with a moderator but are not monotonic across all domains | Medium/descriptive; interpretation, not a causal law or universal optimum claim |

## Table 5. Post-lock dev-only controls for exposure and teacher source

These controls use fixed pre-test thresholds and never evaluate or select on official test labels.

| Dataset | Control | Five-seed mean±SD | Ensemble cF1 | LLM KD − control | 95% CI | Evidence / allowed claim |
|---|---|---:|---:|---:|---:|---|
| eng_laptop | gold_repeat | 0.2971±0.0055 | 0.3154 | +0.0948 | [+0.0494, +0.1426] | High; benefit exceeds exposure |
| eng_laptop | self_training | 0.2973±0.0067 | 0.3169 | +0.0933 | [+0.0476, +0.1407] | High; LLM teacher adds value |
| rus_restaurant | gold_repeat | 0.4897±0.0099 | 0.5238 | +0.0924 | [+0.0247, +0.1634] | High; benefit exceeds exposure |
| rus_restaurant | self_training | 0.4767±0.0171 | 0.4980 | +0.1183 | [+0.0531, +0.1854] | High; LLM teacher adds value |
| tat_restaurant | gold_repeat | 0.4581±0.0181 | 0.5256 | +0.0071 | [-0.0621, +0.0754] | High; reliable exposure-adjusted advantage not established |
| tat_restaurant | self_training | 0.4268±0.0098 | 0.4758 | +0.0569 | [-0.0203, +0.1402] | High; reliable teacher-source advantage not established |
| ukr_restaurant | gold_repeat | 0.4122±0.0187 | 0.4641 | +0.0736 | [+0.0036, +0.1507] | High; benefit exceeds exposure |
| ukr_restaurant | self_training | 0.4525±0.0198 | 0.4940 | +0.0438 | [-0.0231, +0.1069] | High; reliable teacher-source advantage not established |
| zho_restaurant | gold_repeat | 0.5084±0.0042 | 0.5478 | +0.0260 | [-0.0028, +0.0543] | High; reliable exposure-adjusted advantage not established |
| zho_restaurant | self_training | 0.5505±0.0096 | 0.5754 | -0.0016 | [-0.0294, +0.0256] | High; reliable teacher-source advantage not established |

### 5b. Student self-training versus repeated-gold exposure

| Dataset | Self-training − repeated gold | 95% CI | Decision |
|---|---:|---:|---|
| eng_laptop | +0.0015 | [-0.0174, +0.0215] | inconclusive |
| rus_restaurant | -0.0258 | [-0.0773, +0.0229] | inconclusive |
| tat_restaurant | -0.0498 | [-0.1085, +0.0143] | inconclusive |
| ukr_restaurant | +0.0298 | [-0.0245, +0.0867] | inconclusive |
| zho_restaurant | +0.0276 | [+0.0079, +0.0463] | supported |

### 5c. Direct complementarity and error measurements

These are descriptive dev-only measurements, not proof of a universal causal moderator.

| Dataset | Silver rows | Dev span coverage Δ A/O | Category JSD Δ silver−base | Repaired / harmed dev rows | KD−baseline cF1 |
|---|---:|---:|---:|---:|---:|
| eng_laptop | 7547 | +21 / +38 | -0.0141 | 71 / 44 | +0.1058 |
| rus_restaurant | 3000 | -4 / +0 | +0.0503 | 20 / 19 | +0.0895 |
| tat_restaurant | 3915 | -9 / +4 | +0.0571 | 16 / 20 | +0.0506 |
| ukr_restaurant | 5534 | +4 / +9 | +0.0241 | 22 / 18 | +0.0472 |
| zho_restaurant | 3882 | -30 / -56 | -0.0009 | 130 / 112 | +0.0069 |

Exploratory five-domain correlations (10,000 dataset-level bootstrap resamples; descriptive only):

| Descriptor vs KD dev gain | Spearman ρ | 95% CI |
|---|---:|---:|
| Teacher fidelity | -0.7000 | [-1.0000, +0.8750] |
| Span-coverage gain | +0.7000 | [-0.8750, +1.0000] |
| Category-alignment gain | +0.1000 | [-1.0000, +1.0000] |

### 5d. Multiplicity robustness

Holm and BH corrections are applied separately within each pre-declared family. The official-test family is post-hoc and cannot alter frozen choices.

| Family | n | Raw CI excludes zero | Holm p<.05 | BH p<.05 |
|---|---:|---:|---:|---:|
| teacher_fidelity_8x2 | 8 | 3 | 3 | 3 |
| adopted_kd_vs_baseline_dev | 5 | 2 | 1 | 2 |
| exposure_control_dev | 5 | 3 | 2 | 2 |
| same_text_self_training_dev | 5 | 2 | 2 | 2 |
| self_training_vs_exposure_dev | 5 | 1 | 1 | 1 |
| frozen_official_test_posthoc | 5 | 4 | 3 | 3 |

Fixed-task macro averages give each adopted dataset equal weight and resample paired examples within dataset.

| Macro family | ΔcF1 | 95% CI | Decision |
|---|---:|---:|---|
| adopted_kd_vs_baseline_dev | +0.0600 | [+0.0291, +0.0911] | supported |
| exposure_control_dev | +0.0588 | [+0.0318, +0.0853] | supported |
| same_text_self_training_dev | +0.0621 | [+0.0351, +0.0888] | supported |
| self_training_vs_exposure_dev | -0.0033 | [-0.0226, +0.0170] | inconclusive |

### 5e. Resource and deployment audit

| Item | Audited value |
|---|---|
| Post-lock training | 134.98 logged run-wall hours on one RTX 4090 (includes checkpoint I/O) |
| Retained post-lock best checkpoints | 50 files, 301.54 GiB |
| DeepSeek fidelity coverage / timed rows | 1344 / 1344; 0 errors |
| Qwen fidelity coverage / timed rows | 1344 / 1264; 0 errors |
| Historical provider tokens / monetary cost | unavailable; not estimated |
| LLM calls at student inference | 0 |

Five-seed ensemble model-forward throughput reconstructed from retained dev inference logs; loading and serialization are excluded.

| Dataset | Encoder | Ensemble output rows/s |
|---|---|---:|
| eng_laptop | microsoft/deberta-v3-base | 26.46 |
| rus_restaurant | xlm-roberta-large | 11.55 |
| tat_restaurant | xlm-roberta-large | 11.08 |
| ukr_restaurant | microsoft/mdeberta-v3-base | 10.77 |
| zho_restaurant | hfl/chinese-roberta-wwm-ext-large | 17.14 |

## Table 6. One-shot official Task-3 test evaluation

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
