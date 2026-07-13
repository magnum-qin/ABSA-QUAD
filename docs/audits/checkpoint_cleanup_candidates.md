# Checkpoint cleanup candidates (read-only audit, 2026-07-13)

No checkpoint was deleted during this audit. The filesystem is 91% used (2.4 TiB / 2.7 TiB); `checkpoints/` occupies about 1.8 TiB physically.

## Retention boundary

- **Protected final ensembles:** 30 seed directories referenced by `submission/final_manifest.json`, 885.53 GiB.
- **Non-final experiments:** 34 seed directories, 925.13 GiB.
- The two historical baseline ensembles already missing locally (`eng_restaurant`, `jpn_hotel`) cannot be recovered by cleanup.

## Reclaim options requiring explicit confirmation

| Policy | Keep | Delete | Estimated reclaim | Trade-off |
|---|---|---|---:|---|
| A — best-only everywhere | Every `checkpoint_best.pt`, configs and result records | 64 `checkpoint_latest.pt` + 192 `checkpoint_epoch*.pt` | **~1,462.55 GiB** | Inference/ensemble remains possible; epoch-boundary resume and top-k re-sweeps are lost. |
| B — best-only non-final | All final directories unchanged; best checkpoint only for non-final runs | Non-final latest + epoch checkpoints | **~754.12 GiB** | Final training remains resumable; non-final runs become inference-only. |
| C — remove non-final runs | All 30 final seed directories | All 34 non-final directories | **~925.13 GiB** | Loses direct re-inference from ablation/淘汰 checkpoints; metrics and docs remain. |
| D — slim best checkpoints | Model/EMA/config/metrics only | Optimizer, scheduler and RNG state inside selected best files | Not estimated without rewriting | Changes checkpoint bytes and removes resume capability; requires compatibility testing and new hashes. |

Options are not additive as written. Policy A gives the largest low-risk reclaim while retaining every best model, but it is still destructive and must be confirmed separately.

## Non-final directory groups

| Group | Seeds/directories | Physical size | Status |
|---|---:|---:|---|
| `v23_rus_restaurant_deepseek_gen_seed*` | 5 | ~219.00 GiB | full-gen8k ablation, superseded |
| `v23_rus_restaurant_deepseek_seed*` | 5 | ~219.00 GiB | prior rus real-pool submission, superseded |
| `v23_rus_restaurant_deepseek_gen8k_cap3k_seed*` | 2 | ~70.08 GiB | paired gate failed |
| `final_zho_restaurant_v16_seed*` | 5 | ~131.65 GiB | baseline superseded by KD |
| `v5_6pivot_eng_laptop_seed*` | 10 | ~149.10 GiB | historical/淘汰 sweep |
| `v5repro_eng_laptop_seed*` | 5 | ~74.55 GiB | historical baseline reproduction |
| `eng_laptop_no_aug_large_seed1024` | 1 | ~34.46 GiB | incomplete/historical arm |
| `retrain_20260603_jpn_hotel_v16_no_aug_seed42` | 1 | ~27.36 GiB | single retained historical seed |

Before executing any policy, rerun `python scripts/audit_final_submission.py`, record an inode-aware dry-run list, and exclude every prefix in `submission/final_manifest.json` from whole-directory deletion.
