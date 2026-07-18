# Best Reproduction Audit

Date: 2026-06-03

> **2026-07-18 retrospective:** this document audits historical reproducibility
> as of its original date. Official test labels were subsequently released and
> evaluated once under a pre-locked protocol. Current test conclusions are in
> [`docs/CONCLUSIONS.md`](CONCLUSIONS.md); the asset-recovery conclusions below
> remain historical and unchanged.

Goal: verify whether current data and training parameters can reproduce the recorded best dev cF1 in `docs/EXPERIMENTS.md`.

## Conclusion

Strict historical reproduction is not currently possible from the inspected assets. The recorded best runs require historical checkpoint prefixes and, for `eng_laptop`, the historical 15,049-row V5/V6 augmented training file. These assets were not found on the local machine or the inspected GPU servers.

Fresh retraining can still be run, but it is not proof of reproducing the recorded best dev cF1.

## Historical Best Requirements

| Dataset | Version | Required checkpoint prefix | Required train rows | Inference |
|---|---|---|---:|---|
| `eng_restaurant` | V16 | `v16_native_eng_restaurant_seed{42,123,456,789,1024}` | 2,284 | vote=4, topk=0.0 |
| `eng_laptop` | V5/V6 | `v5_eng_laptop_seed{42,123,456,789,1024}` | 15,049 | vote=3, topk=0.0, dual-layer |
| `jpn_hotel` | V16 no_aug | `v16_ablation_jpn_no_aug_seed{42,123,456,789,1024}` | 1,600 | vote=4, topk=0.0 |
| `rus_restaurant` | V15 | `v15_xlmr_rus_seed{42,123,456,789,1024}` | 1,766 | vote=4, topk=0.10 |
| `tat_restaurant` | V15 | `v15_xlmr_tat_seed{42,123,456,789,1024}` | 1,599 | vote=3, topk=0.20 |
| `ukr_restaurant` | V13b | `v13b_ukr_seed{42,123,456,789,1024}` | 2,967 | vote=3, topk=0.0 |
| `zho_restaurant` | V16 | `v16_native_zho_restaurant_seed{42,123,456,789,1024}` | 6,050 | vote=4, topk=0.0 |
| `zho_laptop` | V16 | `v16_native_zho_laptop_seed{42,123,456,789,1024}` | 5,534 (`zho_laptop_train_v13c.jsonl`) | vote=4, topk=0.10 |

## Inspected Locations

Checked by read-only SSH/search:

- `rtx4090-47`: `/home`, `/tmp`
- `rtx4090-56`: `/home`, `/tmp`
- `rtx4090-224`: `/home`, `/tmp`
- `a6000-143`: `/home/g24tka12`, `/tmp/g24tka12_absa`
- `a6000-048`: `/home/g24tka12`, `/tmp/g24tka12_absa`
- `a6000-147`: `/home/g24tka12`, `/tmp/g24tka12_absa`
- Local workspace: `D:\Projects\ABSA-QUAD`

No historical checkpoint directories or checkpoint archives matching the required prefixes were found.

## Data Mismatches

`eng_laptop` is the concrete data mismatch:

- Historical V5/V6 record: `eng_laptop_train_alltasks_augmented.jsonl` has 15,049 rows.
- Current remote 4090 files:
  - `eng_laptop_train_alltasks.jsonl`: 4,076 rows
  - `eng_laptop_train_alltasks_augmented_bt_only.jsonl`: 8,556 rows
  - `eng_laptop_train_alltasks_augmented.jsonl`: 12,632 rows
  - `eng_laptop_train_task3_augmented.jsonl` on `rtx4090-224`: 4,827 rows
- Local workspace only has the raw `eng_laptop_train_alltasks.jsonl`.

The current 12,632-row file is raw + BT-only, not the historical 15,049-row V5/V6 file.

## Existing Local Test Predictions

The tracked local `submission/final/*.jsonl` files are old test predictions. At
the time of this audit, test labels were not locally available, so these files
could not verify dev cF1. Released labels are now available and were evaluated
separately under the locked official-test protocol; these historical prediction
files are not the TALLIP clean-inductive systems.

Current local tracked test prediction sizes:

| File | Samples | Quads |
|---|---:|---:|
| `eng_laptop_test.jsonl` | 1,000 | 1,186 |
| `eng_restaurant_test.jsonl` | 1,000 | 1,792 |
| `jpn_hotel_test.jsonl` | 800 | 1,441 |
| `rus_restaurant_test.jsonl` | 630 | 1,140 |
| `tat_restaurant_test.jsonl` | 630 | 1,147 |
| `ukr_restaurant_test.jsonl` | 630 | 1,091 |
| `zho_laptop_test.jsonl` | 1,000 | 1,344 |
| `zho_restaurant_test.jsonl` | 1,000 | 2,677 |

Deleted-but-tracked backup files in `HEAD`:

| File | Samples | Quads |
|---|---:|---:|
| `eng_laptop_test_v16_backup.jsonl` | 1,000 | 1,655 |
| `eng_laptop_test_v19_backup.jsonl` | 1,000 | 1,335 |
| `jpn_hotel_test_v16_aug_backup.jsonl` | 800 | 1,087 |

## Commands

Audit current assets:

```bash
python scripts/audit_best_reproduction.py
```

Generate strict historical test predictions after restoring all required assets:

```bash
REPRO_MODE=historical bash scripts/run_final_best_submission_4090.sh
```

Run a fresh retrain, explicitly not a strict historical reproduction:

```bash
REPRO_MODE=retrain bash scripts/run_final_best_submission_4090.sh
```
