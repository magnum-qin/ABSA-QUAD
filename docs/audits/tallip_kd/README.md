# TALLIP KD evidence closure

Status as of 2026-08-08: audit, 25 clean-inductive training jobs, frozen
inference, the complete 8x2 teacher-fidelity evaluation, the one-shot
official-test evaluation, and 40 post-lock dev-only control jobs are finished.

## Frozen protocol

- Text leakage key: Unicode NFKC, casefold, then Unicode whitespace collapse.
- Seeds: 42, 123, 456, 789, 1024.
- Bootstrap: 10,000 paired example-level resamples, seed 20260713.
- KD ensemble thresholds: eng_laptop vote 4; ukr_restaurant vote 3;
  zho_restaurant vote 3; all use top-k 0.
- No cap5000, no remaining rus cap3000 seeds, and no new teacher/pool grid.
- Post-lock controls are dev-only; official-test evaluation and selection are
  forbidden by `scripts/kd/postlock_v1.yaml`.

## Confirmed leakage and remediation

| Dataset/system | Original | Removed | Clean | Dev | Test |
|---|---:|---:|---:|---:|---:|
| eng_laptop KD | 23,590 | 995 | 22,595 | 1 | 994 |
| eng_laptop baseline | 15,049 | 1 | 15,048 | 1 | 0 |
| ukr_restaurant KD | 9,105 | 605 | 8,500 | 1 | 604 |
| ukr_restaurant baseline | 2,967 | 1 | 2,966 | 1 | 0 |
| zho_restaurant KD-real | 10,911 | 979 | 9,932 | 1 | 979 |

The adopted rus-gen4k and tat-qwen training sets have zero normalized dev/test
text overlap. The full source-level report is in `contamination_audit.json`; the
clean file and removal hashes are in `cleaning_manifest.json`.

## Baseline correction discovered during validation

The historical KD notes compare zho_restaurant KD against 0.4745. Validation
recovered a stronger no-KD V16 dev prediction at 0.5669. The clean KD dev result
is 0.5738, but its paired CI against 0.5669 crosses zero. On official test,
clean KD is 0.4660 versus 0.4798 for the frozen baseline (delta -0.0137,
95% CI [-0.0289, +0.0014]). zho_restaurant is therefore not counted as a
statistically supported KD success.

No paper LaTeX is modified by this workflow.

## Final evidence status

- Training: 25/25 complete; every run has `checkpoint_best.pt`.
- Frozen inference: 60/60 prediction files complete.
- Teacher fidelity: DeepSeek 1,344/1,344 and Qwen 1,344/1,344, both with zero
  final error records and verified model identities.
- Paired teacher bootstrap: DeepSeek is higher on all eight datasets; the
  difference is supported on eng_restaurant, zho_laptop, and zho_restaurant,
  and inconclusive on the other five.
- Core paper tables: `tallip_core_tables.md`.

## Post-lock controls

- Training: 25/25 exposure-matched repeated-gold and 15/15 same-text
  self-training jobs complete; 40/40 `checkpoint_best.pt` files and completion
  markers retained.
- Inference: 48/48 prediction files have exact dev ID/row/order coverage; all
  eight ensembles reproduce exactly from the five seed files and frozen votes.
- Statistics: 11 paired comparisons use 10,000 example-level resamples and seed
  20260713; every comparison was repeated and matched exactly.
- Exposure control: LLM-KD is supported over repeated gold on eng_laptop,
  rus_restaurant, and ukr_restaurant; tat_restaurant and zho_restaurant are
  inconclusive.
- Same-text control: LLM-KD is supported over student self-training on eng and
  rus; zho is statistically tied. On zho, self-training is supported over
  repeated gold.
- Cleanup: 60 `checkpoint_latest.pt`/`checkpoint_epoch*.pt` files were deleted,
  freeing 365,233,267,664 bytes; all best checkpoints, predictions, and audits
  remain.
- Evidence: `postlock_audit.json`, `postlock_metrics.json`, and
  `postlock_table.md`.

## Frozen core tables

The six TALLIP core tables are frozen in `tallip_core_tables.md`. Their output,
generator, protocols, registry, and every machine-readable evidence input are
SHA256-pinned by `tallip_core_tables.lock.json`. Table 4 is no longer hardcoded
in the generator: its evidence rows and source lineage are registered in
`ablation_evidence.json`, while numeric results are recomputed from the cited
artifacts.

Table 5 adds the post-lock dev-only controls; the one-shot official-test table
is Table 6. This append-only evidence update does not alter any frozen test
system, prediction, threshold, or conclusion.

Verify before copying any value into the paper:

```bash
python scripts/kd/build_tallip_report.py --verify
```

The default command is verification-only. Rewriting the tables requires the
explicit `--update-lock` flag. After official-test gold inspection, only a new
dev-only protocol plus an explicit evidence audit may append evidence; frozen
test choices cannot change.

## One-shot official test

- Protocol and system matrix were frozen before gold retrieval; selection used
  dev only and no test-label tuning is allowed.
- Official source commit: `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`.
- All eight local unlabelled test files match the released gold exactly in ID,
  order, and Text.
- KD-vs-baseline is supported on eng_laptop, rus_restaurant, tat_restaurant,
  and ukr_restaurant; zho_restaurant is inconclusive with a negative point
  estimate.
- Tatar matched-pool DeepSeek-vs-Qwen remains inconclusive on test.
- None of the eight dev-selected main systems exceeds the official leaderboard
  best; no SOTA claim is allowed.
- Evidence: `official_test/protocol_lock.json`, `official_test/gold_audit.json`,
  `official_test/official_test_metrics.json`, and
  `official_test/official_test_table.md`.
