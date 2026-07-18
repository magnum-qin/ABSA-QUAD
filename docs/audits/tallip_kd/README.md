# TALLIP KD evidence closure

Status as of 2026-07-18: audit, 25 clean-inductive training jobs, frozen
inference, the complete 8x2 teacher-fidelity evaluation, and the one-shot
official-test evaluation are finished.

## Frozen protocol

- Text leakage key: Unicode NFKC, casefold, then Unicode whitespace collapse.
- Seeds: 42, 123, 456, 789, 1024.
- Bootstrap: 10,000 paired example-level resamples, seed 20260713.
- KD ensemble thresholds: eng_laptop vote 4; ukr_restaurant vote 3;
  zho_restaurant vote 3; all use top-k 0.
- No cap5000, no remaining rus cap3000 seeds, and no new teacher/pool grid.

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

## Frozen core tables

The five TALLIP core tables are frozen in `tallip_core_tables.md`. Their output,
generator, protocols, registry, and every machine-readable evidence input are
SHA256-pinned by `tallip_core_tables.lock.json`. Table 4 is no longer hardcoded
in the generator: its evidence rows and source lineage are registered in
`ablation_evidence.json`, while numeric results are recomputed from the cited
artifacts.

Verify before copying any value into the paper:

```bash
python scripts/kd/build_tallip_report.py --verify
```

The default command is verification-only. Rewriting the tables requires the
explicit `--update-lock` flag and, after official-test gold inspection, a new
protocol version plus an evidence audit.

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
