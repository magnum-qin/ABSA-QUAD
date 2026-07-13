# KD pipeline (trunk + modules)

Consolidates the ~12 bespoke `run_v23_*.sh` scripts into **one parameterized trunk**
driven by a declarative registry. Naming follows [docs/NAMING.md](../../docs/NAMING.md).

## Structure

```
trunk     scripts/kd/pipeline.py     # orchestrates pool -> silver -> distill -> train -> eval
registry  scripts/kd/registry.yaml   # every experiment declared as data (also the result ledger)
modules (reused, already tested):
  pool    scripts/gen_native_ollama.py     (LLM-generated pool)
          scripts/collect_real_pools.py    (real public corpus: Yelp / amazon_zh+opencc)
  silver  scripts/llm_silver_label.py        (deepseek, cloud)
          scripts/llm_silver_label_ollama.py (qwen/gemma, local ollama, think:false)
  train   scripts/train_multi_seed.py
  eval    scripts/inference.py --multi_seed_ensemble  +  scripts/official_eval.py
```

## Usage

```bash
python scripts/kd/pipeline.py --list                       # list all experiments + results
python scripts/kd/pipeline.py --exp zho_restaurant_qwen    # full run
python scripts/kd/pipeline.py --exp eng_laptop_deepseek --steps eval   # re-eval only
python scripts/kd/pipeline.py --exp <name> --dry-run       # print commands, run nothing
```

Add a new experiment = add one block to `registry.yaml` (no new script).

## Steps

| step | what | skip when |
|------|------|-----------|
| pool | build unlabeled pool (gen/real) + test text | `source: existing` or file present |
| silver | label pool with the teacher; unload local model after | — |
| distill | `gold_base + silver` (optional `silver_cap`) | `distill_prebuilt: true` |
| train | derive config (train_file=distill), `train_multi_seed.py` 5 seeds | — |
| eval | vote×topk sweep on dev, `official_eval.py` | — |

Long runs can set `resume_existing: true` in the registry. Training then resumes
from each seed's `checkpoint_latest.pt`, skips seeds carrying
`training_complete.json`, and appends to the existing log. Epoch checkpoints are
written atomically and `latest` / `best` / top-k names share identical bytes via
hard links, avoiding duplicate serialization without changing model state or
ensemble inputs. Resume checkpoints also restore Python, NumPy, PyTorch, and CUDA
RNG state at the epoch boundary. CUDA training defaults to 4 deterministic, non-persistent data
workers with pinned-memory prefetch; set `data.num_workers: 0` to opt out.

GPU note: a local teacher (qwen/gemma) occupies ~24GB, so silver-labeling and training
cannot overlap; the trunk unloads the model after `silver` before `train`.

The old `run_v23_*.sh` remain as run snapshots but are superseded by this trunk.
