#!/usr/bin/env python
"""
KD pipeline TRUNK — one parameterized orchestrator replacing the ~12 bespoke
run_v23_*.sh scripts. Reads scripts/kd/registry.yaml and runs an experiment's
steps: pool -> silver -> distill -> train -> eval.

Modules it composes (already existing, tested):
  pool    : scripts/gen_native_ollama.py (gen) | scripts/collect_real_pools.py (real)
  silver  : scripts/llm_silver_label.py (deepseek) | scripts/llm_silver_label_ollama.py (local)
  train   : scripts/train_multi_seed.py
  eval    : scripts/inference.py --multi_seed_ensemble + scripts/official_eval.py

Usage:
  python scripts/kd/pipeline.py --list
  python scripts/kd/pipeline.py --exp zho_restaurant_qwen                 # full run
  python scripts/kd/pipeline.py --exp zho_restaurant_qwen --steps eval    # just re-eval
  python scripts/kd/pipeline.py --exp zho_restaurant_qwen --dry-run
"""
import os, re, sys, json, argparse, subprocess, random
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
REG = yaml.safe_load(open("scripts/kd/registry.yaml"))
DEF = REG["defaults"]
ALL_STEPS = ["pool", "silver", "distill", "train", "eval"]
DRY = False


def sh(cmd):
    print(f"  $ {cmd}", flush=True)
    if not DRY:
        subprocess.run(cmd, shell=True, check=True)


def conda(py):
    return f"conda run -n absa --no-capture-output python {py}"


def resolve(exp):
    e = dict(REG["experiments"][exp])
    e["name"] = exp
    e["teacher_model"] = {"qwen": DEF["qwen_model"], "gemma": DEF["gemma_model"],
                          "deepseek": DEF["deepseek_model"]}[e["teacher"]]
    e["prefix"] = f"v23_{e['dataset']}_{e['teacher']}" + (f"_{e['variant']}" if e.get("variant") else "")
    e["alltasks"] = f"data/subtask_3/{e['lang']}/{e['dataset']}_train_alltasks.jsonl"
    e["silver_file"] = (f"data/subtask_3/{e['lang']}/{e['dataset']}_silver_v23_{e['teacher']}"
                        + (f"_{e['variant']}" if e.get('variant') else "") + ".jsonl")
    e["distill_file"] = (f"data/subtask_3/{e['lang']}/{e['dataset']}_train_v23_{e['teacher']}"
                         + (f"_{e['variant']}" if e.get('variant') else "") + ".jsonl")
    e["v23_config"] = f"/tmp/pipeline_{e['prefix']}.yaml"
    return e


def step_pool(e):
    pool = e["pool"]; out = pool["file"]
    if pool["source"] == "existing" or os.path.exists(out):
        print(f"  pool exists: {out}"); return
    if pool["source"] == "gen":
        sh(f"OLLAMA_URL={DEF['ollama_url']} {conda('scripts/gen_native_ollama.py')} "
           f"--dataset {e['dataset']} --n {pool.get('n',4000)} --output {out} --model '{e['teacher_model']}'")
    elif pool["source"] == "real":
        sh(f"{conda('scripts/collect_real_pools.py')} --dataset {e['dataset']} --n {pool.get('n',4000)} "
           f"--test_file {e['test']} --output {out}")
    else:
        print(f"  [pool] source={pool['source']} must be prebuilt; expected {out}", file=sys.stderr)


def step_silver(e):
    if e["teacher"] == "deepseek":
        sh(f"{conda('scripts/llm_silver_label.py')} --input {e['pool']['file']} --train_file {e['alltasks']} "
           f"--output {e['silver_file']} --model {e['teacher_model']} --domain {e['domain']} --n_shots 6")
    else:  # local teacher via Unsloth Studio (OpenAI-compatible); key from $UNSLOTH_API_KEY
        sh(f"{conda('scripts/llm_silver_label_openai.py')} --input {e['pool']['file']} "
           f"--train_file {e['alltasks']} --output {e['silver_file']} "
           f"--base_url {DEF['local_base_url']} --model '{e['teacher_model']}' "
           f"--domain {e['domain']} --workers 4 --n_shots 6")
        if not DRY:  # free GPU for training (Unsloth Studio: unload needs {model_path}, lives at /api/..)
            host = DEF["local_base_url"].rsplit("/v1", 1)[0]
            subprocess.run(
                f"curl -s -X POST {host}/api/inference/unload "
                f"-H \"Authorization: Bearer $UNSLOTH_API_KEY\" -H 'Content-Type: application/json' "
                f"-d '{{\"model_path\":\"{e['teacher_model']}\"}}' >/dev/null 2>&1", shell=True)
            import time as _t; _t.sleep(8)  # let VRAM free before training claims it


def step_distill(e):
    if e.get("distill_prebuilt"):
        print(f"  distill prebuilt: {e['gold_base']} (used as-is)")
        e["distill_file"] = e["gold_base"]; return
    if DRY:
        print(f"  build {e['distill_file']} = {e['gold_base']} + silver(cap={e['silver_cap']})"); return
    rows = [json.loads(l) for l in open(e["gold_base"])]; n0 = len(rows)
    sv = []
    for l in open(e["silver_file"]):
        d = json.loads(l)
        if d.get("_err") or not d.get("Quadruplet"): continue
        sv.append({"ID": d["ID"], "Text": d["Text"], "Quadruplet": d["Quadruplet"]})
    if not sv:
        raise SystemExit(f"[distill] ABORT: silver file {e['silver_file']} has 0 valid quads "
                         f"(labeling likely failed, e.g. 404/empty). Refusing to build a gold-only "
                         f"distill set that would silently masquerade as KD. Re-run the silver step.")
    if e["silver_cap"]:
        random.seed(42); random.shuffle(sv); sv = sv[:e["silver_cap"]]
    rows += sv
    with open(e["distill_file"], "w") as f:
        for d in rows: f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  base={n0} + silver={len(sv)} = {len(rows)}  ratio={len(sv)/n0:.2f} -> {e['distill_file']}")


def ensure_config(e):
    """Derive the v23 train/eval config (base_config + distill train_file). Idempotent;
    needed by BOTH train and eval (the /tmp config does not survive across sessions)."""
    if DRY:
        return
    txt = open(e["base_config"]).read()
    txt = re.sub(r'(train_file:\s*)"[^"]*"', r'\1"%s"' % e["distill_file"], txt, count=1)
    open(e["v23_config"], "w").write(txt)


def step_train(e):
    ensure_config(e)
    seeds = " ".join(map(str, DEF["seeds"]))
    sh(f"{conda('scripts/train_multi_seed.py')} --config {e['v23_config']} --seeds {seeds} "
       f"--save_prefix {e['prefix']}_seed --train_file {e['distill_file']} "
       f"> /tmp/{e['prefix']}_train.log 2>&1")


def step_eval(e):
    ensure_config(e)  # regenerate config if missing (e.g. running eval alone after a session restart)
    sd = " ".join(f"checkpoints/{e['prefix']}_seed{s}" for s in DEF["seeds"])
    res = f"/tmp/{e['prefix']}_results.txt"
    if not DRY: open(res, "w").write(f"# {e['name']} (baseline result {e.get('result')})\n")
    for vote in DEF["vote_sweep"]:
        for topk in DEF["topk_sweep"]:
            out = f"/tmp/{e['prefix']}_v{vote}_t{topk}.jsonl"
            sh(f"{conda('scripts/inference.py')} --multi_seed_ensemble --seed_dirs {sd} "
               f"--config {e['v23_config']} --test_file {e['dev']} --output {out} "
               f"--vote_threshold {vote} --span_topk_threshold {topk} --batch_size 16 "
               f"> /tmp/{e['prefix']}_eval_{vote}_{topk}.log 2>&1")
            sh(f"echo -n 'vote={vote} topk={topk} -> ' >> {res}; "
               f"{conda('scripts/official_eval.py')} -p {out} -g {e['dev']} -t 3 2>/dev/null "
               f"| grep -oE 'cF1: [0-9.]+' | tail -1 >> {res}")
    if not DRY:
        print("\n" + open(res).read())


STEP_FN = {"pool": step_pool, "silver": step_silver, "distill": step_distill,
           "train": step_train, "eval": step_eval}


def main():
    global DRY
    p = argparse.ArgumentParser()
    p.add_argument("--exp")
    p.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=ALL_STEPS)
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    DRY = a.dry_run

    if a.list or not a.exp:
        print(f"{'experiment':<28} {'teacher':<9} {'status':<10} result")
        for k, v in REG["experiments"].items():
            print(f"{k:<28} {v['teacher']:<9} {v.get('status',''):<10} {v.get('result','')}")
        return

    e = resolve(a.exp)
    print(f"=== {e['name']}  (prefix={e['prefix']}, teacher={e['teacher']}) ===")
    for s in ALL_STEPS:
        if s in a.steps:
            print(f"\n--- step: {s} ---")
            STEP_FN[s](e)
    print(f"\n=== {e['name']} done ===")


if __name__ == "__main__":
    main()
