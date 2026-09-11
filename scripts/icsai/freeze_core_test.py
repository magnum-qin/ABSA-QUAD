"""Freeze every predeclared core best checkpoint before new test inference."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from scripts.icsai.verify_control import verify_readiness

ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    verify_readiness()
    dest = ROOT / 'docs/audits/icsai_core_test/protocol_lock.json'
    if dest.exists():
        raise FileExistsError(dest)
    design_path = ROOT / 'docs/audits/icsai_phase1/experiment_matrix.json'
    design = json.loads(design_path.read_text())
    rows = list(csv.DictReader((ROOT / 'docs/audits/icsai_core_training/completed_runs.csv').open()))
    expected = {(d, a, str(s)) for d in design['statistics']['primary_datasets'] for a in 'ABC' for s in (42,123,456)}
    assert len(rows) == 27 and {(r['dataset'],r['arm'],r['seed']) for r in rows} == expected
    models = []
    for row in rows:
        base = ROOT / 'checkpoints/icsai_phase1' / row['id']
        path = base / 'checkpoint_best.pt'
        before = path.stat()
        sha = digest(path)
        after = path.stat()
        assert (before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns)
        models.append(dict(row, checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=sha,
                           checkpoint_bytes=after.st_size, config_sha256=digest(base/'resolved_config.json')))
        print(f"Hashed {len(models)}/27: {row['id']}", flush=True)
    inputs = json.loads((ROOT/'docs/audits/icsai_phase1/gold_input_manifest.json').read_text())
    tests = []
    for item in inputs['datasets']:
        if item['dataset'] not in design['statistics']['primary_datasets']:
            continue
        path = ROOT/item['test_input']
        assert digest(path) == item['test_input_sha256']
        ids = [json.loads(line)['ID'] for line in path.read_text().splitlines() if line.strip()]
        assert len(ids) == len(set(ids))
        tests.append(dict(dataset=item['dataset'],path=item['test_input'],sha256=digest(path),rows=len(ids),
                          ordered_id_sha256=hashlib.sha256(json.dumps(ids,ensure_ascii=False).encode()).hexdigest()))
    sources = ['scripts/icsai/freeze_core_test.py','scripts/icsai/analyze_core.py','src/models/icsai_control.py',
               'scripts/icsai/control_runtime.py','scripts/icsai/run_control.py','docs/audits/icsai_core_analysis/dev_statistics.json']
    lock = dict(frozen_at=datetime.now(timezone.utc).isoformat(), status='models_and_protocol_frozen_before_new_test_inference',
                design_sha256=digest(design_path),test_exposure=design['test_exposure'],models=models,test_inputs=tests,
                inference=design['inference'],statistics=design['statistics'],
                p_value_definition='Two-sided centered paired bootstrap: (1 + count(abs(bootstrap_delta-observed_delta) >= abs(observed_delta)))/(10000+1); Holm over six test contrasts separately from dev.',
                checkpoint_loading='load_control_checkpoint; strict state load followed by complete saved EMA overlay; preserve C zero BIO feedback.',
                reporting='All 27 single models are primary; all nine fixed three-seed ensembles are secondary descriptive results. No test-based selection. Hash all predictions before scoring.',
                source_sha256={p:digest(ROOT/p) for p in sources})
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x') as f:
        json.dump(lock,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(str(dest),flush=True)

if __name__ == '__main__':
    main()
