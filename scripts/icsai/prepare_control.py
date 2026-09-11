#!/usr/bin/env python3
"""Prepare all locked configs and measure actual pretrained A/B/C initialization."""
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from scripts.icsai.control_runtime import config_for_run, seed_all, DESIGN
from src.models.icsai_control import build_control_model, state_fingerprint


def main():
    torch.set_num_threads(2)
    design = json.loads(DESIGN.read_text())
    configs = ROOT / 'configs/icsai_phase2'
    configs.mkdir(exist_ok=True)
    config_hashes = {}
    for run in design['experiments']:
        cfg = config_for_run(run['id'])
        p = configs / (run['id'] + '.json')
        p.write_text(json.dumps(cfg, indent=2) + '\n')
        config_hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    core = design['statistics']['primary_datasets']
    # All three core domains have exactly the same ordered 7/6 label maps.
    # Verify this before using one domain as the architecture representative.
    for domain in core:
        labels = design['label_maps'][domain]
        for field in ('entity_labels', 'attribute_labels'):
            assert labels[field] == design['label_maps'][core[0]][field]
    fingerprints = {}
    for seed in design['training']['seeds']:
        entries = {}
        for arm in 'ABC':
            cfg = config_for_run(f'{core[0]}_{arm}_s{seed}', device='cpu')
            seed_all(seed)
            model = build_control_model(cfg)
            entries[arm] = state_fingerprint(model)
            print(f"seed={seed} arm={arm}: {entries[arm]}", flush=True)
            del model; gc.collect()
        assert len({x['shared_sha256'] for x in entries.values()}) == 1
        assert entries['B'] == entries['C']
        fingerprints[str(seed)] = entries
    sources = ['src/models/icsai_control.py', 'scripts/icsai/control_runtime.py',
               'scripts/icsai/run_control.py', 'scripts/icsai/prepare_control.py',
               'src/models/dimabsa_model.py', 'src/models/feature_rethink.py', 'src/models/va_module.py',
               'src/data/dataset.py', 'src/evaluation/metrics.py', 'scripts/train.py', 'scripts/inference.py']
    result = {'passed': True, 'scope': 'Actual offline pretrained initialization only; no optimizer steps, dev/test evaluation or performance claims.',
              'design_sha256': hashlib.sha256(DESIGN.read_bytes()).hexdigest(), 'core_datasets': core,
              'core_initialization': fingerprints, 'config_sha256': config_hashes,
              'implementation_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
              'torch_version': torch.__version__, 'pretrained_revision': design['encoder']['revision']}
    out = ROOT / 'docs/audits/icsai_phase2/preflight.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print('All 42 configs prepared; nine actual A/B/C initializations passed; no training started.')


if __name__ == '__main__': main()
