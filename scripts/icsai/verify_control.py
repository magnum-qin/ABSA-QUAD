"""Check the immutable implementation/evidence gate before formal training."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/audits/icsai_phase2'


def verify_readiness():
    lock = json.loads((OUT / 'implementation.lock.json').read_text())
    for path, expected in lock['sha256'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Control readiness evidence changed: {path}')
    preflight = json.loads((OUT / 'preflight.json').read_text())
    smoke = json.loads((OUT / 'gpu_smoke.json').read_text())
    assert preflight['passed'] and smoke['passed']
    assert set(preflight['core_initialization']) == {'42', '123', '456'}
    for seed, entries in preflight['core_initialization'].items():
        assert set(entries) == {'A', 'B', 'C'}
        assert entries['B'] == entries['C']
        assert len({e['shared_sha256'] for e in entries.values()}) == 1
    assert {r['arm'] for r in smoke['arms']} == {'A', 'B', 'C'}
    for r in smoke['arms']:
        assert r['optimizer_updates_executed'] == 2 and r['fgm_exercised_on_second_update']
        assert r['prediction_id_and_unique_triple_check']
        if r['arm'] == 'C':
            assert all(g['bio_embedding_gradient_l1'] == 0 for g in r['gradient_evidence'])
    log = (OUT / 'unit_tests.log').read_text()
    assert 'Ran 8 tests' in log and log.rstrip().endswith('OK')
    assert len(preflight['config_sha256']) == 42
    for path, expected in preflight['config_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    print('ICSAI control gate PASS: tests, nine pretrained initializations, A/B/C GPU smoke and 42 configs verified.')
    return True


if __name__ == '__main__': verify_readiness()
