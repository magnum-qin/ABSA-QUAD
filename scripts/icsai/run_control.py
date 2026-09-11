#!/usr/bin/env python3
"""Train one locked ICSAI run or export its dev predictions; no test API."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from scripts.icsai.control_runtime import config_for_run, seed_all, EpochOrderSampler, ControlledTrainer
from src.models.icsai_control import build_control_model, state_fingerprint, load_control_checkpoint, validate_seed_predictions
from src.data.dataset import DimASQPDataset, collate_fn


def verified_config(run_id, device):
    cfg = config_for_run(run_id, device)
    for split in ('train', 'dev'):
        p = ROOT / cfg['data'][f'{split}_file']
        if hashlib.sha256(p.read_bytes()).hexdigest() != cfg['icsai'][f'{split}_sha256']:
            raise ValueError(f'Frozen {split} input changed')
    gate = json.loads((ROOT / 'docs/audits/icsai_phase2/preflight.json').read_text())
    if not gate['passed'] or gate['design_sha256'] != cfg['icsai']['design_sha256']:
        raise ValueError('Missing or stale initialization gate')
    for path, expected in gate['implementation_sha256'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Implementation changed after preflight: {path}')
    return cfg, gate


def make_loader(cfg, tokenizer, split):
    dataset = DimASQPDataset(
        str(ROOT / cfg['data'][f'{split}_file']), tokenizer,
        max_length=cfg['data']['max_seq_length'],
        entity_labels=cfg['data']['entity_labels'], attribute_labels=cfg['data']['attribute_labels'],
    )
    generator = torch.Generator().manual_seed(cfg['seed'])
    return DataLoader(dataset, batch_size=cfg['training']['batch_size'],
                      sampler=EpochOrderSampler(len(dataset), cfg['seed']) if split == 'train' else None,
                      shuffle=False, num_workers=0, collate_fn=collate_fn,
                      generator=generator, pin_memory=cfg['device'].startswith('cuda'))


def train_one(run_id, device):
    from scripts.icsai.verify_control import verify_readiness
    verify_readiness()
    cfg, gate = verified_config(run_id, device)
    save = ROOT / cfg['training']['save_dir']
    save.mkdir(parents=True, exist_ok=True)
    # Explicitly refuse accidental retraining/overwriting; resume is a later
    # operation requiring run-state reconciliation, not an automatic restart.
    claim = save / 'run_claim.json'
    with claim.open('x') as f:
        json.dump({'run_id': run_id, 'pid': os.getpid(), 'status': 'starting'}, f)
    try:
        seed_all(cfg['seed'])
        tokenizer = AutoTokenizer.from_pretrained(cfg['model']['encoder_name'], local_files_only=True)
        train_loader = make_loader(cfg, tokenizer, 'train')
        dev_loader = make_loader(cfg, tokenizer, 'dev')
        seed_all(cfg['seed'])  # Match preflight even if tokenizer setup consumed RNG.
        model = build_control_model(cfg)
        fingerprint = state_fingerprint(model)
        core = gate['core_initialization'].get(str(cfg['seed']), {}).get(cfg['icsai']['arm'])
        if cfg['icsai']['dataset'] in gate['core_datasets']:
            if fingerprint != core:
                raise ValueError('Actual initial parameters differ from preflight')
        (save / 'initialization.json').write_text(json.dumps(fingerprint, indent=2) + '\n')
        (save / 'resolved_config.json').write_text(json.dumps(cfg, indent=2) + '\n')
        trainer = ControlledTrainer(model, train_loader, dev_loader, cfg, device)
        trainer.train()
        completed = {'run_id': run_id, 'status': 'complete', 'global_step': trainer.global_step,
                     'best_dev_cf1': trainer.best_metric}
        (save / 'status.json').write_text(json.dumps(completed, indent=2) + '\n')
    except BaseException as exc:
        (save / 'status.json').write_text(json.dumps({'run_id': run_id, 'status': 'interrupted_or_failed',
                                                      'error': repr(exc)}, indent=2) + '\n')
        raise


def predict_dev(run_id, device):
    cfg, _ = verified_config(run_id, device)
    checkpoint = ROOT / cfg['training']['save_dir'] / 'checkpoint_best.pt'
    model, saved = load_control_checkpoint(checkpoint, device)
    if saved['icsai'] != cfg['icsai']:
        raise ValueError('Checkpoint belongs to a different run/design')
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['encoder_name'], local_files_only=True)
    loader = make_loader(cfg, tokenizer, 'dev')
    predictions = []; ids = []
    with torch.no_grad():
        for batch in loader:
            quads = model.predict(batch['input_ids'].to(device), batch['attention_mask'].to(device),
                                  batch['texts'], batch['offset_mapping'], pair_threshold=0.3)
            ids.extend(batch['sample_ids'])
            predictions.extend({'ID': i, 'Quadruplet': q} for i, q in zip(batch['sample_ids'], quads))
    validate_seed_predictions(predictions, ids)
    output = ROOT / cfg['icsai']['prediction_dir'] / 'dev.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        for row in predictions: f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'predict-dev'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'train': train_one(args.run_id, args.device)
    else: predict_dev(args.run_id, args.device)


if __name__ == '__main__': main()
