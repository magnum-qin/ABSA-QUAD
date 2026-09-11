"""Frozen-design config factory, RNG-independent sampler and strict trainer."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import Sampler

from scripts.train import Trainer

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / 'docs/audits/icsai_phase1/experiment_matrix.json'
INPUTS = ROOT / 'docs/audits/icsai_phase1/gold_input_manifest.json'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pinned_snapshot(design):
    hub = Path(os.environ.get('HF_HUB_CACHE', str(Path.home() / '.cache/huggingface/hub')))
    snapshot = hub / ('models--' + design['encoder']['name'].replace('/', '--')) / 'snapshots' / design['encoder']['revision']
    for name in ('config.json', 'pytorch_model.bin', 'spm.model', 'tokenizer_config.json'):
        if not (snapshot / name).is_file():
            raise FileNotFoundError(f'Pinned offline encoder asset missing: {snapshot / name}')
    return snapshot


def config_for_run(run_id, device='cuda'):
    design = json.loads(DESIGN.read_text())
    runs = {r['id']: r for r in design['experiments']}
    if run_id not in runs:
        raise ValueError(f'Run not in frozen matrix: {run_id}')
    run = runs[run_id]
    inp = next(r for r in json.loads(INPUTS.read_text())['datasets'] if r['dataset'] == run['dataset'])
    labels = design['label_maps'][run['dataset']]
    t = design['training']
    snapshot = pinned_snapshot(design)
    hidden = json.loads((snapshot / 'config.json').read_text())['hidden_size']
    return {
        'icsai': {'run_id': run_id, 'arm': run['arm'], 'dataset': run['dataset'],
                  'design_sha256': hashlib.sha256(DESIGN.read_bytes()).hexdigest(),
                  'encoder_revision': design['encoder']['revision'],
                  'train_sha256': inp['train_sha256'], 'dev_sha256': inp['dev_sha256'],
                  'prediction_dir': run['prediction_dir']},
        'model': {'encoder_name': str(snapshot), 'hidden_size': hidden,
                  'num_entity_labels': len(labels['entity_labels']),
                  'num_attribute_labels': len(labels['attribute_labels']),
                  'va_hidden_size': 256, 'dropout': t['dropout'], 'va_min': 1.0, 'va_max': 9.0,
                  'use_crf': False, 'use_uml': False, 'category_embed_dim': 0,
                  'rethink': {'enabled': run['arm'] != 'A', 'num_iterations': 2, 'label_embedding_dim': 64}},
        # No test file in the training config or training API.
        'data': {'train_file': inp['train_file'], 'dev_file': inp['dev_file'],
                 'max_seq_length': t['max_seq_length'], 'num_workers': 0,
                 'entity_labels': labels['entity_labels'], 'attribute_labels': labels['attribute_labels']},
        'training': {'batch_size': t['batch_size'], 'gradient_accumulation_steps': t['gradient_accumulation_steps'],
                     'num_epochs': t['epochs'], 'patience': t['patience'],
                     'encoder_lr': t['encoder_lr'], 'head_lr': t['head_lr'],
                     'weight_decay': t['weight_decay'], 'warmup_ratio': t['warmup_ratio'],
                     'max_grad_norm': t['max_grad_norm'], 'fgm': t['fgm'], 'ema': t['ema'],
                     'loss_weights': t['loss_weights'], 'save_top_k': 1, 'save_dir': run['checkpoint_dir']},
        'seed': run['seed'], 'device': device,
    }


class EpochOrderSampler(Sampler):
    """Per-epoch permutations independent of model/dropout/global RNG use."""
    def __init__(self, size, seed):
        self.size = size
        self.seed = seed
        self.epoch = 1

    def order(self):
        generator = torch.Generator().manual_seed(self.seed + 1_000_003 * self.epoch)
        return torch.randperm(self.size, generator=generator).tolist()

    def __iter__(self):
        return iter(self.order())

    def __len__(self):
        return self.size


def select_best(score, best_score):
    if not math.isfinite(score):
        raise ValueError('Non-finite dev cF1')
    return best_score is None or score > best_score


class ControlledTrainer(Trainer):
    """Reuse original forward/backward/FGM/EMA, with frozen selection semantics."""
    def train(self):
        best_score = self.best_metric if (self.save_dir / 'checkpoint_best.pt').exists() else None
        for epoch in range(self.start_epoch, self.config['training']['num_epochs'] + 1):
            sampler = self.train_dataloader.sampler
            sampler.epoch = epoch
            order = sampler.order()
            self.train_dataloader.generator.manual_seed(self.config['seed'] + epoch)
            self.val_dataloader.generator.manual_seed(self.config['seed'] + epoch)
            started = time.monotonic()
            previous_steps = self.global_step
            train_metrics = self.train_epoch(epoch)
            expected_updates = len(self.train_dataloader)  # frozen accumulation=1
            if self.global_step - previous_steps != expected_updates:
                raise RuntimeError('Training update count differs from frozen exposure')
            val_metrics = self.evaluate()
            current = float(val_metrics['continuous_f1'])
            is_best = select_best(current, best_score)
            if is_best:
                best_score = current
                self.best_metric = current
            self.patience_counter = 0 if is_best else self.patience_counter + 1
            self.best_val_loss = min(getattr(self, 'best_val_loss', float('inf')), val_metrics['val_loss'])
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            row = {'epoch': epoch, 'global_step': self.global_step, 'updates': expected_updates,
                   'order_sha256': hashlib.sha256(json.dumps(order).encode()).hexdigest(),
                   'train_metrics': train_metrics, 'dev_metrics': val_metrics,
                   'selected_best': is_best, 'wall_seconds': time.monotonic() - started}
            with (self.save_dir / 'epochs.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
        return self.best_metric
