#!/usr/bin/env python3
"""Isolated real-GPU training-path smoke test; no dev/test scoring or saved models."""
import gc
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from scripts.icsai.control_runtime import seed_all, EpochOrderSampler, ControlledTrainer
from scripts.icsai.run_control import verified_config, make_loader
from src.models.icsai_control import build_control_model, state_fingerprint, validate_seed_predictions
from src.data.dataset import collate_fn


def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; this real-GPU gate is not passed')
    torch.set_num_threads(2)
    reports = []
    for arm in 'ABC':
        cfg, gate = verified_config(f'eng_restaurant_{arm}_s42', 'cuda')
        seed_all(42)
        tokenizer = AutoTokenizer.from_pretrained(cfg['model']['encoder_name'], local_files_only=True)
        source = make_loader(cfg, tokenizer, 'train')
        # First eight clean train rows; deterministic, no dev-based selection.
        subset = Subset(source.dataset, list(range(8)))
        loader = DataLoader(subset, batch_size=8, sampler=EpochOrderSampler(8,42),
                            collate_fn=collate_fn, generator=torch.Generator().manual_seed(42))
        seed_all(42)
        model = build_control_model(cfg)
        initial = state_fingerprint(model)
        assert initial == gate['core_initialization']['42'][arm]
        gradient_evidence = []
        with tempfile.TemporaryDirectory(prefix='icsai_smoke_') as temp:
            cfg['training']['save_dir'] = temp
            trainer = ControlledTrainer(model, loader, loader, cfg, 'cuda')
            torch.cuda.reset_peak_memory_stats()
            def observed_step(optimizer, args, kwargs):
                row = {'nonzero_gradient_parameters': 0, 'gradient_present_parameters': 0}
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all().item(), name
                        row['gradient_present_parameters'] += p.numel()
                        if torch.count_nonzero(p.grad).item():
                            row['nonzero_gradient_parameters'] += p.numel()
                        if name.endswith('label_embedding.bio_embedding.weight'):
                            row['bio_embedding_gradient_l1'] = p.grad.abs().sum().item()
                gradient_evidence.append(row)
            step_hook = trainer.optimizer.register_step_pre_hook(observed_step)
            metrics = []; started = time.monotonic()
            for epoch in (1, 2):
                loader.sampler.epoch = epoch
                if epoch == 2:
                    # Exercise the existing FGM branch explicitly. This is a
                    # smoke-only step counter intervention, never a formal run.
                    trainer.global_step = trainer.num_warmup_steps
                metrics.append(trainer.train_epoch(epoch))
                assert all(torch.isfinite(torch.tensor(v)).item() for v in metrics[-1].values())
            model.eval()
            batch = next(iter(loader))
            with torch.no_grad():
                quads = model.predict(batch['input_ids'].cuda(), batch['attention_mask'].cuda(),
                                      batch['texts'], batch['offset_mapping'], pair_threshold=0.3)
            predictions = [{'ID':i,'Quadruplet':q} for i,q in zip(batch['sample_ids'],quads)]
            validate_seed_predictions(predictions, batch['sample_ids'])
            if arm == 'C': assert all(r['bio_embedding_gradient_l1'] == 0 for r in gradient_evidence)
            if arm == 'B': assert any(r['bio_embedding_gradient_l1'] > 0 for r in gradient_evidence)
            record = {'arm':arm, 'initialization': initial, 'optimizer_updates_executed':2,
                      'fgm_exercised_on_second_update': True, 'train_metrics':metrics,
                      'gradient_evidence': gradient_evidence, 'prediction_id_and_unique_triple_check':True,
                      'wall_seconds':time.monotonic()-started,
                      'peak_allocated_cuda_bytes':torch.cuda.max_memory_allocated()}
            reports.append(record)
            print(json.dumps(record), flush=True)
            step_hook.remove()
            del observed_step, trainer
        del model, source, loader, subset, tokenizer
        gc.collect(); torch.cuda.empty_cache()
    result = {'passed':True,'scope':'Two isolated optimizer updates per arm on first eight clean English train rows. Second update manually enters FGM branch. Not a formal experiment, accuracy comparison or full-epoch cost estimate. No dev/test data scored; no checkpoint retained.',
              'gpu':torch.cuda.get_device_name(), 'torch_version':torch.__version__, 'arms':reports}
    (ROOT/'docs/audits/icsai_phase2/gpu_smoke.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__': main()
