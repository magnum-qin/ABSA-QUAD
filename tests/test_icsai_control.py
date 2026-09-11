import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from scripts.inference import _create_model
from scripts.icsai.control_runtime import EpochOrderSampler, select_best, seed_all, config_for_run, ControlledTrainer
from src.models.icsai_control import build_control_model, state_fingerprint, load_control_checkpoint, validate_seed_predictions


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 16)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def config(arm):
    return {
        'icsai': {'arm': arm},
        'model': {'encoder_name': 'synthetic-offline', 'hidden_size': 16,
                  'num_entity_labels': 7, 'num_attribute_labels': 6,
                  'va_hidden_size': 256, 'dropout': 0.0, 'va_min': 1.0, 'va_max': 9.0,
                  'rethink': {'enabled': arm != 'A', 'num_iterations': 2, 'label_embedding_dim': 64}},
        'data': {'entity_labels': ['AMBIENCE','DRINKS','FOOD','LOCATION','NULL','RESTAURANT','SERVICE'],
                 'attribute_labels': ['GENERAL','MISCELLANEOUS','NULL','PRICES','QUALITY','STYLE_OPTIONS']},
        'training': {'loss_weights': {'span':2.0,'pair':1.0,'entity':1.0,'attribute':1.0,
                                     'va':1.0,'va_valence':1.0,'va_arousal':2.0,'consistency':0.5}},
    }


def fixed_spans(model):
    extract = lambda logits, mask: [[(1, 1, 'aspect'), (3, 3, 'opinion')] for _ in range(logits.shape[0])]
    model.span_extractor.extract_spans = extract
    if model.use_rethink:
        model.span_extractor_stage2.extract_spans = extract
    return model.eval()


ARGS = {'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6]]),
        'attention_mask': torch.ones(1, 6, dtype=torch.long), 'return_loss': False}
KEYS = ['bio_logits', 'pair_logits', 'entity_logits', 'attribute_logits', 'valence', 'arousal']


class ControlTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.encoder_patch = patch('src.models.dimabsa_model.AutoModel.from_pretrained', side_effect=lambda *a, **k: TinyEncoder())
        self.config_patch = patch('src.models.dimabsa_model.AutoConfig.from_pretrained', return_value=SimpleNamespace(hidden_size=16))
        self.encoder_patch.start(); self.config_patch.start()
        self.addCleanup(self.encoder_patch.stop); self.addCleanup(self.config_patch.stop)

    def build(self, arm):
        seed_all(42)
        return fixed_spans(build_control_model(config(arm)))

    def test_initial_weights_rng_and_legacy_b_parity(self):
        models = {arm: self.build(arm) for arm in 'ABC'}
        fingerprints = {arm: state_fingerprint(m) for arm, m in models.items()}
        self.assertEqual(len({f['shared_sha256'] for f in fingerprints.values()}), 1)
        self.assertEqual(fingerprints['B'], fingerprints['C'])
        seed_all(42); legacy = fixed_spans(_create_model(config('B')))
        after_legacy = torch.get_rng_state()
        controlled = self.build('B')
        self.assertTrue(torch.equal(after_legacy, torch.get_rng_state()))
        expected, actual = legacy(**ARGS), controlled(**ARGS)
        for k in KEYS:
            self.assertTrue(torch.equal(expected[k], actual[k]), k)

    def test_zero_feedback_causal_path_and_gradients(self):
        deltas = {}
        for arm in 'BC':
            model = self.build(arm)
            ref = model(**ARGS)
            calls = []
            def perturb(module, args, kw):
                calls.append(1)
                kw = dict(kw)
                kw['bio_probs'] = torch.zeros_like(kw['bio_probs'])
                kw['bio_probs'][..., 0] = 1
                return args, kw
            hook = model.rethink_module.register_forward_pre_hook(perturb, with_kwargs=True)
            changed = model(**ARGS); hook.remove()
            self.assertEqual(len(calls), 2)
            deltas[arm] = max((changed[k] - ref[k]).abs().max().item() for k in KEYS)
            sum(ref[k].square().sum() for k in KEYS).backward()
            emb_grad = model.rethink_module.label_embedding.bio_embedding.weight.grad
            self.assertIsNotNone(emb_grad)
            if arm == 'B': self.assertGreater(emb_grad.abs().sum().item(), 0)
            else: self.assertEqual(emb_grad.abs().sum().item(), 0)
            self.assertGreater(model.rethink_module.self_attention.in_proj_weight.grad.abs().sum().item(), 0)
            self.assertIsNone(model.rethink_module.label_embedding.va_projector.weight.grad)
        self.assertGreater(deltas['B'], 0)
        self.assertEqual(deltas['C'], 0)

    def test_real_loss_optimizer_step_all_arms(self):
        for arm in 'ABC':
            model = self.build(arm).train()
            before = model.encoder.embedding.weight.detach().clone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            args = {**ARGS, 'return_loss': True, 'bio_labels': torch.tensor([[0,1,0,3,0,0]]),
                    'pair_labels': [[{'aspect_start':1, 'aspect_end':1, 'opinion_start':3, 'opinion_end':3,
                                     'entity':2, 'attribute':4, 'valence':7.0, 'arousal':5.0}]]}
            out = model(**args)
            self.assertTrue(torch.isfinite(out['loss']).item())
            out['loss'].backward(); optimizer.step()
            self.assertFalse(torch.equal(before, model.encoder.embedding.weight))

    def test_checkpoint_restores_c_arm_and_ema(self):
        model = self.build('C')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'checkpoint.pt'
            torch.save({'config': config('C'), 'model_state_dict': model.state_dict(),
                        'ema_state_dict': {'shadow': {n:p.detach().clone() for n,p in model.named_parameters()}}}, path)
            restored, cfg = load_control_checkpoint(path)
            fixed_spans(restored)
            self.assertFalse(restored.rethink_module.label_embedding.bio_feedback_enabled)
            for k in KEYS: self.assertTrue(torch.equal(model(**ARGS)[k], restored(**ARGS)[k]))

    def test_sampler_ignores_dropout_and_can_resume_epoch(self):
        sampler = EpochOrderSampler(57, 42)
        for epoch in range(1, 4):
            sampler.epoch = epoch
            before = list(sampler)
            torch.rand(1000 * epoch)
            fresh = EpochOrderSampler(57, 42); fresh.epoch = epoch
            self.assertEqual(before, list(fresh))
            self.assertEqual(sorted(before), list(range(57)))
        previous = list(sampler); sampler.epoch += 1
        self.assertNotEqual(previous, list(sampler))

    def test_earliest_best_even_at_zero_and_no_early_stop(self):
        trainer = object.__new__(ControlledTrainer)
        with tempfile.TemporaryDirectory() as temp:
            trainer.save_dir = Path(temp)
            trainer.config = {'seed': 42, 'training': {'num_epochs': 3, 'patience': 1}}
            trainer.start_epoch = 1; trainer.global_step = 0
            trainer.best_metric = 0.0; trainer.patience_counter = 0
            class Loader:
                sampler = EpochOrderSampler(2, 42)
                generator = torch.Generator()
                def __len__(self): return 1
            trainer.train_dataloader = Loader(); trainer.val_dataloader = Loader()
            def train_epoch(epoch):
                trainer.global_step += 1
                return {'loss': 1.0}
            trainer.train_epoch = train_epoch
            evaluations = iter([{'continuous_f1': 0.0, 'val_loss': v} for v in [3.0,2.0,1.0]])
            trainer.evaluate = lambda: next(evaluations)
            saved = []
            trainer.save_checkpoint = lambda epoch, metrics, is_best: saved.append((epoch,is_best))
            trainer.train()
            self.assertEqual(saved, [(1,True),(2,False),(3,False)])
        self.assertFalse(select_best(0.4,0.4))
        with self.assertRaises(ValueError): select_best(float('nan'),0.0)

    def test_duplicate_votes_and_wrong_order_rejected(self):
        q={'Aspect':'food','Category':'FOOD#QUALITY','Opinion':'good','VA':'7#5'}
        validate_seed_predictions([{'ID':'1','Quadruplet':[q]}], ['1'])
        with self.assertRaises(ValueError): validate_seed_predictions([{'ID':'1','Quadruplet':[q,q]}], ['1'])
        with self.assertRaises(ValueError): validate_seed_predictions([{'ID':'1','Quadruplet':[q]}], ['2'])

    def test_all_configs_match_frozen_run_design(self):
        design=json.loads(Path('docs/audits/icsai_phase1/experiment_matrix.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            snapshot = Path(temp)
            (snapshot / 'config.json').write_text(json.dumps({'hidden_size': 768}))
            with patch('scripts.icsai.control_runtime.pinned_snapshot', return_value=snapshot):
                for run in design['experiments']:
                    cfg=config_for_run(run['id'])
                    self.assertEqual(cfg['seed'],run['seed'])
                    self.assertEqual(cfg['icsai']['arm'],run['arm'])
                    self.assertNotIn('test_file',cfg['data'])
                    self.assertEqual(cfg['training']['save_top_k'],1)
                    self.assertEqual(cfg['model']['rethink']['enabled'],run['arm']!='A')

if __name__=='__main__': unittest.main()
