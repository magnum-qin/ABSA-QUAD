"""CPU-only causal path probe; no pretrained downloads, training, or dataset reads."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from src.models.dimabsa_model import DimASQPModel

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 16)
    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

def main():
    torch.manual_seed(20260907)
    torch.set_num_threads(1)
    with patch('src.models.dimabsa_model.AutoModel.from_pretrained', return_value=Encoder()), patch('src.models.dimabsa_model.AutoConfig.from_pretrained', return_value=SimpleNamespace(hidden_size=16)):
        model = DimASQPModel(encoder_name='synthetic-offline', use_rethink=True, num_rethink_iterations=2, dropout=0.0)
    model.eval()
    # Hold decoded spans fixed to guarantee a nonempty pair and isolate feedback.
    fixed_spans = lambda logits, mask: [[(1, 1, 'aspect'), (3, 3, 'opinion')] for _ in range(logits.shape[0])]
    model.span_extractor.extract_spans = fixed_spans
    model.span_extractor_stage2.extract_spans = fixed_spans
    args = dict(input_ids=torch.tensor([[1,2,3,4,5,6]]), attention_mask=torch.ones(1,6,dtype=torch.long), return_loss=False)
    keys = ['bio_logits','pair_logits','entity_logits','attribute_logits','valence','arousal']
    calls = []
    h = model.rethink_module.register_forward_hook(lambda *_: calls.append(1))
    ref = model(**args)
    h.remove()
    def max_delta(out):
        return max((out[k]-ref[k]).abs().max().item() for k in keys)
    def perturb_pair_inputs(module, positional, kw):
        kw = dict(kw)
        for k in ('pair_repr','pair_valid_probs','entity_probs','attribute_probs','valence','arousal'):
            kw[k] = kw[k] * 0 + 17
        return positional, kw
    h = model.rethink_module.register_forward_pre_hook(perturb_pair_inputs, with_kwargs=True)
    pair_input_delta = max_delta(model(**args)); h.remove()
    h = model.rethink_module.register_forward_hook(lambda m,a,o: (o[0], o[1]*0+12345))
    pair_output_delta = max_delta(model(**args)); h.remove()
    def perturb_bio(module, positional, kw):
        kw = dict(kw)
        kw['bio_probs'] = torch.zeros_like(kw['bio_probs'])
        kw['bio_probs'][...,0] = 1
        return positional, kw
    h = model.rethink_module.register_forward_pre_hook(perturb_bio, with_kwargs=True)
    bio_delta = max_delta(model(**args)); h.remove()
    model.zero_grad(set_to_none=True)
    out = model(**args)
    sum(out[k].square().sum() for k in keys).backward()
    grads = {n: None if p.grad is None else p.grad.abs().sum().item() for n,p in model.rethink_module.label_embedding.named_parameters()}
    inactive = ('entity_embedding','attribute_embedding','pair_embedding','va_projector','pair_fusion')
    passed = len(calls)==2 and pair_input_delta==0 and pair_output_delta==0 and bio_delta>0 and all(v is None for n,v in grads.items() if n.startswith(inactive)) and grads['bio_embedding.weight']>0
    report = dict(scope='Current full model with synthetic encoder and fixed nonempty spans; inference feedback and final-output gradient only, not accuracy or historical checkpoint provenance.', refinement_calls=len(calls), pair_input_intervention_max_delta=pair_input_delta, discarded_pair_output_intervention_max_delta=pair_output_delta, bio_intervention_max_delta=bio_delta, label_embedding_gradient_l1=grads, passed=passed, torch_version=torch.__version__, source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/models/dimabsa_model.py','src/models/feature_rethink.py','scripts/icsai/probe_rethink_path.py']})
    dest = ROOT/'docs/audits/icsai_phase1/rethink_probe.json'
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not passed: raise SystemExit(1)

if __name__=='__main__': main()
