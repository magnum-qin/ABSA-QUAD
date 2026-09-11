"""Post-review diagnostics on frozen checkpoints/predictions; never trains.

Stages uses all 27 core models and their original development inputs. Forward
hooks observe the unmodified inference graph. Final quads must exactly replay
the archived development predictions. Analysis outputs live separately from the
original protocol and do not alter its model-selection or hypothesis family.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'docs/audits/icsai_revision_v3'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def spans(labels):
    """Mirror greedy five-tag extraction, including its orphan-I semantics."""
    found, current = set(), None
    for i, label in enumerate(labels):
        if label in (1, 3):
            if current is not None:
                found.add(current)
            current = (i, i, 'aspect' if label == 1 else 'opinion')
        elif label in (2, 4):
            if current is not None:
                kind = 'aspect' if label == 2 else 'opinion'
                if current[2] == kind:
                    current = (current[0], i, kind)
                else:
                    found.add(current)
                    current = None
        else:
            if current is not None:
                found.add(current)
                current = None
    if current is not None:
        found.add(current)
    return found


def label_counts(gold, pred):
    assert len(gold) == len(pred)
    gs, ps = spans(gold), spans(pred)
    return dict(tokens=len(gold), correct=sum(g == p for g, p in zip(gold, pred)),
                positive_gold=sum(g != 0 for g in gold),
                positive_pred=sum(p != 0 for p in pred),
                positive_correct=sum(g == p and g != 0 for g, p in zip(gold, pred)),
                gold_spans=len(gs), pred_spans=len(ps), matched_spans=len(gs & ps))


def label_metrics(counts):
    def ratio(n, d):
        return n / d if d else 0.0
    return dict(counts=counts,
                bio_accuracy=ratio(counts['correct'], counts['tokens']),
                bio_non_o_micro_f1=ratio(2 * counts['positive_correct'], counts['positive_gold'] + counts['positive_pred']),
                span_precision=ratio(counts['matched_spans'], counts['pred_spans']),
                span_recall=ratio(counts['matched_spans'], counts['gold_spans']),
                span_f1=ratio(2 * counts['matched_spans'], counts['gold_spans'] + counts['pred_spans']))


def stage_summary(records):
    stages = list(records[0]['stages'])
    result = {}
    for stage in stages:
        per_row = [label_counts(r['gold'], r['stages'][stage]) for r in records]
        counts = {k: sum(c[k] for c in per_row) for k in per_row[0]}
        result[stage] = label_metrics(counts)
    return result


def stages(device):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    from transformers import AutoTokenizer
    from scripts.icsai.run_control import make_loader, verified_config
    from src.models.icsai_control import load_control_checkpoint
    torch.set_num_threads(4)
    lock_path = ROOT / 'docs/audits/icsai_core_test/protocol_lock.json'
    lock = read(lock_path)
    inputs = []
    for r in lock['models']:
        cfg, _ = verified_config(r['id'], device)
        dev = ROOT / cfg['data']['dev_file']
        pred = ROOT / cfg['icsai']['prediction_dir'] / 'dev.jsonl'
        inputs.append(dict(run_id=r['id'], dataset=r['dataset'], arm=r['arm'], seed=int(r['seed']),
                           checkpoint=r['checkpoint'], checkpoint_sha256=r['checkpoint_sha256'],
                           dev=str(dev.relative_to(ROOT)), dev_sha256=sha(dev),
                           final_prediction=str(pred.relative_to(ROOT)), final_prediction_sha256=sha(pred)))
    spec = dict(schema='icsai-post-review-stage-diagnostic/1',
                scope='All 27 original core development runs; original best checkpoints with EMA; descriptive post-review analysis, no new training or selection.',
                stages={'A': ['draft'], 'B': ['draft', 'refine1', 'final'], 'C': ['draft', 'refine1', 'final']},
                metrics='Micro counts per complete development split, then mean and sample SD across seeds. BIO accuracy and exact-tag non-O micro F1; typed exact token-span P/R/F1 from the deployed greedy decoder. All attention-mask=1 positions including CLS/SEP; padding excluded. Gold is the existing five-way BIO projection, not a complete overlapping character-span annotation.',
                replay='Forward hooks on span heads only; final model.predict outputs must equal original development predictions exactly.',
                script_sha256=sha(Path(__file__)), original_protocol_sha256=sha(lock_path), inputs=inputs)
    protocol = OUT / 'stage_protocol.json'
    if protocol.exists():
        assert read(protocol) == spec, 'Diagnostic specification changed; use a new analysis version.'
    else:
        write(protocol, spec)  # Frozen before the first forward pass.
    results = []
    for inp in inputs:
        dest = OUT / 'stages' / (inp['run_id'] + '.json')
        trace_path = dest.with_suffix('.jsonl')
        if dest.exists():
            result = read(dest)
            assert result['protocol_sha256'] == sha(protocol)
            assert result['trace_sha256'] == sha(trace_path)
            assert result['stages'] == stage_summary(rows(trace_path))
            results.append(result)
            print('verified completed', inp['run_id'], flush=True)
            continue
        write(OUT / 'stage_status.json', dict(stage='running', pid=os.getpid(), run_id=inp['run_id'], completed=len(results)))
        print('loading', inp['run_id'], flush=True)
        checkpoint = ROOT / inp['checkpoint']
        assert sha(checkpoint) == inp['checkpoint_sha256']
        assert sha(ROOT / inp['dev']) == inp['dev_sha256']
        assert sha(ROOT / inp['final_prediction']) == inp['final_prediction_sha256']
        model, cfg = load_control_checkpoint(checkpoint, device)
        tokenizer = AutoTokenizer.from_pretrained(cfg['model']['encoder_name'], local_files_only=True)
        loader = make_loader(cfg, tokenizer, 'dev')
        captured = []
        def capture(module, args, output):
            captured.append(output.detach().cpu())
        handles = [model.span_extractor.register_forward_hook(capture)]
        if model.use_rethink:
            handles.append(model.span_extractor_stage2.register_forward_hook(capture))
        trace, predictions = [], []
        with torch.no_grad():
            for batch in loader:
                captured.clear()
                quads = model.predict(batch['input_ids'].to(device), batch['attention_mask'].to(device),
                                      batch['texts'], batch['offset_mapping'], pair_threshold=0.3)
                stage_names = spec['stages'][inp['arm']]
                assert len(captured) == len(stage_names)
                decoded = [model.span_extractor.extract_spans(logits, batch['attention_mask']) for logits in captured]
                for i, ident in enumerate(batch['sample_ids']):
                    n = int(batch['attention_mask'][i].sum())
                    stage_labels = {name: logits[i, :n].argmax(-1).tolist() for name, logits in zip(stage_names, captured)}
                    for name, actual in zip(stage_names, decoded):
                        assert spans(stage_labels[name]) == set(actual[i]), 'Metric decoder differs from deployed decoder'
                    trace.append(dict(ID=ident, gold=batch['bio_labels'][i, :n].tolist(), stages=stage_labels))
                    predictions.append(dict(ID=ident, Quadruplet=quads[i]))
        assert predictions == rows(ROOT / inp['final_prediction']), 'Final predictions did not replay'
        for handle in handles:
            handle.remove()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in trace))
        result = dict(**inp, rows=len(trace), stages=stage_summary(trace),
                      final_prediction_replay=True, trace_sha256=sha(trace_path), protocol_sha256=sha(protocol))
        write(dest, result)
        results.append(result)
        print('complete', inp['run_id'], {k: round(v['span_f1'], 4) for k, v in result['stages'].items()}, flush=True)
        del model, loader, captured, decoded
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    summary = []
    for dataset in dict.fromkeys(r['dataset'] for r in inputs):
        for arm in 'ABC':
            runs = [r for r in results if r['dataset'] == dataset and r['arm'] == arm]
            assert len(runs) == 3
            for stage in spec['stages'][arm]:
                entry = dict(dataset=dataset, arm=arm, stage=stage, seeds=[r['seed'] for r in runs])
                for metric in ('bio_accuracy', 'bio_non_o_micro_f1', 'span_precision', 'span_recall', 'span_f1'):
                    values = [r['stages'][stage][metric] for r in runs]
                    entry[metric] = dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values))
                summary.append(entry)
    write(OUT / 'stage_summary.json', dict(protocol_sha256=sha(protocol), runs=results, summary=summary))
    write(OUT / 'stage_status.json', dict(stage='complete', pid=os.getpid(), completed=len(results), time=datetime.now(timezone.utc).isoformat()))


def implicit(q):
    return any(q[k].lower() == 'null' for k in ('Aspect', 'Opinion'))


def existing():
    from scripts.kd.tallip_common import evaluate_records, official_row_counts
    core_path = ROOT / 'docs/audits/icsai_core_test/test_statistics.json'
    ext_path = ROOT / 'docs/audits/icsai_extension_test/test_statistics.json'
    core, ext = read(core_path), read(ext_path)
    records = [r for r in ext['single_models'] + ext['ensembles'] if r['dataset'] == 'jpn_hotel']
    gold_path = ROOT / 'data/official_test_gold/DimABSA2026/bdc93be1224106ae7d3eb95739c02a76ed4ae8a1/jpn/jpn_hotel_test_task3.jsonl'
    gold = rows(gold_path)
    expected = next(r for r in ext['sources'] if r['dataset'] == 'jpn_hotel')['gold_sha256']
    assert sha(gold_path) == expected
    protocol = dict(scope='Post-review descriptive partition of existing fixed Japanese hotel test predictions; no training, inference, selection or significance test.',
                    partitions='Sentence: any gold NULL aspect/opinion versus none, retain all quads and predictions in selected rows. Quad: partition gold and predictions separately by their own NULL status, retains all rows.',
                    script_sha256=sha(Path(__file__)), gold_sha256=expected,
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (core_path, ext_path)})
    write(OUT / 'existing_protocol.json', protocol)
    results = []
    sentence_counts = {k: 0 for k in ('implicit', 'explicit')}
    quad_counts = {k: 0 for k in ('implicit', 'explicit', 'both_null')}
    for g in gold:
        sentence_counts['implicit' if any(implicit(q) for q in g['Quadruplet']) else 'explicit'] += 1
        for q in g['Quadruplet']:
            quad_counts['implicit' if implicit(q) else 'explicit'] += 1
            quad_counts['both_null'] += all(q[k].lower() == 'null' for k in ('Aspect', 'Opinion'))
    for rec in records:
        pred_path = ROOT / rec['path']
        assert sha(pred_path) == rec['sha256']
        pred = rows(pred_path)
        assert [r['ID'] for r in gold] == [r['ID'] for r in pred]
        overall = evaluate_records(gold, pred)
        assert abs(overall['cF1'] - rec['metrics']['cF1']) < 1e-12
        for unit in ('sentence', 'quad'):
            for is_implicit in (False, True):
                if unit == 'sentence':
                    indices = [i for i, g in enumerate(gold) if any(implicit(q) for q in g['Quadruplet']) == is_implicit]
                    gs, ps = [gold[i] for i in indices], [pred[i] for i in indices]
                else:
                    gs = [dict(ID=g['ID'], Quadruplet=[q for q in g['Quadruplet'] if implicit(q) == is_implicit]) for g in gold]
                    ps = [dict(ID=p['ID'], Quadruplet=[q for q in p['Quadruplet'] if implicit(q) == is_implicit]) for p in pred]
                metrics = evaluate_records(gs, ps)
                metrics.pop('_row_counts', None)
                results.append(dict(run_id=rec['run_id'], seed=rec['seed'], kind=rec['kind'], unit=unit,
                                    stratum='implicit' if is_implicit else 'explicit', rows=len(gs),
                                    gold_quads=sum(len(g['Quadruplet']) for g in gs),
                                    pred_quads=sum(len(p['Quadruplet']) for p in ps), metrics=metrics,
                                    prediction_sha256=rec['sha256']))
    # Row-count conservation is checked independently of aggregate F1 averaging.
    for rec in records:
        for unit in ('sentence', 'quad'):
            parts = [r for r in results if r['run_id'] == rec['run_id'] and r['unit'] == unit]
            assert sum(r['gold_quads'] for r in parts) == sum(len(g['Quadruplet']) for g in gold)
            assert sum(r['pred_quads'] for r in parts) == sum(len(p['Quadruplet']) for p in rows(ROOT / rec['path']))
    write(OUT / 'implicit_results.json', dict(protocol=protocol, sentence_counts=sentence_counts,
                                            gold_quad_counts=quad_counts, results=results))
    contrasts = []
    for r in core['contrasts']:
        contrasts.append(dict(dataset=r['dataset'], contrast=r['contrast'],
                              paired_seed_delta_pp=dict(zip(['42', '123', '456'], [100 * d for d in r['paired_seed_deltas']])),
                              mean_delta_pp=100*r['delta'], holm_p=r['holm_p']))
    write(OUT / 'seed_contrasts.json', dict(source_sha256=sha(core_path), contrasts=contrasts))
    print('Existing-prediction partitions complete', sentence_counts, quad_counts, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stages', 'existing'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'stages':
        stages(args.device)
    else:
        existing()
