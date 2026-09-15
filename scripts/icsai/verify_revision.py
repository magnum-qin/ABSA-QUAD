"""Verify v3 records; --local-assets additionally checks original prediction data.

Default mode is standard-library-only and works inside the source archive.
The local-assets mode requires the exact external data, original predictions,
and scipy for the official scorer. It does not load model weights or train.
"""
import argparse
import ast
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.icsai.revision_diagnostics import OUT, implicit, read, rows, sha, stage_summary, write


def cf1(gold, pred):
    """Independent direct calculation of the manuscript's continuous credit."""
    assert [r['ID'] for r in gold] == [r['ID'] for r in pred]
    credit = 0.0
    denominator = 0
    for g, p in zip(gold, pred):
        gs, ps = g['Quadruplet'], p['Quadruplet']
        denominator += len(gs) + len(ps)
        for q in gs:
            matches = [v for v in ps if all(q[k].lower() == v[k].lower() for k in ('Aspect', 'Category', 'Opinion'))]
            if len(matches) == 1:
                truth = list(map(float, q['VA'].split('#')))
                guess = list(map(float, matches[0]['VA'].split('#')))
                credit += max(0, 1 - math.dist(truth, guess) / math.sqrt(128))
    return 2 * credit / denominator if denominator else 0.0


def close(a, b):
    assert abs(a - b) < 1e-12, (a, b)


def record_checks():
    plan = read(OUT / 'revision_plan.json')
    for path, expected in plan['protected_original_sha256'].items():
        assert sha(ROOT / path) == expected, f'Protected original changed: {path}'
    protocol = read(OUT / 'stage_protocol.json')
    assert sha(ROOT / 'scripts/icsai/revision_diagnostics.py') == protocol['script_sha256']
    report = read(OUT / 'stage_summary.json')
    assert report['protocol_sha256'] == sha(OUT / 'stage_protocol.json')
    assert len(report['runs']) == 27
    expected_runs = {r['run_id']: r for r in protocol['inputs']}
    assert len(expected_runs) == 27
    assert {r['run_id'] for r in report['runs']} == set(expected_runs)
    for run in report['runs']:
        trace_path = OUT / 'stages' / (run['run_id'] + '.jsonl')
        assert sha(trace_path) == run['trace_sha256']
        assert read(trace_path.with_suffix('.json')) == run
        for key, value in expected_runs[run['run_id']].items():
            assert run[key] == value
        trace = rows(trace_path)
        assert len(trace) == run['rows']
        assert len({r['ID'] for r in trace}) == len(trace)
        assert all(list(r['stages']) == protocol['stages'][run['arm']] for r in trace)
        assert stage_summary(trace) == run['stages']
        assert run['final_prediction_replay'] is True
    assert len(report['summary']) == 21
    for item in report['summary']:
        group = [r for r in report['runs'] if r['dataset'] == item['dataset'] and r['arm'] == item['arm']]
        assert sorted(r['seed'] for r in group) == [42, 123, 456]
        for metric in ('bio_accuracy', 'bio_non_o_micro_f1', 'span_precision', 'span_recall', 'span_f1'):
            values = [r['stages'][item['stage']][metric] for r in group]
            close(statistics.mean(values), item[metric]['mean'])
            close(statistics.stdev(values), item[metric]['sample_sd'])
    core = read(ROOT / 'docs/audits/icsai_core_test/test_statistics.json')
    paired = read(OUT / 'seed_contrasts.json')
    assert paired['source_sha256'] == sha(ROOT / 'docs/audits/icsai_core_test/test_statistics.json')
    assert len(paired['contrasts']) == len(core['contrasts']) == 6
    last = 0.0
    for index, c in enumerate(sorted(core['contrasts'], key=lambda r: r['p'])):
        last = max(last, min(1, c['p'] * (6 - index)))
        close(last, c['holm_p'])
    for original, added in zip(core['contrasts'], paired['contrasts']):
        assert (original['dataset'], original['contrast']) == (added['dataset'], added['contrast'])
        for seed, delta in zip(('42', '123', '456'), original['paired_seed_deltas']):
            close(added['paired_seed_delta_pp'][seed], 100 * delta)
        close(added['mean_delta_pp'], 100 * statistics.mean(original['paired_seed_deltas']))
    partitions = read(OUT / 'implicit_results.json')
    assert len(partitions['results']) == 16
    for path, expected in partitions['protocol']['source_sha256'].items():
        assert sha(ROOT / path) == expected
    assert partitions['sentence_counts'] == {'implicit': 152, 'explicit': 648}
    assert partitions['gold_quad_counts'] == {'implicit': 175, 'explicit': 1268, 'both_null': 0}
    main = (ROOT / 'docs/paper_icsai_v3.tex').read_text()
    cited = {key for group in re.findall(r'\\cite\{([^}]+)\}', main) for key in group.split(',')}
    bibliography = set(re.findall(r'\\bibitem\{([^}]+)\}', main))
    assert cited == bibliography and len(cited) == 13
    assert 'qin2026leader' not in cited
    for name in ('paper_icsai_v3', 'paper_icsai_v3_supplement'):
        tex = (ROOT / 'docs' / (name + '.tex')).read_text()
        for path in re.findall(r'\\input\{([^}]+)\}', tex):
            assert (ROOT / 'docs' / path).is_file()
    return dict(protected_originals=len(plan['protected_original_sha256']),
                stage_runs=27, stage_summary_rows=21, all_stage_traces_recomputed=True,
                six_holm_values_recomputed=True, seed_differences_recomputed=18,
                citation_count=13, implicit_partition_records=16)


def local_checks():
    core = read(ROOT / 'docs/audits/icsai_core_test/test_statistics.json')
    ext = read(ROOT / 'docs/audits/icsai_extension_test/test_statistics.json')
    gold_root = ROOT / 'data/official_test_gold/DimABSA2026/bdc93be1224106ae7d3eb95739c02a76ed4ae8a1'
    golds = {}
    checked = 0
    max_error = 0
    for source in (core, ext):
        for s in source['sources']:
            ds = s['dataset']
            path = gold_root / ds.split('_')[0] / (ds + '_test_task3.jsonl')
            assert sha(path) == s.get('gold_sha256', s.get('sha256'))
            golds[ds] = rows(path)
        for r in source['single_models'] + source['ensembles']:
            path = ROOT / r['path']
            assert sha(path) == r['sha256']
            error = abs(cf1(golds[r['dataset']], rows(path)) - r['metrics']['cF1'])
            assert error < 1e-12
            max_error = max(max_error, error)
            checked += 1
    assert checked == 56
    for r in read(OUT / 'stage_protocol.json')['inputs']:
        assert sha(ROOT / r['final_prediction']) == r['final_prediction_sha256']
        assert sha(ROOT / r['dev']) == r['dev_sha256']
        assert [x['ID'] for x in rows(ROOT / r['dev'])] == [x['ID'] for x in rows(OUT / 'stages' / (r['run_id'] + '.jsonl'))]
    evaluator = gold_root / 'evaluation_script/metrics_subtask_1_2_3.py'
    assert sha(evaluator) == core['evaluator_sha256']
    official_records = []
    gold = golds['jpn_hotel']
    for record in read(OUT / 'implicit_results.json')['results']:
        original = next(r for r in ext['single_models'] + ext['ensembles'] if r['run_id'] == record['run_id'])
        prediction = rows(ROOT / original['path'])
        flag = record['stratum'] == 'implicit'
        if record['unit'] == 'sentence':
            indices = [i for i, r in enumerate(gold) if any(implicit(q) for q in r['Quadruplet']) == flag]
            gs, ps = [gold[i] for i in indices], [prediction[i] for i in indices]
        else:
            gs = [dict(ID=r['ID'], Text=r.get('Text', ''), Quadruplet=[q for q in r['Quadruplet'] if implicit(q) == flag]) for r in gold]
            ps = [dict(ID=r['ID'], Quadruplet=[q for q in r['Quadruplet'] if implicit(q) == flag]) for r in prediction]
        assert len(gs) == record['rows']
        assert sum(len(r['Quadruplet']) for r in gs) == record['gold_quads']
        assert sum(len(r['Quadruplet']) for r in ps) == record['pred_quads']
        close(cf1(gs, ps), record['metrics']['cF1'])
        with tempfile.TemporaryDirectory(prefix='icsai-slice-') as temp:
            gpath, ppath = Path(temp) / 'gold.jsonl', Path(temp) / 'pred.jsonl'
            for path, data in ((gpath, gs), (ppath, ps)):
                path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data))
            output = subprocess.check_output([sys.executable, str(evaluator), '-t', '3', '-g', str(gpath), '-p', str(ppath)], text=True)
        parsed = ast.literal_eval(output.split('Final Results: ')[1].strip())
        official_cf1 = parsed['cF1'] if 'cF1' in parsed else parsed['continuous_f1']
        close(official_cf1, record['metrics']['cF1'])
        official_records.append(dict(run_id=record['run_id'], unit=record['unit'], stratum=record['stratum'], result=parsed))
    return dict(test_prediction_files=checked, max_cf1_absolute_error=max_error,
                original_dev_prediction_hashes=27, official_slice_checks=official_records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-assets', action='store_true')
    args = parser.parse_args()
    report = dict(record_checks=record_checks(), local_assets='not_requested')
    if args.local_assets:
        report['local_assets'] = local_checks()
    path = OUT / ('numeric_validation_local.json' if args.local_assets else 'numeric_validation_records.json')
    write(path, report)
    print(json.dumps(report, indent=2))
