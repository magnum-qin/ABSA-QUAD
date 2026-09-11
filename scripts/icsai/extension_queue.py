#!/usr/bin/env python3
"""Run the frozen 15-cell uniform extension serially, auditing each run and exporting dev only."""
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'docs/audits/icsai_extension_training'
OUT.mkdir(parents=True, exist_ok=True)


def now(): return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def event(kind, **kwargs):
    row = {'time': now(), 'event': kind, **kwargs}
    with (OUT / 'events.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def audit_run(run, input_rows):
    import torch
    folder = ROOT / run['checkpoint_dir']
    status = json.loads((folder / 'status.json').read_text())
    epochs = [json.loads(l) for l in (folder / 'epochs.jsonl').read_text().splitlines()]
    assert status['status'] == 'complete' and len(epochs) == 30
    assert [r['epoch'] for r in epochs] == list(range(1,31))
    expected_updates = math.ceil(input_rows / 8)
    assert all(r['updates'] == expected_updates for r in epochs)
    assert status['global_step'] == 30 * expected_updates
    best_epoch = max(epochs, key=lambda r: r['dev_metrics']['continuous_f1'])['epoch']
    for filename, expected_epoch in [('checkpoint_best.pt', best_epoch), ('checkpoint_latest.pt', 30)]:
        ck = torch.load(folder / filename, map_location='cpu', mmap=True, weights_only=False)
        assert ck['epoch'] == expected_epoch
        assert ck['config']['icsai']['run_id'] == run['id']
        assert ck['config']['icsai']['arm'] == run['arm']
        if filename == 'checkpoint_latest.pt': assert ck['global_step'] == status['global_step']
        assert ck.get('ema_state_dict', {}).get('shadow')
        del ck
    # All completed arms for a paired dataset/seed must see identical epoch order.
    for arm in 'ABC':
        other = folder.parent / f"{run['dataset']}_{arm}_s{run['seed']}" / 'epochs.jsonl'
        if other.exists():
            other_epochs = [json.loads(l) for l in other.read_text().splitlines()]
            for a,b in zip(epochs,other_epochs):
                assert (a['epoch'],a['order_sha256'],a['updates']) == (b['epoch'],b['order_sha256'],b['updates'])
    return {'best_epoch': best_epoch, 'best_dev_cf1': status['best_dev_cf1'],
            'global_step': status['global_step'], 'epochs': 30,
            'epoch_wall_seconds': sum(e['wall_seconds'] for e in epochs)}


def summarize(completed):
    groups = {}
    for row in completed: groups.setdefault((row['dataset'],row['arm']),[]).append(row)
    summary=[]
    for (dataset,arm), values in sorted(groups.items()):
        if len(values) != 3: continue
        scores=[v['best_dev_cf1'] for v in values]
        summary.append({'dataset':dataset,'arm':arm,'n_seeds':3,'dev_cf1_mean':statistics.mean(scores),
                        'dev_cf1_sample_sd':statistics.stdev(scores)})
    write_json(OUT / 'dev_summary.json', {'scope':'Dev-selected single-model results only; no test scoring or significance claims.', 'groups':summary})
    with (OUT/'completed_runs.csv').open('w') as f:
        fields=['id','dataset','arm','seed','best_epoch','best_dev_cf1','global_step','train_wall_seconds','prediction_wall_seconds']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(completed)


def main():
    os.chdir(ROOT)
    lockfile = (OUT / 'queue.lock').open('a')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT/'queue_manifest.json').exists():
        raise RuntimeError('A queue manifest already exists; reconcile its actual process/run state before resuming')
    from scripts.icsai.verify_control import verify_readiness
    verify_readiness()
    design_path=ROOT/'docs/audits/icsai_phase1/experiment_matrix.json'
    design=json.loads(design_path.read_text())
    inputs=json.loads((ROOT/'docs/audits/icsai_phase1/gold_input_manifest.json').read_text())
    sizes={r['dataset']:r['kept_rows'] for r in inputs['datasets']}
    by_id={r['id']:r for r in design['experiments'] if r['priority']=='uniform_extension'}
    domains=list(dict.fromkeys(r['dataset'] for r in by_id.values()))
    order=[f'{d}_A_s{seed}' for seed in design['training']['seeds'] for d in domains]
    assert len(order)==15 and set(order)==set(by_id)
    for run in by_id.values():
        if (ROOT/run['checkpoint_dir']/'run_claim.json').exists():
            raise RuntimeError(f"Existing run claim: {run['id']}; no automatic restart")
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1')
    manifest={'created_at':now(),'pid':os.getpid(),'ordered_run_ids':order,'ordering':'seed, dataset in frozen extension design order; A only',
              'design_sha256':hashlib.sha256(design_path.read_bytes()).hexdigest(),
              'implementation_lock_sha256':hashlib.sha256((ROOT/'docs/audits/icsai_phase2/implementation.lock.json').read_bytes()).hexdigest(),
              'queue_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'environment':{k:env[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','TOKENIZERS_PARALLELISM','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','CUDA_VISIBLE_DEVICES']},
              'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip(),
              'test_policy':'No test prediction/scoring. After each audited training run, export its best-EMA dev predictions only.'}
    write_json(OUT/'queue_manifest.json',manifest)
    completed=[]; started=time.monotonic(); current=None; phase=None
    try:
        for run_id in order:
            current=run_id; run=by_id[run_id]; phase='train'
            entry={'id':run_id,'dataset':run['dataset'],'arm':run['arm'],'seed':run['seed']}
            for action in ['train','predict-dev']:
                phase=action
                log=OUT/f'{run_id}.{action}.log'
                command=[sys.executable,'-u','scripts/icsai/run_control.py',action,'--run-id',run_id,'--device','cuda']
                with log.open('x') as f:
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                    event('phase_started',run_id=run_id,phase=phase,pid=child.pid,log=str(log.relative_to(ROOT)))
                    write_json(OUT/'status.json',{'state':'running','queue_pid':os.getpid(),'child_pid':child.pid,'current_run':run_id,'phase':phase,'completed_runs':len(completed),'total_runs':15,'updated_at':now()})
                    phase_started=time.monotonic()
                    code=child.wait()
                if code: raise RuntimeError(f'{run_id} {action} exited {code}; inspect {log}')
                entry['train_wall_seconds' if action=='train' else 'prediction_wall_seconds']=time.monotonic()-phase_started
                if action=='train': entry.update(audit_run(run,sizes[run['dataset']]))
                event('phase_completed',run_id=run_id,phase=phase)
            completed.append(entry); summarize(completed)
            # Work-normalized ETA: use measured epoch seconds per optimizer update,
            # without claiming unobserved B/C or language runtimes are identical.
            sec_per_update=sum(r['epoch_wall_seconds'] for r in completed)/sum(r['global_step'] for r in completed)
            remaining=order[len(completed):]
            remaining_updates=sum(30*math.ceil(sizes[by_id[r]['dataset']]/8) for r in remaining)
            estimate=sec_per_update*remaining_updates
            write_json(OUT/'progress.json',{'updated_at':now(),'completed':completed,'remaining_run_ids':remaining,
                       'elapsed_seconds':time.monotonic()-started,'rough_remaining_training_seconds':estimate,
                       'eta_caveat':'Based on observed optimizer-update costs; unobserved arms/domains, loading, inference, disk writes and input lengths may change it.'})
            event('run_completed',run_id=run_id,completed_runs=len(completed),rough_remaining_training_hours=estimate/3600)
        write_json(OUT/'status.json',{'state':'complete','completed_runs':15,'total_runs':15,'finished_at':now()})
        event('queue_completed',completed_runs=15)
    except BaseException as exc:
        write_json(OUT/'status.json',{'state':'stopped_on_error','current_run':current,'phase':phase,'completed_runs':len(completed),'error':repr(exc),'updated_at':now()})
        event('queue_stopped',run_id=current,error=repr(exc))
        raise


if __name__=='__main__': main()
