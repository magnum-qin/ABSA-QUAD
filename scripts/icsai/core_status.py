"""Read live queue/process state and the last completed epoch without modifying it."""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/audits/icsai_core_training'


def process(pid, expected):
    path=Path(f'/proc/{pid}/cmdline')
    try: cmd=path.read_bytes().replace(b'\0',b' ').decode()
    except (FileNotFoundError, ProcessLookupError): return False
    return expected in cmd


def main():
    status=json.loads((OUT/'status.json').read_text())
    if status['state']=='running':
        status['queue_live']=process(status['queue_pid'],'scripts/icsai/core_queue.py')
        status['child_live']=process(status['child_pid'],'scripts/icsai/run_control.py')
        run=status['current_run']
        p=ROOT/'checkpoints/icsai_phase1'/run/'epochs.jsonl'
        if p.exists():
            epoch=json.loads(p.read_text().splitlines()[-1])
            status['last_completed_epoch']=epoch['epoch']
            status['last_epoch_seconds']=epoch['wall_seconds']
            status['last_dev_cf1']=epoch['dev_metrics']['continuous_f1']
            status['global_step']=epoch['global_step']
        log=OUT/f"{run}.{status['phase']}.log"
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-16384))
                tail=f.read().decode(errors='replace')
            matches=re.findall(r'Epoch (\d+):\s+(\d+)%',tail)
            if matches: status['last_logged_training_progress']={'epoch':int(matches[-1][0]),'percent':int(matches[-1][1])}
        if not status['queue_live'] or not status['child_live']:
            status['process_note']='A process was absent/mismatched at read time; inspect status/log transition before any restart.'
    progress=OUT/'progress.json'
    if progress.exists():
        p=json.loads(progress.read_text())
        status['rough_remaining_training_hours']=p['rough_remaining_training_seconds']/3600
        status['eta_caveat']=p['eta_caveat']
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
