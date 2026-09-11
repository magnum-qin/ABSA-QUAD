"""Inventory ICONIP assets without training, test scoring, or gold-test reads."""
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.audit_best_reproduction import TASKS, SEEDS
from src.evaluation.metrics import DimASQPEvaluator
import yaml

def info(path, hash_bytes=True):
    p=Path(path); p=p if p.is_absolute() else ROOT/p
    d={'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'exists':p.is_file()}
    if d['exists']:
        d['bytes']=p.stat().st_size
        if hash_bytes: d['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    return d

def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

encoders=['microsoft/deberta-v3-large','microsoft/deberta-v3-base','ku-nlp/deberta-v2-large-japanese','xlm-roberta-large','xlm-roberta-large','microsoft/mdeberta-v3-base','hfl/chinese-roberta-wwm-ext-large','hfl/chinese-roberta-wwm-ext-large']
configs=['configs/pipeline_eng_restaurant_v16.yaml','configs/pipeline_eng_laptop_v5repro.yaml','configs/pipeline_jpn_v16.yaml','configs/pipeline_rus_v15.yaml','configs/pipeline_tat_v15.yaml','configs/pipeline_ukr_v13b.yaml','configs/pipeline_zho_restaurant_v16.yaml','configs/pipeline_zho_laptop_v16.yaml']
lock=json.loads((ROOT/'docs/audits/tallip_kd/official_test/protocol_lock.json').read_text())
all_preds=sorted((ROOT/'predictions').rglob('*.jsonl'))
report={'as_of':datetime.now(timezone.utc).isoformat(),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'scope':'Inventory of local ICONIP best-dev recipe references. Prediction scores below are dev-only diagnostics, not new system selection. Filename, same score or row count alone does not prove provenance. No test labels read. Checkpoint sizes are not content hashes.','seeds':SEEDS,'source_files':[info(p) for p in ['docs/paper_iconip_current.tex','scripts/audit_best_reproduction.py','scripts/run_final_best_submission_4090.sh','src/models/dimabsa_model.py','src/models/feature_rethink.py','scripts/inference.py','scripts/train.py','docs/audits/tallip_kd/official_test/protocol_lock.json']],'datasets':[]}
for spec,encoder,cfg in zip(TASKS,encoders,configs):
    d=asdict(spec); d['encoder_reference']=encoder
    d['historical_config']=info(spec.config); d['retained_config_candidate']=info(cfg)
    if d['retained_config_candidate']['exists']:
        conf=yaml.safe_load((ROOT/cfg).read_text()); d['retained_config_candidate']['model']=conf['model']; d['retained_config_candidate']['train_file']=conf['data']['train_file']
    d['train']=info(spec.train_file)
    if d['train']['exists']: d['train']['rows']=len(rows(ROOT/spec.train_file))
    d['dev']=info(spec.dev_file); gold=rows(ROOT/spec.dev_file); gm={r['ID']:r for r in gold}; d['dev']['rows']=len(gold)
    d['test_input']=info(spec.test_file); test=rows(ROOT/spec.test_file); tm={r['ID']:r for r in test}; d['test_input']['rows']=len(test)
    d['historical_checkpoints']=[info(f'checkpoints/{spec.checkpoint_prefix}{s}/checkpoint_best.pt',False) for s in SEEDS]
    lang=spec.name.split('_')[0]
    d['other_checkpoint_candidates']=[info(p,False) for p in sorted((ROOT/'checkpoints').glob('*/checkpoint_best.pt')) if (spec.name in str(p) or (lang not in ['eng','zho'] and f'_{lang}_' in str(p))) and any(str(p.parent.name).startswith(v) for v in ['v5','v13','v15','v16','final_']) and not p.parent.name.startswith(spec.checkpoint_prefix)]
    preds=[]
    for p in all_preds:
        if not (spec.name in str(p) or (lang not in ['eng','zho'] and f'_{lang}_' in p.name)): continue
        rr=rows(p); ids=[r.get('ID') for r in rr]
        if len(ids)!=len(gm) or len(set(ids))!=len(ids) or set(ids)!=set(gm): continue
        q=info(p); q['rows']=len(rr)
        q['text_matches_where_present']=all('Text' not in r or r['Text']==gm[r['ID']]['Text'] for r in rr)
        if not q['text_matches_where_present']: continue
        ev=DimASQPEvaluator(); ev.add_batch([r.get('Quadruplet',[]) for r in rr],[gm[r['ID']].get('Quadruplet',[]) for r in rr]); q['local_dev_cf1']=ev.compute_continuous_f1()['continuous_f1']; q['matches_paper_rounded_score']=round(q['local_dev_cf1'],4)==round(spec.expected_dev_cf1,4)
        preds.append(q)
    d['dev_prediction_candidates']=preds
    d['test_prediction_candidates']=[]
    for p in sorted((ROOT/'submission/final').glob(f'{spec.name}_test*.jsonl')):
        rr=rows(p); ids=[r.get('ID') for r in rr]; q=info(p); q['rows']=len(rr); q['ids_match_test']=len(ids)==len(tm) and len(set(ids))==len(ids) and set(ids)==set(tm)
        q['existing_20260718_lock_systems']=[k for k,v in lock['datasets'][spec.name]['systems'].items() if v.get('sha256')==q['sha256']]
        d['test_prediction_candidates'].append(q)
    report['datasets'].append(d)
p=ROOT/'docs/audits/icsai_phase1/inventory.json'; p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
for d in report['datasets']:
    matches=[x['path'] for x in d['dev_prediction_candidates'] if x['matches_paper_rounded_score']]
    print(d['name'],'train',d['train'].get('rows'),'historical checkpoints',sum(x['exists'] for x in d['historical_checkpoints']),'dev candidates',len(d['dev_prediction_candidates']),'matching-score files',matches)
