"""Locked core test pipeline: all inference, prediction freeze, then all scoring."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.icsai.freeze_core_test import digest
OUT=ROOT/'docs/audits/icsai_extension_test'
LOCK=OUT/'protocol_lock.json'
PRED=ROOT/'predictions/icsai_extension_test'
GOLDROOT=ROOT/'data/official_test_gold/DimABSA2026/bdc93be1224106ae7d3eb95739c02a76ed4ae8a1'

def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def write(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)

def status(stage,**kwargs):
    write(OUT/'status.json',dict(stage=stage,pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),**kwargs))
    print(stage,kwargs,flush=True)

def validate(rows,ids):
    from src.models.icsai_control import validate_seed_predictions
    validate_seed_predictions(rows,ids)
    for row in rows:
        for q in row['Quadruplet']:
            assert all(1 <= float(v) <= 9 for v in q['VA'].split('#'))

def ensemble(seeds):
    ids=[r['ID'] for r in seeds[0]]
    for rows in seeds: validate(rows,ids)
    result=[]
    for index,ident in enumerate(ids):
        votes=defaultdict(list)
        for rows in seeds:
            for q in rows[index]['Quadruplet']:
                votes[(q['Aspect'],q['Category'],q['Opinion'])].append(tuple(map(float,q['VA'].split('#'))))
        quads=[]
        for (aspect,category,opinion),values in votes.items():
            if len(values)>=2:
                v=sum(x[0] for x in values)/len(values);a=sum(x[1] for x in values)/len(values)
                quads.append(dict(Aspect=aspect,Category=category,Opinion=opinion,VA=f'{v:.2f}#{a:.2f}'))
        result.append(dict(ID=ident,Quadruplet=quads))
    validate(result,ids)
    return result

def save_rows(path,rows):
    with path.open('x') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')

def infer(lock):
    import torch
    from transformers import AutoTokenizer
    from src.models.icsai_control import load_control_checkpoint
    from scripts.icsai.run_control import make_loader
    from scripts.icsai.control_runtime import seed_all
    PRED.mkdir(exist_ok=True,parents=True)
    records=[]
    for index,r in enumerate(lock['models']):
        status('verifying_checkpoint',completed=index,total=15,run_id=r['id'])
        path=ROOT/r['checkpoint'];stat=path.stat()
        assert digest(path)==r['checkpoint_sha256'],r['id']
        assert digest(path.parent/'resolved_config.json')==r['config_sha256']
        status('predicting',completed=index,total=15,run_id=r['id'])
        seed_all(int(r['seed']))
        start=time.monotonic()
        model,cfg=load_control_checkpoint(path,'cuda')
        assert cfg['icsai']['arm']==r['arm'] and cfg['seed']==int(r['seed'])
        assert (path.stat().st_size,path.stat().st_mtime_ns)==(stat.st_size,stat.st_mtime_ns)
        source=next(t for t in lock['test_inputs'] if t['dataset']==r['dataset'])
        assert digest(ROOT/source['path'])==source['sha256']
        inputs=read(ROOT/source['path']);ids=[x['ID'] for x in inputs]
        cfg['data']['test_file']=source['path']
        tok=AutoTokenizer.from_pretrained(cfg['model']['encoder_name'],local_files_only=True)
        loader=make_loader(cfg,tok,'test')
        load_seconds=time.monotonic()-start
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        rows=[]
        with torch.no_grad():
            for b in loader:
                q=model.predict(b['input_ids'].to('cuda'),b['attention_mask'].to('cuda'),b['texts'],b['offset_mapping'],pair_threshold=.3)
                rows.extend(dict(ID=i,Quadruplet=v) for i,v in zip(b['sample_ids'],q))
        torch.cuda.synchronize();elapsed=time.monotonic()-start
        validate(rows,ids)
        dest=PRED/(r['id']+'.jsonl');save_rows(dest,rows)
        records.append(dict(run_id=r['id'],dataset=r['dataset'],arm=r['arm'],seed=int(r['seed']),kind='single',path=str(dest.relative_to(ROOT)),sha256=digest(dest),rows=len(rows),inference_seconds=elapsed,load_seconds=load_seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),timing_scope='one pass including first batch, transfers, decoding; excludes model/tokenizer loading; descriptive, not repeated benchmark'))
        write(OUT/'prediction_progress.json',records)
        del model,loader,tok;gc.collect();torch.cuda.empty_cache()
    for domain in lock['datasets']:
        for arm in 'A':
            members=[next(r for r in records if r['dataset']==domain and r['arm']==arm and r['seed']==s) for s in (42,123,456)]
            rows=ensemble([read(ROOT/r['path']) for r in members]);dest=PRED/f'{domain}_{arm}_ensemble.jsonl';save_rows(dest,rows)
            records.append(dict(run_id=dest.stem,dataset=domain,arm=arm,seed=None,kind='ensemble',path=str(dest.relative_to(ROOT)),sha256=digest(dest),rows=len(rows),members=[r['run_id'] for r in members]))
    frozen=dict(frozen_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=digest(LOCK),predictions=records)
    assert len(records)==20
    with (OUT/'predictions.lock.json').open('x') as f:json.dump(frozen,f,indent=2)
    return frozen

def checked_metrics(goldpath,predpath,tag):
    import re,subprocess
    from scripts.kd.tallip_common import evaluate_records
    gold=read(goldpath);pred=read(predpath);validate(pred,[r['ID'] for r in gold])
    m=evaluate_records(gold,pred);m.pop('_row_counts')
    output=subprocess.check_output([sys.executable,str(GOLDROOT/'evaluation_script/metrics_subtask_1_2_3.py'),'-t','3','-p',str(predpath),'-g',str(goldpath)],text=True)
    (OUT/'official_raw'/f'{tag}.txt').write_text(output)
    values=re.findall(r"['\"]cF1['\"]:\s*([0-9.eE+-]+)",output) or re.findall(r'cF1:\s*([0-9.eE+-]+)',output)
    assert values and abs(float(values[-1])-m['cF1'])<1e-10
    errors=[]
    for g,p in zip(gold,pred):
        for q in g['Quadruplet']:
            matches=[z for z in p['Quadruplet'] if all(z[k].lower()==q[k].lower() for k in ('Aspect','Category','Opinion'))]
            if len(matches)==1:
                a=list(map(float,q['VA'].split('#')));b=list(map(float,matches[0]['VA'].split('#')))
                errors.append([abs(a[i]-b[i]) for i in (0,1)])
    m['matched_v_mae']=sum(e[0] for e in errors)/len(errors) if errors else None
    m['matched_a_mae']=sum(e[1] for e in errors)/len(errors) if errors else None
    m['matched_quad_count']=len(errors)
    n=sum(len(g['Quadruplet']) for g in gold)
    m['matched_gold_coverage']=len(errors)/n if n else None
    return m

def prepare():
    import csv
    from scripts.icsai.verify_control import verify_readiness
    from src.evaluation.metrics import DimASQPEvaluator
    verify_readiness()
    designpath=ROOT/'docs/audits/icsai_phase1/experiment_matrix.json';design=json.loads(designpath.read_text())
    runs=[r for r in design['experiments'] if r['priority']=='uniform_extension']
    inputs={r['dataset']:r for r in json.loads((ROOT/'docs/audits/icsai_phase1/gold_input_manifest.json').read_text())['datasets']}
    complete=list(csv.DictReader((ROOT/'docs/audits/icsai_extension_training/completed_runs.csv').open()))
    assert len(runs)==len(complete)==15 and {r['id'] for r in runs}=={r['id'] for r in complete}
    dev=[]
    for r in runs:
        status('checking_dev',completed=len(dev),total=15,run_id=r['id'])
        source=inputs[r['dataset']];goldpath=ROOT/source['dev_file'];predpath=ROOT/r['prediction_dir']/'dev.jsonl'
        assert digest(goldpath)==source['dev_sha256']
        m=checked_metrics(goldpath,predpath,'dev_'+r['id'])
        ev=DimASQPEvaluator();ev.add_batch([x['Quadruplet'] for x in read(predpath)],[x['Quadruplet'] for x in read(goldpath)])
        local=ev.compute_continuous_f1()['continuous_f1']
        st=json.loads((ROOT/r['checkpoint_dir']/'status.json').read_text());assert st['status']=='complete' and abs(local-st['best_dev_cf1'])<1e-10
        dev.append(dict(run_id=r['id'],dataset=r['dataset'],seed=r['seed'],local_cf1=local,official_metrics=m,prediction_sha256=digest(predpath),gold_sha256=digest(goldpath)))
    write(OUT/'dev_statistics.json',dict(single_models=dev,official_crosschecks=15,selection='Keep original local-dev best checkpoint; no reselection under official scoring.'))
    models=[]
    for row in complete:
        status('freezing_models',completed=len(models),total=15,run_id=row['id'])
        base=ROOT/'checkpoints/icsai_phase1'/row['id'];path=base/'checkpoint_best.pt';before=path.stat();h=digest(path);after=path.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        models.append(dict(row,checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=h,checkpoint_bytes=after.st_size,config_sha256=digest(base/'resolved_config.json')))
        write(OUT/'freeze_progress.json',dict(completed=len(models),total=15))
    domains=list(dict.fromkeys(r['dataset'] for r in runs));tests=[]
    for d in domains:
        s=inputs[d];path=ROOT/s['test_input'];assert digest(path)==s['test_input_sha256']
        ids=[r['ID'] for r in read(path)];assert len(ids)==len(set(ids))
        tests.append(dict(dataset=d,path=s['test_input'],sha256=digest(path),rows=len(ids)))
    sources=['scripts/icsai/extension_test.py','scripts/icsai/core_test.py','scripts/kd/tallip_common.py','src/models/icsai_control.py','scripts/icsai/run_control.py','docs/audits/icsai_extension_test/dev_statistics.json']
    lock=dict(frozen_at=datetime.now(timezone.utc).isoformat(),design_sha256=digest(designpath),datasets=domains,models=models,test_inputs=tests,inference=design['inference'],test_exposure=design['test_exposure'],statistics='Each seed, mean and sample SD; fixed three-seed vote-2 ensembles secondary. Equal-domain eight-domain macro descriptive only. No new B/C tests or significance claims.',source_sha256={p:digest(ROOT/p) for p in sources},evaluator_sha256=digest(GOLDROOT/'evaluation_script/metrics_subtask_1_2_3.py'))
    with LOCK.open('x') as f:json.dump(lock,f,indent=2)
    return lock

def score(lock,frozen):
    import statistics
    assert digest(LOCK)==frozen['protocol_sha256']
    for p,h in lock['source_sha256'].items():assert digest(ROOT/p)==h,p
    assert digest(GOLDROOT/'evaluation_script/metrics_subtask_1_2_3.py')==lock['evaluator_sha256']
    for r in frozen['predictions']:assert digest(ROOT/r['path'])==r['sha256']
    singles=[];ensembles=[];sources=[]
    for domain in lock['datasets']:
        goldpath=GOLDROOT/domain.split('_')[0]/f'{domain}_test_task3.jsonl';gold=read(goldpath)
        inp=read(ROOT/next(t['path'] for t in lock['test_inputs'] if t['dataset']==domain))
        assert [(r['ID'],r['Text']) for r in gold]==[(r['ID'],r['Text']) for r in inp]
        implicit=lambda q:any(q[k].lower()=='null' for k in ('Aspect','Opinion'))
        sources.append(dict(dataset=domain,gold_sha256=digest(goldpath),rows=len(gold),implicit_rows=sum(any(implicit(q) for q in r['Quadruplet']) for r in gold),explicit_only_rows=sum(not any(implicit(q) for q in r['Quadruplet']) for r in gold)))
        for rec in [r for r in frozen['predictions'] if r['dataset']==domain]:
            m=checked_metrics(goldpath,ROOT/rec['path'],'test_'+rec['run_id'])
            (singles if rec['kind']=='single' else ensembles).append(dict(rec,metrics=m))
    report=dict(as_of=datetime.now(timezone.utc).isoformat(),single_models=singles,ensembles=ensembles,sources=sources,official_crosschecks=20,protocol_sha256=digest(LOCK),prediction_lock_sha256=digest(OUT/'predictions.lock.json'),assessment='Share with caveats: post-test-exposure fixed supplementary evaluation; no tuning or model selection on test.')
    write(OUT/'test_statistics.json',report)
    corepath=ROOT/'docs/audits/icsai_core_test/test_statistics.json';core=json.loads(corepath.read_text());corelockpath=ROOT/'docs/audits/icsai_core_test/predictions.lock.json'
    assert digest(corelockpath)==core['prediction_lock_sha256']
    for r in json.loads(corelockpath.read_text())['predictions']:assert digest(ROOT/r['path'])==r['sha256']
    allsingle=[r for r in core['single_models'] if r['arm']=='A']+singles
    allens=[r for r in core['ensembles'] if r['arm']=='A']+ensembles
    assert len(allsingle)==24 and len(allens)==8
    summary=[]
    for d in sorted({r['dataset'] for r in allsingle}):
        rows=[r for r in allsingle if r['dataset']==d];assert {r['seed'] for r in rows}=={42,123,456}
        scores=[r['metrics']['cF1'] for r in rows];ens=next(r for r in allens if r['dataset']==d)
        summary.append(dict(dataset=d,mean=statistics.mean(scores),sample_sd=statistics.stdev(scores),ensemble_cf1=ens['metrics']['cF1']))
    macro=dict(single_mean=statistics.mean(r['mean'] for r in summary),ensemble_mean=statistics.mean(r['ensemble_cf1'] for r in summary),definition='Equal weight to each of eight fixed datasets; descriptive, no macro significance test.')
    write(OUT/'eight_domain_summary.json',dict(summary=summary,macro=macro,single_models=allsingle,ensembles=allens,source_sha256={str(corepath.relative_to(ROOT)):digest(corepath),'docs/audits/icsai_extension_test/test_statistics.json':digest(OUT/'test_statistics.json')}))
    lines=['# ICSAI 八域 A 单阶段模型 test 结果','',report['assessment'],'','| 数据集 | 三 seed cF1 均值±SD | 固定三 seed 集成 cF1 |','|---|---:|---:|']
    for r in summary:lines.append(f"| {r['dataset']} | {r['mean']:.4f} ± {r['sample_sd']:.4f} | {r['ensemble_cf1']:.4f} |")
    lines += ['',f"八域等权宏平均：单模型 {macro['single_mean']:.4f}；集成 {macro['ensemble_mean']:.4f}。仅作描述性汇总。",'', '历史 test 已暴露，本轮不是盲测。保留原本地 dev 选定的 checkpoint。扩展五域仅评估 A，不能据此扩展三域 Rethink 消融结论。集成有额外推理成本；单次资源记录不能替代重复基准。','', '逐 seed、离散 F1、匹配 V/A MAE、匹配覆盖率与数据来源见 eight_domain_summary.json；本轮 dev 复核见 dev_statistics.json。']
    (OUT/'eight_domain_summary.md').write_text('\n'.join(lines)+'\n')
    status('complete',single_models=15,ensembles=5,dev_crosschecks=15,test_crosschecks=20,eight_domain_single_models=24)

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'official_raw').mkdir(exist_ok=True)
    with (OUT/'pipeline_claim.json').open('x') as f:json.dump(dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat()),f)
    try:
        lock=prepare();frozen=infer(lock);status('scoring',predictions_frozen=20);score(lock,frozen)
    except BaseException as exc:
        status('failed',error=repr(exc));traceback.print_exc();raise

if __name__=='__main__':main()
