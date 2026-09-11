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
OUT=ROOT/'docs/audits/icsai_core_test'
LOCK=OUT/'protocol_lock.json'
PRED=ROOT/'predictions/icsai_core_test'
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
        status('verifying_checkpoint',completed=index,total=27,run_id=r['id'])
        path=ROOT/r['checkpoint'];stat=path.stat()
        assert digest(path)==r['checkpoint_sha256'],r['id']
        assert digest(path.parent/'resolved_config.json')==r['config_sha256']
        status('predicting',completed=index,total=27,run_id=r['id'])
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
    for domain in lock['statistics']['primary_datasets']:
        for arm in 'ABC':
            members=[next(r for r in records if r['dataset']==domain and r['arm']==arm and r['seed']==s) for s in (42,123,456)]
            rows=ensemble([read(ROOT/r['path']) for r in members]);dest=PRED/f'{domain}_{arm}_ensemble.jsonl';save_rows(dest,rows)
            records.append(dict(run_id=dest.stem,dataset=domain,arm=arm,seed=None,kind='ensemble',path=str(dest.relative_to(ROOT)),sha256=digest(dest),rows=len(rows),members=[r['run_id'] for r in members]))
    frozen=dict(frozen_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=digest(LOCK),predictions=records)
    assert len(records)==36
    with (OUT/'predictions.lock.json').open('x') as f:json.dump(frozen,f,indent=2)
    return frozen

def score(lock,frozen):
    import numpy as np
    from scripts.icsai.analyze_core import cf1,holm
    from scripts.kd.tallip_common import official_row_counts,evaluate_records
    import re,subprocess
    assert digest(LOCK)==frozen['protocol_sha256']
    for r in frozen['predictions']:assert digest(ROOT/r['path'])==r['sha256']
    evaluator=GOLDROOT/'evaluation_script/metrics_subtask_1_2_3.py'
    (OUT/'official_raw').mkdir(exist_ok=True)
    rng=np.random.default_rng(20260907);summaries=[];contrasts=[];single=[];ensembles=[];sources=[]
    for domain in lock['statistics']['primary_datasets']:
        goldpath=GOLDROOT/domain.split('_')[0]/f'{domain}_test_task3.jsonl';gold=read(goldpath)
        ids=[r['ID'] for r in gold];assert len(ids)==len(set(ids))
        inp=read(ROOT/next(t['path'] for t in lock['test_inputs'] if t['dataset']==domain))
        assert [(x['ID'],x['Text']) for x in gold]==[(x['ID'],x['Text']) for x in inp]
        implicit=lambda q:any(str(q[k]).lower()=='null' for k in ('Aspect','Opinion'))
        sources.append(dict(dataset=domain,gold=str(goldpath.relative_to(ROOT)),sha256=digest(goldpath),rows=len(gold),implicit_rows=sum(any(implicit(q) for q in g['Quadruplet']) for g in gold),explicit_only_rows=sum(not any(implicit(q) for q in g['Quadruplet']) for g in gold),implicit_quads=sum(implicit(q) for g in gold for q in g['Quadruplet']),explicit_quads=sum(not implicit(q) for g in gold for q in g['Quadruplet'])))
        arrays=[]
        for arm in 'ABC':
            counts=[];scores=[]
            for seed in (42,123,456,None):
                rec=next(r for r in frozen['predictions'] if (r['dataset'],r['arm'],r['seed'])==(domain,arm,seed))
                pred=read(ROOT/rec['path']);validate(pred,ids)
                metrics=evaluate_records(gold,pred);metrics.pop('_row_counts')
                # V and A errors are conditional on exactly one categorical match, as in the official metric.
                errors=[]
                for g,p in zip(gold,pred):
                    for q in g['Quadruplet']:
                        match=[z for z in p['Quadruplet'] if all(z[k].lower()==q[k].lower() for k in ('Aspect','Opinion','Category'))]
                        if len(match)==1:
                            va=list(map(float,q['VA'].split('#')));pv=list(map(float,match[0]['VA'].split('#')))
                            errors.append([abs(va[i]-pv[i]) for i in (0,1)])
                metrics.update(matched_v_mae=float(np.mean(errors,axis=0)[0]) if errors else None,matched_a_mae=float(np.mean(errors,axis=0)[1]) if errors else None,matched_quad_count=len(errors),matched_gold_coverage=len(errors)/sum(len(g['Quadruplet']) for g in gold))
                output=subprocess.check_output([sys.executable,str(evaluator),'-t','3','-p',str(ROOT/rec['path']),'-g',str(goldpath)],text=True)
                (OUT/'official_raw'/f"{rec['run_id']}.txt").write_text(output)
                values=re.findall(r"['\"]cF1['\"]:\s*([0-9.eE+-]+)",output) or re.findall(r'cF1:\s*([0-9.eE+-]+)',output)
                assert values and abs(float(values[-1])-metrics['cF1'])<1e-10
                row=dict(rec,metrics=metrics)
                if seed is None:ensembles.append(row);continue
                single.append(row);scores.append(metrics['cF1'])
                counts.append([[official_row_counts(g['Quadruplet'],p['Quadruplet'])['ctp'],len(g['Quadruplet']),len(p['Quadruplet'])] for g,p in zip(gold,pred)])
            arrays.append(counts);summaries.append(dict(dataset=domain,arm=arm,mean=float(np.mean(scores)),sample_sd=float(np.std(scores,ddof=1))))
        arr=np.asarray(arrays,float);n=len(gold)
        indices=rng.integers(0,n,size=(10000,n));weights=np.array([np.bincount(i,minlength=n) for i in indices],dtype=np.int32)
        boot=cf1(np.einsum('rn,asnk->rask',weights,arr,optimize=True)).mean(axis=2)
        obs=cf1(arr.sum(axis=2)).mean(axis=1)
        for ai in range(3):assert abs(np.mean([cf1(arr[ai,si,indices[0],:].sum(axis=0)) for si in range(3)])-boot[0,ai])<1e-12
        for other,oi in [('A',0),('C',2)]:
            delta=float(obs[1]-obs[oi]);dist=boot[:,1]-boot[:,oi]
            contrasts.append(dict(dataset=domain,contrast='B-'+other,delta=delta,ci95=np.quantile(dist,[.025,.975]).tolist(),p=float((1+np.count_nonzero(abs(dist-delta)>=abs(delta)))/10001),paired_seed_deltas=(cf1(arr[1].sum(axis=1))-cf1(arr[oi].sum(axis=1))).tolist()))
    for r,p in zip(contrasts,holm([r['p'] for r in contrasts])):r['holm_p']=p
    report=dict(as_of=datetime.now(timezone.utc).isoformat(),assessment='Share with caveats: post-test-exposure supplementary evaluation; fixed trained seeds; three domains only.',protocol_sha256=digest(LOCK),prediction_lock_sha256=digest(OUT/'predictions.lock.json'),evaluator_sha256=digest(evaluator),official_crosschecks=36,sources=sources,summary=summaries,contrasts=contrasts,single_models=single,ensembles=ensembles,statistics=lock['statistics'],p_value_definition=lock['p_value_definition'])
    write(OUT/'test_statistics.json',report)
    lines=['# ICSAI 固定核心 test 对照','',report['assessment'],'','全部27个单模型与9个固定投票集成完成，逐份官方评分交叉核验通过。','', '| 数据集 | A cF1±SD | B cF1±SD | C cF1±SD |','|---|---:|---:|---:|']
    for d in lock['statistics']['primary_datasets']:
        rows=[next(r for r in summaries if r['dataset']==d and r['arm']==a) for a in 'ABC']
        lines.append('| '+d+' | '+' | '.join(f"{r['mean']:.4f} ± {r['sample_sd']:.4f}" for r in rows)+' |')
    lines+=['','| 数据集 | 对照 | 差值（百分点） | 未校正95% CI | Holm p |','|---|---|---:|---:|---:|']
    for r in contrasts:lines.append(f"| {r['dataset']} | {r['contrast']} | {100*r['delta']:+.2f} | [{100*r['ci95'][0]:+.2f}, {100*r['ci95'][1]:+.2f}] | {r['holm_p']:.4f} |")
    lines+=['','CI 仅表示在固定三个训练 seed 下的样本重采样不确定性，不覆盖整个训练过程；不显著不等于等效。历史 test 已暴露，不能称为盲测。三域结果不能外推到八域。','', '逐 seed、集成、离散抽取 F1、匹配 V/A MAE 与覆盖率、显式/隐式样本计数和单次推理资源记录见 [test_statistics.json](test_statistics.json)。计时包含首批，尚不能代替重复性能基准。']
    (OUT/'test_statistics.md').write_text('\n'.join(lines)+'\n')
    status('complete',single_models=27,ensembles=9,official_crosschecks=36)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-freeze-pid',type=int);args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    with (OUT/'pipeline_claim.json').open('x') as f:json.dump(dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat()),f)
    try:
        status('waiting_for_model_freeze')
        while not LOCK.exists():
            if args.wait_for_freeze_pid:
                os.kill(args.wait_for_freeze_pid,0)
            else:raise FileNotFoundError(LOCK)
            time.sleep(10)
        lock=json.loads(LOCK.read_text())
        for p,h in lock['source_sha256'].items():assert digest(ROOT/p)==h,p
        sources=['scripts/icsai/core_test.py','scripts/kd/tallip_common.py','scripts/icsai/analyze_core.py']
        write(OUT/'execution_lock.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=digest(LOCK),source_sha256={p:digest(ROOT/p) for p in sources}))
        frozen=infer(lock)
        status('scoring',predictions_frozen=36)
        score(lock,frozen)
    except BaseException as exc:
        status('failed',error=repr(exc));traceback.print_exc();raise

if __name__=='__main__':main()
