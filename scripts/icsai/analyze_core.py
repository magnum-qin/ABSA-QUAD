"""Reconcile all core dev predictions, then run frozen paired seed-mean bootstrap."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.kd.tallip_common import official_row_counts
from src.evaluation.metrics import DimASQPEvaluator

OUT=ROOT/'docs/audits/icsai_core_analysis'
EVALUATOR=ROOT/'data/official_test_gold/DimABSA2026/bdc93be1224106ae7d3eb95739c02a76ed4ae8a1/evaluation_script/metrics_subtask_1_2_3.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def official_check(args):
    rid,pred,gold,score=args
    output=subprocess.check_output([sys.executable,str(EVALUATOR),'-t','3','-p',str(pred),'-g',str(gold)],text=True)
    (OUT/'official_dev_raw'/f'{rid}.txt').write_text(output)
    matches=re.findall(r"['\"]cF1['\"]:\s*([0-9.eE+-]+)",output)
    if not matches:matches=re.findall(r'cF1:\s*([0-9.eE+-]+)',output)
    if not matches:raise RuntimeError(output)
    actual=float(matches[-1]);assert abs(actual-score)<1e-10,(rid,score,actual)
    return rid,actual


def cf1(counts):
    denom=counts[...,1]+counts[...,2]
    return np.divide(2*counts[...,0],denom,out=np.zeros_like(denom),where=denom>0)


def holm(ps):
    order=np.argsort(ps);result=np.zeros(len(ps));running=0.0
    for rank,idx in enumerate(order):
        running=max(running,min(1.0,(len(ps)-rank)*ps[idx]));result[idx]=running
    return result.tolist()


def main():
    OUT.mkdir(exist_ok=True);(OUT/'official_dev_raw').mkdir(exist_ok=True)
    design=json.loads((ROOT/'docs/audits/icsai_phase1/experiment_matrix.json').read_text())
    inputs=json.loads((ROOT/'docs/audits/icsai_phase1/gold_input_manifest.json').read_text())['datasets']
    runs=[r for r in design['experiments'] if r['priority']=='core']
    seeds=design['training']['seeds']; domains=design['statistics']['primary_datasets'];arms=['A','B','C']
    arrays={};summaries=[];checks=[];tasks=[];single_rows=[]
    rng=np.random.default_rng(20260907);bootstrap={};contrasts=[]
    for domain in domains:
        source=next(x for x in inputs if x['dataset']==domain);gold_path=ROOT/source['dev_file']
        assert sha(gold_path)==source['dev_sha256']
        gold=read(gold_path);ids=[r['ID'] for r in gold];assert len(ids)==len(set(ids))
        domain_counts=[]
        for arm in arms:
            arm_counts=[];scores=[]
            for seed in seeds:
                run=next(r for r in runs if (r['dataset'],r['arm'],r['seed'])==(domain,arm,seed))
                pred_path=ROOT/run['prediction_dir']/'dev.jsonl';pred=read(pred_path)
                assert [r['ID'] for r in pred]==ids
                for row in pred:
                    keys=[(q['Aspect'],q['Category'],q['Opinion']) for q in row['Quadruplet']]
                    assert len(keys)==len(set(keys)),run['id']
                    for q in row['Quadruplet']:assert all(1<=float(v)<=9 for v in q['VA'].split('#'))
                counts=np.array([[official_row_counts(g['Quadruplet'],p['Quadruplet'])['ctp'],len(g['Quadruplet']),len(p['Quadruplet'])] for g,p in zip(gold,pred)],dtype=float)
                score=float(cf1(counts.sum(axis=0)));scores.append(score);arm_counts.append(counts)
                ev=DimASQPEvaluator();ev.add_batch([r['Quadruplet'] for r in pred],[r['Quadruplet'] for r in gold]);local=ev.compute_continuous_f1()['continuous_f1']
                status=json.loads((ROOT/run['checkpoint_dir']/'status.json').read_text());assert status['status']=='complete'
                assert abs(local-status['best_dev_cf1'])<1e-10,(run['id'],local,status['best_dev_cf1'])
                single_rows.append({'run_id':run['id'],'dataset':domain,'arm':arm,'seed':seed,'dev_rows':len(gold),'local_dev_cf1':local,'official_dev_cf1':score,'official_minus_local':score-local,'prediction':str(pred_path.relative_to(ROOT)),'prediction_sha256':sha(pred_path),'gold':str(gold_path.relative_to(ROOT)),'gold_sha256':sha(gold_path)})
                tasks.append((run['id'],pred_path,gold_path,score))
            domain_counts.append(arm_counts)
            summaries.append({'dataset':domain,'arm':arm,'official_dev_cf1_mean':float(np.mean(scores)),'sample_sd':float(np.std(scores,ddof=1)),'seed_scores':dict(zip(map(str,seeds),scores))})
        arr=np.asarray(domain_counts);arrays[domain]=arr
        n=len(gold);indices=rng.integers(0,n,size=(10000,n));weights=np.zeros((10000,n),dtype=np.int32)
        for i,index in enumerate(indices): weights[i]=np.bincount(index,minlength=n)
        # Same sampled examples across every arm and training seed.
        totals=np.einsum('rn,asnk->rask',weights,arr,optimize=True)
        boot=cf1(totals).mean(axis=2);observed=cf1(arr.sum(axis=2)).mean(axis=1)
        bootstrap[domain]=boot
        # Independent slow recomputation on one resample checks aggregation order.
        index=indices[0]
        for ai in range(3):
            direct=np.mean([float(cf1(arr[ai,si,index,:].sum(axis=0))) for si in range(3)])
            assert abs(direct-boot[0,ai])<1e-12
        for other in ['A','C']:
            bi=1;oi=arms.index(other);delta=float(observed[bi]-observed[oi]);dist=boot[:,bi]-boot[:,oi]
            # Null-centered, two-sided bootstrap test with Monte Carlo correction.
            p=(1+np.count_nonzero(np.abs(dist-delta)>=abs(delta)))/(len(dist)+1)
            seed_delta=cf1(arr[bi].sum(axis=1))-cf1(arr[oi].sum(axis=1))
            contrasts.append({'dataset':domain,'contrast':f'B-{other}','delta':delta,'ci95':np.quantile(dist,[.025,.975]).tolist(),'bootstrap_p_two_sided_centered':float(p),'paired_seed_deltas':dict(zip(map(str,seeds),seed_delta.tolist())),'paired_seed_delta_sd':float(np.std(seed_delta,ddof=1))})
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        checks=list(pool.map(official_check,tasks))
    adjusted=holm([r['bootstrap_p_two_sided_centered'] for r in contrasts])
    for r,p in zip(contrasts,adjusted):r['holm_p_six_contrasts']=p;r['holm_significant_005']=p<.05
    macro=[]
    for other,oi in [('A',0),('C',2)]:
        dist=np.mean([bootstrap[d][:,1]-bootstrap[d][:,oi] for d in domains],axis=0)
        obs=float(np.mean([r['delta'] for r in contrasts if r['contrast']==f'B-{other}']))
        macro.append({'contrast':f'B-{other}','delta':obs,'ci95':np.quantile(dist,[.025,.975]).tolist(),'scope':'Descriptive equal-domain macro over fixed three domains only'})
    report={'as_of':datetime.now(timezone.utc).isoformat(),'assessment':'Share with caveats: dev selected checkpoints, post-test-exposure protocol; not independent test confirmation.','protocol':{'iterations':10000,'seed':20260907,'resampling':'Paired examples shared across all seeds/arms within each domain; seeds held fixed, then mean per-seed cF1. Independent domain resampling for descriptive macro.','p_definition':'(1 + count(abs(bootstrap_delta-observed_delta) >= abs(observed_delta)))/(10000+1); centered two-sided bootstrap. Holm family=6 dev contrasts.','ci':'Unadjusted percentile 95%; sample uncertainty conditional on three trained seeds. Not training-procedure uncertainty.','checkpoint_selection':'Existing local dev best, earliest tie. Do not reselect epochs under official scorer.'},'official_evaluator':str(EVALUATOR.relative_to(ROOT)),'official_evaluator_sha256':sha(EVALUATOR),'official_cli_crosschecks':len(checks),'local_score_reconciliation_count':len(single_rows),'single_models':single_rows,'summary':summaries,'contrasts':contrasts,'macro_descriptive':macro}
    (OUT/'dev_statistics.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# ICSAI 核心实验：配对 dev 统计','',f"计算日期：{report['as_of']}。27 份预测与训练日志复核通过，逐份与官方 evaluator 对照通过。",'', '## 三 seed 单模型均值（官方 dev cF1）','', '| 数据集 | A 均值±SD | B 均值±SD | C 均值±SD |','|---|---:|---:|---:|']
    for d in domains:
        row=[next(r for r in summaries if r['dataset']==d and r['arm']==a) for a in arms]
        lines.append('| '+d+' | '+' | '.join(f"{r['official_dev_cf1_mean']:.4f} ± {r['sample_sd']:.4f}" for r in row)+' |')
    lines+=['','## 六项预定对照','', '| 数据集 | 对照 | ΔcF1（百分点） | 未校正95% CI（百分点） | Holm p |','|---|---|---:|---:|---:|']
    for r in contrasts:lines.append(f"| {r['dataset']} | {r['contrast']} | {100*r['delta']:+.2f} | [{100*r['ci95'][0]:+.2f}, {100*r['ci95'][1]:+.2f}] | {r['holm_p_six_contrasts']:.4f} |")
    lines+=['','评估判定：Share with caveats。这是 dev 选模后的描述与条件推断；不能替代后续固定 test 对照。bootstrap 只重采样句子，不能替代训练 seed 方差或整个调参流程的不确定性。CI 未作多重校正，与 Holm p 分开报告。','', '官方与本地 scorer 的最高单模型绝对差值：'+f"{max(abs(r['official_minus_local']) for r in single_rows):.12f}"+'。不按重新评分结果更换 checkpoint。','', '全部逐 seed 差值、p 值定义与宏平均见 [dev_statistics.json](dev_statistics.json)。未拒绝零差异不等于统计等效；三域结果不能外推到全部八域。']
    (OUT/'dev_statistics.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'contrasts':contrasts,'macro':macro},indent=2))

if __name__=='__main__':main()
