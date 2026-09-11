"""Fixed-input repeated cost measurements, without scoring or training."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import sys,json,time,gc,statistics,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from src.models.icsai_control import load_control_checkpoint,validate_seed_predictions
from src.data.dataset import DimASQPDataset,collate_fn
from scripts.icsai.core_test import ensemble
from scripts.icsai.freeze_core_test import digest
OUT=ROOT/'docs/audits/icsai_cost'
def write(name,x):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def status(stage,**kw):
 x=dict(stage=stage,pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),**kw);write('status.json',x);print(x,flush=True)
def sync():torch.cuda.synchronize()
def run(model,batches):
 rows=[]
 with torch.no_grad():
  for b in batches:
   q=model.predict(b['input_ids'].to('cuda'),b['attention_mask'].to('cuda'),b['texts'],b['offset_mapping'],pair_threshold=.3)
   rows.extend(dict(ID=i,Quadruplet=v) for i,v in zip(b['sample_ids'],q))
 return rows

def main():
 with (OUT/'claim.json').open('x') as f:json.dump(dict(pid=os.getpid()),f)
 lockpath=ROOT/'docs/audits/icsai_core_test/protocol_lock.json';lock=json.loads(lockpath.read_text())
 source=ROOT/'data/subtask_3/eng/eng_restaurant_dev_task3.jsonl'
 raw=[json.loads(l) for l in source.read_text().splitlines()][:64]
 # Strip labels so benchmark cannot consume gold annotations.
 inp=OUT/'inputs.jsonl';inp.write_text(''.join(json.dumps({'ID':r['ID'],'Text':r['Text'],'Quadruplet':[]})+'\n' for r in raw))
 ids=[r['ID'] for r in raw]
 manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),dataset='eng_restaurant',selection='first 64 dev rows in existing order, no label or outcome selection',rows=64,batch_size=8,max_length=256,precision='float32',seeds=[42,123,456],arms=['A','B','C'],warmup_full_passes=1,repeats=5,ensemble='three GPU-resident models, sequential inference, exact-string vote2 and VA mean',timing='synchronized wall time; pretokenized CPU batches; includes H2D, model.predict and decoding, excludes tokenization, loading, scoring; ensemble includes CPU voting',memory='torch peak allocated/reserved; single model on GPU alone; ensemble all three resident; allocator cleared before warmup',caveat='One fixed English dev workload, no training or test scoring; does not establish all-language or production latency.',gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),torch=torch.__version__,cuda=torch.version.cuda,protocol_sha256=digest(lockpath),input_sha256=digest(inp),script_sha256=digest(Path(__file__)),environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','TOKENIZERS_PARALLELISM']})
 write('protocol.json',manifest);results=[];loads=[]
 for arm in 'ABC':
  models=[]
  for seed in (42,123,456):
   r=next(r for r in lock['models'] if r['dataset']=='eng_restaurant' and r['arm']==arm and int(r['seed'])==seed)
   status('loading',arm=arm,seed=seed)
   path=ROOT/r['checkpoint'];assert digest(path)==r['checkpoint_sha256']
   start=time.perf_counter();m,cfg=load_control_checkpoint(path,'cpu');elapsed=time.perf_counter()-start
   loads.append(dict(arm=arm,seed=seed,seconds=elapsed,checkpoint_sha256=r['checkpoint_sha256'],scope='CPU checkpoint/model construction after hash verification; not a cold-cache measurement'))
   models.append(m)
  tok=AutoTokenizer.from_pretrained(cfg['model']['encoder_name'],local_files_only=True)
  data=DimASQPDataset(str(inp),tok,max_length=256,entity_labels=cfg['data']['entity_labels'],attribute_labels=cfg['data']['attribute_labels'])
  batches=[collate_fn([data[i] for i in range(j,j+8)]) for j in range(0,64,8)]
  for seed,m in zip((42,123,456),models):
   status('single_benchmark',arm=arm,seed=seed)
   m.to('cuda').eval();gc.collect();torch.cuda.empty_cache();run(m,batches);sync()
   expected=None
   for repeat in range(5):
    torch.cuda.reset_peak_memory_stats();sync();start=time.perf_counter();rows=run(m,batches);sync();elapsed=time.perf_counter()-start
    validate_seed_predictions(rows,ids);encoded=json.dumps(rows,sort_keys=True)
    if expected is None:expected=encoded
    assert encoded==expected,'Repeated inference changed predictions'
    results.append(dict(arm=arm,seed=seed,kind='single',repeat=repeat,seconds=elapsed,rows_per_second=64/elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
   m.to('cpu');gc.collect();torch.cuda.empty_cache()
  status('ensemble_benchmark',arm=arm)
  for m in models:m.to('cuda').eval()
  gc.collect();torch.cuda.empty_cache()
  ensemble([run(m,batches) for m in models]);sync();expected=None
  for repeat in range(5):
   torch.cuda.reset_peak_memory_stats();sync();start=time.perf_counter();pred=[run(m,batches) for m in models];sync();inference=time.perf_counter()-start
   vote_start=time.perf_counter();rows=ensemble(pred);vote=time.perf_counter()-vote_start;elapsed=inference+vote
   encoded=json.dumps(rows,sort_keys=True)
   if expected is None:expected=encoded
   assert encoded==expected
   results.append(dict(arm=arm,seed=None,kind='ensemble',repeat=repeat,seconds=elapsed,inference_seconds=inference,voting_seconds=vote,rows_per_second=64/elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
  write('measurements.json',dict(records=results,loads=loads))
  for m in models:m.to('cpu')
  del models,m;gc.collect();torch.cuda.empty_cache()
 summary=[]
 for arm in 'ABC':
  for kind in ['single','ensemble']:
   rows=[r for r in results if r['arm']==arm and r['kind']==kind];times=[r['seconds'] for r in rows]
   summary.append(dict(arm=arm,kind=kind,n=len(rows),mean_seconds=statistics.mean(times),sample_sd_seconds=statistics.stdev(times),median_seconds=statistics.median(times),rows_per_second=64/statistics.mean(times),peak_allocated_gib=max(r['peak_allocated_bytes'] for r in rows)/2**30,peak_reserved_gib=max(r['peak_reserved_bytes'] for r in rows)/2**30))
 assert len(results)==60
 write('summary.json',dict(protocol_sha256=digest(OUT/'protocol.json'),summary=summary,scope=manifest['caveat'],single_sd='Across 3 seeds x 5 repeats, combines seed and timing variation; raw per-seed records retained.'))
 lines=['# 重复推理成本实验','',manifest['caveat'],'','| Arm | Kind | n | Seconds mean±SD /64 rows | rows/s | Peak allocated GiB |','|---|---|---:|---:|---:|---:|']
 for r in summary:lines.append(f"| {r['arm']} | {r['kind']} | {r['n']} | {r['mean_seconds']:.3f}±{r['sample_sd_seconds']:.3f} | {r['rows_per_second']:.1f} | {r['peak_allocated_gib']:.2f} |")
 (OUT/'summary.md').write_text('\n'.join(lines)+'\n');status('complete',measurements=60)
if __name__=='__main__':
 try:main()
 except BaseException as e:status('failed',error=repr(e));raise
