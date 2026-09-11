"""Deterministic descriptive case selection from previously exposed dev results."""
import json,hashlib
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/audits/icsai_cases'
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def keys(qs):return Counter(tuple(q[k].lower() for k in ['Aspect','Category','Opinion']) for q in qs)
goldpath=ROOT/'data/subtask_3/eng/eng_restaurant_dev_task3.jsonl';gold=read(goldpath)
paths={a:ROOT/f'predictions/icsai_phase1/eng_restaurant_{a}_s42/dev.jsonl' for a in 'ABC'};pred={a:read(p) for a,p in paths.items()}
for a,rows in pred.items():assert [r['ID'] for r in rows]==[r['ID'] for r in gold]
protocol=json.loads((OUT/'protocol.json').read_text());selected=[];used=set();counts=[]
for cat in range(4):
 eligible=[]
 for i,g in enumerate(gold):
  kg=keys(g['Quadruplet']);ka=keys(pred['A'][i]['Quadruplet']);kb=keys(pred['B'][i]['Quadruplet'])
  gooda=bool(kg) and kg==ka and max(ka.values(),default=0)==1;goodb=bool(kg) and kg==kb and max(kb.values(),default=0)==1
  nullmiss=any(('null'==k[0] or 'null'==k[2]) and ka[k]<n for k,n in kg.items())
  if [gooda,not gooda and goodb,gooda and not goodb,nullmiss][cat]:eligible.append(i)
 counts.append(dict(category=protocol['categories'][cat],eligible_rows=len(eligible)))
 available=[i for i in eligible if i not in used]
 if not available: selected.append(dict(category=protocol['categories'][cat],status='no eligible unused example'));continue
 i=available[0];used.add(i);g=gold[i]
 selected.append(dict(category=protocol['categories'][cat],row_1based=i+1,ID=g['ID'],text=g['Text'],gold=g['Quadruplet'],predictions={a:pred[a][i]['Quadruplet'] for a in 'ABC'}))
report=dict(protocol_sha256=sha(OUT/'protocol.json'),sources={str(p.relative_to(ROOT)):sha(p) for p in [goldpath,*paths.values()]},eligibility_counts=counts,cases=selected)
(OUT/'cases.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report['cases'],ensure_ascii=False,indent=2))
