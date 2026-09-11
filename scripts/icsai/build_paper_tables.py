"""Generate manuscript tables from locked evaluation artifacts, without rescoring."""
import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/icsai_tables';OUT.mkdir(exist_ok=True)
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
paths=['docs/audits/icsai_core_test/test_statistics.json','docs/audits/icsai_extension_test/test_statistics.json','docs/audits/icsai_extension_test/eight_domain_summary.json','docs/audits/icsai_phase1/gold_input_manifest.json']
core,ext,eight,inputs=map(read,paths)
for folder,report in [('icsai_core_test',core),('icsai_extension_test',ext)]:
 p=f'docs/audits/{folder}/predictions.lock.json';lock=read(p)
 assert sha(p)==report['prediction_lock_sha256']
 assert sha(f'docs/audits/{folder}/protocol_lock.json')==report['protocol_sha256']==lock['protocol_sha256']
 for r in lock['predictions']:assert sha(r['path'])==r['sha256']
for p,h in eight['source_sha256'].items():assert sha(p)==h
names={'eng_laptop':'English laptop','eng_restaurant':'English restaurant','jpn_hotel':'Japanese hotel','rus_restaurant':'Russian restaurant','tat_restaurant':'Tatar restaurant','ukr_restaurant':'Ukrainian restaurant','zho_laptop':'Chinese laptop','zho_restaurant':'Chinese restaurant'}
def table(file,caption,label,cols,head,rows):
 text='\\begin{table}[t]\n\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabular}{'+cols+'}\n\\toprule\n'+head+' \\\\\n\\midrule\n'+'\n'.join(r+' \\\\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
 if file=='ablation.tex': text=text.replace('\\centering\\small', '\\centering\\small\\setlength{\\tabcolsep}{3pt}')
 (OUT/file).write_text(text)
rows=[]
for r in inputs['datasets']:
 dev=sum(1 for l in (ROOT/r['dev_file']).open() if l.strip());test=sum(1 for l in (ROOT/r['test_input']).open() if l.strip())
 rows.append(f"{names[r['dataset']]} & {r['kept_rows']} & {dev} & {test}")
table('data.tex','Dataset sizes after removing training texts identical to development or unlabelled test texts.','tab:data','lrrr','Dataset & Train & Dev & Test',rows)
rows=[]
for r in eight['summary']:
 scores=[s['metrics']['cF1'] for s in eight['single_models'] if s['dataset']==r['dataset']]
 assert abs(statistics.mean(scores)-r['mean'])<1e-12
 rows.append(f"{names[r['dataset']]} & ${r['mean']:.4f}\\pm{r['sample_sd']:.4f}$ & {r['ensemble_cf1']:.4f}")
rows.append(f"Equal-domain mean & {eight['macro']['single_mean']:.4f} & {eight['macro']['ensemble_mean']:.4f}")
table('eight.tex','Official test cF1 for A: three-seed mean and sample SD; fixed vote-2 ensemble. Macro values are descriptive.','tab:eight','lcc','Dataset & Single model & Ensemble',rows)
rows=[]
for d in ['eng_restaurant','zho_restaurant','tat_restaurant']:
 vals=[next(r for r in core['summary'] if r['dataset']==d and r['arm']==a) for a in 'ABC']
 rows.append(names[d].replace(' restaurant','')+' & '+' & '.join(f"${r['mean']:.4f}\\pm{r['sample_sd']:.4f}$" for r in vals))
table('ablation.tex','Core restaurant test cF1 (mean $\\pm$ sample SD over three seeds).','tab:ablation','lccc','Language & A & B & C',rows)
rows=[]
for r in core['contrasts']:
 rows.append(f"{names[r['dataset']].replace(' restaurant','')} & {r['contrast']} & {100*r['delta']:+.2f} & $[{100*r['ci95'][0]:+.2f},{100*r['ci95'][1]:+.2f}]$ & {r['holm_p']:.4f}")
table('contrasts.tex','Paired test contrasts in cF1 percentage points. Percentile intervals are unadjusted; $p$ values use Holm correction over all six contrasts.','tab:contrasts','llrrr','Language & Contrast & $\\Delta$ & 95\\% CI & $p_{\\rm Holm}$',rows)
rows=[]
for d in sorted(names):
 rs=[r['metrics'] for r in eight['single_models'] if r['dataset']==d]
 rows.append(names[d]+ ' & '+' & '.join(f"{statistics.mean(r[k] for r in rs):.3f}" for k in ['exact_f1','matched_gold_coverage','matched_v_mae','matched_a_mae']))
table('diagnostics.tex','A model diagnostics, averaged across seeds. Coverage is the proportion of gold quads with exactly one categorical match; V/A MAE is conditional on these matches.','tab:diag','lrrrr','Dataset & F1 & Cov. & V MAE & A MAE',rows)
(OUT/'sources.json').write_text(json.dumps({'source_sha256':{p:sha(p) for p in paths},'validated_prediction_files':56,'tables':5},indent=2)+'\n')
print('Verified 56 prediction hashes and source locks; generated five tables.')
