"""Integrate a completed benchmark, refusing incomplete runs."""
import json,hashlib,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'docs/audits/icsai_cost'
assert json.loads((out/'status.json').read_text())['stage']=='complete'
s=json.loads((out/'summary.json').read_text());raw=json.loads((out/'measurements.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
assert hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest()==s['protocol_sha256']
assert len(raw['records'])==60 and len(raw['loads'])==9
for row in s['summary']:
 vals=[r for r in raw['records'] if r['arm']==row['arm'] and r['kind']==row['kind']]
 assert abs(statistics.mean(r['seconds'] for r in vals)-row['mean_seconds'])<1e-12
 assert row['n']==(15 if row['kind']=='single' else 5)
rows=[]
for r in s['summary']:
 rows.append(f"{r['arm']} & {r['kind'].capitalize()} & ${r['mean_seconds']:.3f}\\pm{r['sample_sd_seconds']:.3f}$ & {r['rows_per_second']:.1f} & {r['peak_allocated_gib']:.2f} \\\\")
table='''\\begin{table}[t]
\\centering\\small\\setlength{\\tabcolsep}{3pt}
\\caption{Repeated cost on 64 fixed English development sentences. Wall time includes decoding and, for ensembles, voting; memory is peak allocated GPU memory. Single-model SD combines three seeds and five repeats per seed.}\\label{tab:cost}
\\begin{tabular}{llrrr}
\\toprule
Arm & Mode & Seconds $\\pm$ SD & Rows/s & GiB \\\\
\\midrule
'''+ '\n'.join(rows)+'''
\\bottomrule
\\end{tabular}
\\end{table}
'''
(ROOT/'docs/icsai_tables/cost.tex').write_text(table)
a=next(r for r in s['summary'] if r['arm']=='A' and r['kind']=='single')
b=next(r for r in s['summary'] if r['arm']=='B' and r['kind']=='single')
paragraph=r'''\subsection{Repeated Inference Cost}
\input{icsai_tables/cost.tex}
We benchmark the first 64 English restaurant development sentences in their existing order, stripping their labels. The workload is fixed before measurement and is not selected by model accuracy. All runs use the same RTX 4090, float32, batch size 8, maximum length 256, and pretokenized CPU batches. After one complete warmup pass, each seed model is measured five times while it alone resides on the GPU. For each arm, the fixed three-seed ensemble is measured five times with all three models resident and evaluated sequentially. CUDA synchronization brackets each inference pass; timing includes host-to-device transfers and decoding, plus CPU voting for ensembles. Tokenization and loading are excluded; CPU loading durations are recorded separately after checkpoint hash verification and are not cold-cache measurements.

Table~\ref{tab:cost} reports mean wall time, sample SD, throughput, and peak allocated GPU memory. Single-model variation combines three seeds and five repeats per seed; ensemble variation covers five repeats. Identical outputs across repeats are checked. '''+f"B takes {b['mean_seconds']/a['mean_seconds']:.2f} times A's mean single-model inference time on this workload. "+r'''The ensemble rows measure a specified residency policy, not every possible deployment strategy. Arms were measured in fixed A/B/C order rather than counterbalanced order, so small differences may include temporal effects. These measurements establish a local cost comparison for a short fixed English workload, not an all-language latency guarantee. They should be interpreted alongside the mixed feedback accuracy findings, rather than used alone to choose a universally preferable system.

'''
p=ROOT/'docs/paper_icsai_draft.tex';tex=p.read_text()
assert '\\subsection{Repeated Inference Cost}' not in tex
tex=tex.replace('\\subsection{Scope and Limitations}',paragraph+'\\subsection{Scope and Limitations}')
old='The parameter comparison does not substitute for repeated latency or memory benchmarks. Existing single-pass inference records include first-batch effects, so we make no strong efficiency claim from them. Additional cost measurements are needed before recommending a deployment configuration.'
assert old in tex
tex=tex.replace(old,'Table~\\ref{tab:cost} complements parameter counts with repeated measurements under one explicit inference setup. Input length, language, hardware, concurrency, and model-residency choices may change the deployment trade-off.')
p.write_text(tex)
(out/'integration.json').write_text(json.dumps(dict(measurements_verified=60,loading_records=9,B_over_A_mean_seconds=b['mean_seconds']/a['mean_seconds'],protocol=protocol),indent=2)+'\n')
print('Integrated verified benchmark into manuscript.')
