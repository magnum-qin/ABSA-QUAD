"""Verify the Phase-1 evidence scope, file lock and unlabelled-input isolation."""
import hashlib,json,unicodedata,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/audits/icsai_phase1'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def norm(s): return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())

def main():
    checks=[]
    lock=read(OUT/'phase1.lock.json')
    for name,value in lock['files'].items():
        p=ROOT/name
        assert p.is_file() and sha(p)==value, f'Frozen file missing/changed: {name}'
    checks.append(f"{len(lock['files'])} frozen file hashes match")
    inv=read(OUT/'inventory.json'); matrix=read(OUT/'experiment_matrix.json'); inputs=read(OUT/'gold_input_manifest.json'); probe=read(OUT/'rethink_probe.json')
    assert len(inv['datasets'])==len(inputs['datasets'])==8
    for d in inv['datasets']:
        assert len(d['historical_checkpoints'])==5
        for c in d['historical_checkpoints']:
            assert (ROOT/c['path']).is_file()==c['exists'], f"Checkpoint existence changed: {c['path']}"
        for pred in d['dev_prediction_candidates']+d['test_prediction_candidates']:
            assert sha(ROOT/pred['path'])==pred['sha256'], f"Prediction changed: {pred['path']}"
    checks.append('Eight historical systems: 40 checkpoint existence entries and candidate prediction hashes rechecked')
    for d in inputs['datasets']:
        for pathkey,hashkey in [('source','source_sha256'),('train_file','train_sha256'),('dev_file','dev_sha256'),('test_input','test_input_sha256')]:
            assert sha(ROOT/d[pathkey])==d[hashkey], f"Input changed: {d[pathkey]}"
        source=rows(ROOT/d['source']); train=rows(ROOT/d['train_file']); dev=rows(ROOT/d['dev_file']); test=rows(ROOT/d['test_input'])
        blocked={norm(r['Text']) for r in dev+test}
        expected=[r for r in source if norm(r['Text']) not in blocked]
        assert train==expected and len(train)==d['kept_rows'] and len(source)==d['source_rows']
        assert not ({norm(r['Text']) for r in train}&blocked)
        labels=matrix['label_maps'][d['dataset']]
        for r in train:
            for q in r.get('Quadruplet',[]):
                e,a=q['Category'].split('#'); assert e in labels['entity_labels'] and a in labels['attribute_labels']
    checks.append('Eight frozen gold inputs reconstructed logically; no exact normalized train/dev or train/unlabelled-test overlap; label maps cover train')
    runs=matrix['experiments']; assert len(runs)==42 and len({r['id'] for r in runs})==42
    core=set(matrix['statistics']['primary_datasets']); assert core=={'eng_restaurant','zho_restaurant','tat_restaurant'}
    assert sum(r['priority']=='core' for r in runs)==27
    for d in inputs['datasets']:
        for arm in ['A','B','C'] if d['dataset'] in core else ['A']:
            rr=[r for r in runs if r['dataset']==d['dataset'] and r['arm']==arm]
            assert sorted(r['seed'] for r in rr)==[42,123,456]
            assert all(r['train_file']==d['train_file'] and r['status']=='planned_not_run' for r in rr)
    assert matrix['training']['early_stopping'] is False and matrix['test_exposure']['already_exposed'] is True
    assert matrix['inference']['pair_threshold']==0.3 and matrix['inference']['three_seed_ensemble']['vote_threshold']==2
    checks.append('42 unique planned runs: 27 complete-design core cells + 15 extension cells; no completed-run claim')
    assert probe['passed'] and probe['refinement_calls']==2
    assert probe['pair_input_intervention_max_delta']==probe['discarded_pair_output_intervention_max_delta']==0
    assert probe['bio_intervention_max_delta']>0
    for file,h in probe['source_sha256'].items(): assert sha(ROOT/file)==h
    checks.append('Dynamic path evidence passes and matches current source hashes; no inference about accuracy')
    assert matrix['conference']['submission_deadline']=='2026-09-15'
    assert matrix['conference']['initial_page_limit'] is None and matrix['conference']['registration_included_pages']==6
    sources=read(OUT/'conference_sources.json')
    for src in sources: assert sha(ROOT/src['snapshot'])==src['sha256']
    home=(OUT/'conference_home.html').read_text(); sub=(OUT/'conference_submission.html').read_text()
    assert '15 September 2026' in home and 'text-decoration:none' in home
    assert 'no page limits for initial submissions' in sub and 'covers 6 pages' in sub
    with zipfile.ZipFile(OUT/'conference_latex.zip') as z: assert {'IEEEtran.cls','bare_conf.tex'}<=set(z.namelist())
    checks.append('Official saved pages and IEEE template support conference date/page facts; timezone remains explicitly unspecified')
    result={'status':'pass','scope':'Phase-1 inventory/design integrity only. Does not establish historical run lineage, complete training, independent blind test, publication clearance, or statistical performance.','checks':checks}
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
