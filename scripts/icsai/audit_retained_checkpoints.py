"""Read config/metrics from retained alternative checkpoints on CPU with mmap."""
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
rows=[]
for prefix in ['final_zho_laptop_v16_seed','final_zho_restaurant_v16_seed','v5repro_eng_laptop_seed','v5repro_eng_laptop_inductive_seed','v13b_ukr_inductive_seed']:
    for seed in [42,123,456,789,1024]:
        p=ROOT/'checkpoints'/f'{prefix}{seed}'/'checkpoint_best.pt'
        row={'path':str(p.relative_to(ROOT)),'seed':seed,'exists':p.is_file()}
        if p.is_file():
            ck=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
            row.update(bytes=p.stat().st_size,keys=list(ck),config=ck.get('config',{}),epoch=ck.get('epoch'),metrics=ck.get('metrics',{}))
            del ck
        rows.append(row)
(ROOT/'docs/audits/icsai_phase1/retained_checkpoint_metadata.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print('Read',sum(r['exists'] for r in rows),'of',len(rows),'alternative checkpoints; no weight hashes or inference')
