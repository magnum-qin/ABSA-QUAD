"""Diagnose the installed Transformers Mistral-regex warning on mDeBERTa."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import transformers
from transformers import AutoTokenizer
from scripts.icsai.control_runtime import config_for_run

cfg=config_for_run('eng_restaurant_A_s42', 'cpu')
p=Path(cfg['model']['encoder_name'])
base=AutoTokenizer.from_pretrained(str(p),local_files_only=True)
explicit=AutoTokenizer.from_pretrained(str(p),local_files_only=True,fix_mistral_regex=False)
samples=[]
for f in sorted((ROOT/'data/icsai_phase1/gold_clean').glob('*.jsonl')):
    samples.extend(json.loads(line)['Text'] for line in f.read_text().splitlines()[:8])
for text in samples:
    assert base(text, return_offsets_mapping=True)==explicit(text, return_offsets_mapping=True)
backend=json.loads(base.backend_tokenizer.to_str())
source=Path(transformers.__file__).parent/'tokenization_utils_base.py'
result={'passed':True,'transformers_version':transformers.__version__,
        'tokenizer_class':type(base).__name__,'model_config':json.loads((p/'config.json').read_text()),
        'backend_model_type':backend['model']['type'],'pre_tokenizer':backend['pre_tokenizer'],
        'samples':len(samples),'default_equals_explicit_no_mistral_fix_ids_and_offsets':True,
        'source_file':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'interpretation':'Installed 4.57.6 local-tokenizer detection can enter the Mistral warning branch when config has no transformers_version; this snapshot is DebertaV2/Unigram, not Mistral. Default leaves tokenizer unchanged. No Mistral regex patch applied; no runtime or protocol change. Sample equivalence is not a universal tokenizer correctness proof.'}
(ROOT/'docs/audits/icsai_phase2/tokenizer_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['passed','tokenizer_class','backend_model_type','samples']},indent=2))
