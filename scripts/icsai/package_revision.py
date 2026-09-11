"""Build the local ICSAI v3 source/evidence archive with per-file hashes."""
from pathlib import Path
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.icsai.revision_diagnostics import OUT, read, sha


def main():
    selected = set(read(OUT / 'revision_plan.json')['protected_original_sha256'])
    selected.update(['environment.yml', 'docs/IEEEtran.cls', 'docs/paper_icsai_v3.tex',
                     'docs/paper_icsai_v3.pdf', 'docs/paper_icsai_v3_supplement.tex',
                     'docs/paper_icsai_v3_supplement.pdf', 'docs/ICSAI_REPRODUCIBILITY_V3.md',
                     'docs/ICSAI_REVISION_V3.md', 'tests/test_icsai_revision_diagnostics.py'])
    for folder in ('src', 'scripts', 'configs'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.yml', '.sh'):
                selected.add(str(path.relative_to(ROOT)))
    for folder in ('icsai_phase1', 'icsai_phase2', 'icsai_core_analysis', 'icsai_cost',
                   'icsai_cases', 'icsai_related_work', 'icsai_archive', 'icsai_revision_v3'):
        for path in (ROOT / 'docs/audits' / folder).rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.md', '.txt', '.gz'):
                # The external delivery validation records this archive's hash;
                # excluding it avoids self-referential checksums.
                if path.name not in ('archive_validation.json',):
                    selected.add(str(path.relative_to(ROOT)))
    selected.update(str(path.relative_to(ROOT)) for path in (ROOT / 'docs/icsai_tables_v3').glob('*.tex'))
    for path in selected:
        assert (ROOT / path).is_file(), path
        assert not (ROOT / path).is_symlink(), path
        assert not path.startswith(('data/', 'checkpoints/', 'predictions/'))
    manifest = {path: sha(ROOT / path) for path in sorted(selected)}
    archive = ROOT / 'docs/paper_icsai_v3_source.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('ARCHIVE_MANIFEST.json', json.dumps(dict(schema='icsai-v3-archive/1', files=manifest), indent=2) + '\n')
        for path in sorted(selected):
            z.write(ROOT / path, path)
    print(json.dumps(dict(archive=str(archive.relative_to(ROOT)), files=len(manifest),
                          bytes=archive.stat().st_size, sha256=sha(archive)), indent=2))


if __name__ == '__main__':
    main()
