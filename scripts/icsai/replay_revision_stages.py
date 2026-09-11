"""Explicit fresh-output replay of the frozen v3 diagnostics, without training."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.icsai import revision_diagnostics as diagnostics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    dest = Path(args.output_dir).resolve()
    if dest.exists():
        parser.error('Fresh replay requires an output directory that does not exist.')
    dest.mkdir(parents=True)
    diagnostics.OUT = dest
    diagnostics.stages(args.device)
