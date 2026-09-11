"""Build v3-only tables from the post-review records; preserve original tables."""
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.icsai.revision_diagnostics import OUT, read, sha, stage_summary, rows

TABLES = ROOT / 'docs/icsai_tables_v3'
NAMES = {'eng_restaurant': 'English', 'zho_restaurant': 'Chinese', 'tat_restaurant': 'Tatar'}


def table(name, caption, label, columns, header, body, wide=False):
    # Wide tables are used in the single-column article supplement.
    env = 'table'
    placement = 'htbp' if wide else 't'
    text = ('\\begin{' + env + '}[' + placement + ']\n\\centering\\small\n\\setlength{\\tabcolsep}{3pt}\n'
            '\\caption{' + caption + '}\\label{' + label + '}\n'
            '\\begin{tabular}{' + columns + '}\n\\toprule\n' + header + ' \\\\\n\\midrule\n' +
            '\n'.join(line + ' \\\\' for line in body) +
            '\n\\bottomrule\n\\end{tabular}\n\\end{' + env + '}\n')
    (TABLES / name).write_text(text)


def main():
    TABLES.mkdir(exist_ok=True)
    report = read(OUT / 'stage_summary.json')
    assert report['protocol_sha256'] == sha(OUT / 'stage_protocol.json')
    for run in report['runs']:
        trace = OUT / 'stages' / (run['run_id'] + '.jsonl')
        assert sha(trace) == run['trace_sha256']
        assert stage_summary(rows(trace)) == run['stages']
    compact, detailed = [], []
    for dataset in NAMES:
        for arm in 'ABC':
            entries = [r for r in report['summary'] if r['dataset'] == dataset and r['arm'] == arm]
            scores = {r['stage']: r['span_f1']['mean'] for r in entries}
            compact.append(f"{NAMES[dataset]} & {arm} & {scores['draft']:.3f} & " +
                           (f"{scores['refine1']:.3f} & {scores['final']:.3f}" if arm != 'A' else '-- & --'))
            for r in entries:
                vals = [f"${r[m]['mean']:.3f}\\pm{r[m]['sample_sd']:.3f}$" for m in ('bio_accuracy', 'bio_non_o_micro_f1', 'span_f1')]
                detailed.append(f"{NAMES[dataset]} & {arm} & {r['stage']} & " + ' & '.join(vals))
    table('stages.tex', 'Post-review development typed token-span F1, averaged over three fixed seeds. A has only a draft output. These are diagnostics of the existing BIO projection, not official quad scores.',
          'tab:stages', 'llrrr', 'Language & Arm & Draft & Refine 1 & Final', compact)
    table('stages_full.tex', 'Development stage diagnostics: mean and sample SD over three seeds. All attended positions, including CLS/SEP, are scored; padding is excluded. Span F1 uses exact token boundaries and aspect/opinion type.',
          'tab:stagesfull', 'lllccc', 'Language & Arm & Stage & BIO accuracy & Non-O BIO F1 & Span F1', detailed, wide=True)
    implicit = read(OUT / 'implicit_results.json')
    lines = []
    for unit in ('sentence', 'quad'):
        for stratum in ('explicit', 'implicit'):
            selected = [r for r in implicit['results'] if r['unit'] == unit and r['stratum'] == stratum]
            singles = [r['metrics']['cF1'] for r in selected if r['kind'] == 'single']
            ens = next(r for r in selected if r['kind'] == 'ensemble')
            n = ens['rows'] if unit == 'sentence' else ens['gold_quads']
            lines.append(f"{unit.title()} & {stratum.title()} & {n} & ${statistics.mean(singles):.4f}\\pm{statistics.stdev(singles):.4f}$ & {ens['metrics']['cF1']:.4f}")
    table('implicit.tex', 'Japanese hotel test cF1 by sentence and quad partition. N counts sentences in the first partition and gold quads in the second. Single-model entries are three-seed means and sample SDs; ensemble is fixed vote-2.',
          'tab:implicit', 'llrcc', 'Unit & Stratum & N & Single & Ensemble', lines, wide=True)
    contrasts = read(OUT / 'seed_contrasts.json')['contrasts']
    lines = []
    for r in contrasts:
        vals = [f"{r['paired_seed_delta_pp'][s]:+.2f}" for s in ('42', '123', '456')]
        lines.append(f"{NAMES[r['dataset']]} & {r['contrast']} & " + ' & '.join(vals) + f" & {r['mean_delta_pp']:+.2f} & {r['holm_p']:.4f}")
    table('seed_contrasts.tex', 'All original paired test seed differences, in cF1 percentage points. The mean and Holm p-values reproduce the original six-contrast family; no new tests are introduced.',
          'tab:seedcontrasts', 'llrrrrr', 'Language & Contrast & Seed 42 & Seed 123 & Seed 456 & Mean & Holm p', lines, wide=True)
    print('Generated four v3 tables with source and trace checks')


if __name__ == '__main__':
    main()
