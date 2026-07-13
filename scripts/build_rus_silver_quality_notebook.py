#!/usr/bin/env python3
"""Build the reproducible Russian silver-quality audit notebook."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/rus_silver_quality_audit.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (absa)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
nb["cells"] = [
    nbf.v4.new_markdown_cell(
        "# Russian silver quality audit: gen4k vs gen8k\n\n"
        "Reproducible low-cost audit of schema validity, span fidelity, distribution drift, "
        "lexical diversity, and the already-completed paired-seed decision gate. "
        "Automated proxies do **not** establish semantic label correctness; a deterministic "
        "100-row sample is exported separately for human adjudication."
    ),
    nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import importlib.util, json\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "ROOT = Path.cwd()\n"
        "if not (ROOT / 'scripts/analyze_rus_silver_quality.py').exists():\n"
        "    ROOT = ROOT.parent\n"
        "spec = importlib.util.spec_from_file_location('silver_audit', ROOT / 'scripts/analyze_rus_silver_quality.py')\n"
        "silver_audit = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(silver_audit)\n"
        "metrics = silver_audit.run_analysis(sample_size=3000)\n"
        "print('Inputs:', metrics['source_files'])"
    ),
    nbf.v4.new_markdown_cell("## Core data-quality indicators"),
    nbf.v4.new_code_cell(
        "rows = []\n"
        "for batch in ('gen4k', 'gen8k'):\n"
        "    x = metrics['batches'][batch]\n"
        "    rows.append({\n"
        "        'batch': batch, 'rows': x['rows'], 'invalid_rows': x['invalid_rows'],\n"
        "        'duplicate_rows': x['normalized_duplicate_rows'],\n"
        "        'quads_per_row': x['quadruplets_per_row']['mean'],\n"
        "        'category_JSD_vs_gold': x['category_jsd_vs_gold'],\n"
        "        'category_top_share': x['category_top_share'],\n"
        "        'MSTTR_100_equal_n': x['equal_size_lexical_sample']['msttr_100'],\n"
        "        'unigram_novelty_equal_n': x['equal_size_lexical_sample']['unigram_novelty_vs_reference'],\n"
        "    })\n"
        "quality = pd.DataFrame(rows).set_index('batch')\n"
        "quality.round(4)"
    ),
    nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))\n"
        "quality['category_JSD_vs_gold'].plot.bar(ax=axes[0], color=['#2f6f9f', '#a55b3b'], rot=0)\n"
        "axes[0].set_title('Category drift vs gold (lower is closer)')\n"
        "axes[0].set_ylabel('Jensen–Shannon divergence')\n"
        "quality['invalid_rows'].div(quality['rows']).mul(100).plot.bar(ax=axes[1], color=['#2f6f9f', '#a55b3b'], rot=0)\n"
        "axes[1].set_title('Rows with automated validity flags')\n"
        "axes[1].set_ylabel('% of batch')\n"
        "plt.tight_layout(); plt.show()"
    ),
    nbf.v4.new_markdown_cell("## Decision evidence"),
    nbf.v4.new_code_cell(
        "paired = metrics['comparisons']['paired_seed_dev_cf1']\n"
        "performance = pd.DataFrame({\n"
        "    'gen8k full': [paired['gen8k_full']['seed42'], paired['gen8k_full']['seed123']],\n"
        "    'same-pool cap3000': [paired['gen8k_cap3000']['seed42'], paired['gen8k_cap3000']['seed123']],\n"
        "}, index=['seed42', 'seed123'])\n"
        "performance.loc['mean'] = performance.mean()\n"
        "performance.round(4)"
    ),
    nbf.v4.new_code_cell(
        "ax = performance.loc[['seed42', 'seed123']].plot.bar(figsize=(7, 3.5), rot=0, color=['#2f6f9f', '#a55b3b'])\n"
        "ax.set_ylabel('dev cF1'); ax.set_ylim(0.50, 0.58)\n"
        "ax.set_title('Same gen8k pool: cap3000 did not improve paired seeds')\n"
        "plt.tight_layout(); plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "## Interpretation\n\n"
        "- The same-pool cap3000 mean is 1.14pp below the full-gen8k paired mean, so the pre-registered "
        "+1.5pp continuation gate failed. No remaining seeds or cap5000 curve are warranted.\n"
        "- gen4k is slightly closer to gold in category distribution and has fewer flagged rows proportionally, "
        "but the proxy differences are small. The strong performance gap is therefore consistent with — but does "
        "not independently prove — higher semantic quality in the gen4k generation batch.\n"
        "- The defensible conclusion is batch-quality sensitivity, not a universal 3k optimum."
    ),
    nbf.v4.new_markdown_cell(
        "## Limitations and manual-review handoff\n\n"
        "The 100-row file at `docs/audits/rus_silver_manual_sample_100.jsonl` contains 50 deterministic "
        "examples per batch with empty `manual_label_correct` and `manual_notes` fields. Until a Russian-speaking "
        "reviewer fills those fields, no human-adjudicated precision claim should be reported."
    ),
]
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT)
print(OUTPUT)
