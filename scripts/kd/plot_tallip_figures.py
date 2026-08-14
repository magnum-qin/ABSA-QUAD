#!/usr/bin/env python
"""Render manuscript figures from frozen TALLIP KD evidence artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATISTICS = ROOT / "docs/audits/tallip_kd/statistical_robustness.json"
DEFAULT_OUTPUT_DIR = ROOT / "docs/figures"

DATASET_ORDER = [
    "eng_laptop",
    "rus_restaurant",
    "tat_restaurant",
    "ukr_restaurant",
    "zho_restaurant",
]
DATASET_LABELS = {
    "eng_laptop": "English laptop",
    "rus_restaurant": "Russian restaurant",
    "tat_restaurant": "Tatar restaurant",
    "ukr_restaurant": "Ukrainian restaurant",
    "zho_restaurant": "Chinese restaurant",
}
PANELS = [
    ("adopted_kd_vs_baseline_dev", "KD − baseline"),
    ("exposure_control_dev", "KD − repeated gold"),
    ("same_text_self_training_dev", "KD − same-text self-training"),
]

BLUE = "#2F6B9A"
BLUE_LIGHT = "#8DB3CE"
BLUE_DARK = "#173F5F"
INK = "#20252B"
MID_GREY = "#747B83"
GRID = "#D9DEE3"
MACRO_FILL = "#EAF2F7"


def load_statistics(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    families = report.get("families", {})
    macro = report.get("macro_average", {})
    for family, _ in PANELS:
        if family not in families or family not in macro:
            raise ValueError(f"missing required statistical family: {family}")
        rows = families[family]
        names = [row["dataset"] for row in rows]
        if set(names) != set(DATASET_ORDER) or len(names) != len(DATASET_ORDER):
            raise ValueError(f"unexpected dataset coverage for {family}: {names}")
        if macro[family].get("datasets") != DATASET_ORDER:
            raise ValueError(f"unexpected macro dataset order for {family}")
    return report


def ordered_rows(report: dict, family: str) -> list[dict]:
    by_dataset = {
        row["dataset"]: row for row in report["families"][family]
    }
    rows = [by_dataset[name] for name in DATASET_ORDER]
    macro = report["macro_average"][family]
    rows.append(
        {
            "dataset": "macro",
            "delta_a_minus_b": macro["macro_delta_a_minus_b"],
            "ci95": macro["ci95"],
            "classification": macro["classification"],
            "holm_reject_0.05": False,
        }
    )
    return rows


def render_forest(report: dict, output_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    labels = [DATASET_LABELS[name] for name in DATASET_ORDER] + ["Macro (equal weight)"]
    y_positions = list(reversed(range(len(labels))))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 5.25), sharex=True, sharey=True)

    for panel_index, (ax, (family, panel_title)) in enumerate(zip(axes, PANELS)):
        rows = ordered_rows(report, family)
        ax.axvspan(-0.08, 0, color="#F5F6F7", zorder=0)
        ax.axhspan(-0.42, 0.42, color=MACRO_FILL, zorder=0)
        ax.axvline(0, color=MID_GREY, linewidth=1.0, linestyle="--", zorder=1)
        ax.axhline(0.5, color=GRID, linewidth=0.8, zorder=1)

        for row, y in zip(rows, y_positions):
            delta = row["delta_a_minus_b"]
            low, high = row["ci95"]
            is_macro = row["dataset"] == "macro"
            supported = row["classification"] == "supported"
            color = BLUE_DARK if is_macro else (BLUE if supported else BLUE_LIGHT)
            marker = "D" if is_macro else "o"
            face = BLUE_DARK if is_macro or supported else "white"
            size = 6.2 if is_macro else 5.7

            ax.plot([low, high], [y, y], color=color, linewidth=1.8, zorder=2)
            ax.plot(
                delta,
                y,
                marker=marker,
                markersize=size,
                markerfacecolor=face,
                markeredgecolor=BLUE_DARK if is_macro else BLUE,
                markeredgewidth=1.25,
                linestyle="none",
                zorder=3,
            )
            value_label = f"{delta:+.3f}"
            if row.get("holm_reject_0.05"):
                value_label += "*"
            ax.text(
                high + 0.004,
                y,
                value_label,
                ha="left",
                va="center",
                fontsize=7.7,
                color=INK,
                fontweight="semibold" if is_macro else "normal",
            )

        ax.set_title(panel_title, color=INK, pad=10, fontweight="semibold")
        ax.set_xlim(-0.08, 0.21)
        ax.set_xticks([-0.05, 0.00, 0.05, 0.10, 0.15, 0.20])
        ax.grid(axis="x", color=GRID, linewidth=0.65)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=MID_GREY, length=3)
        ax.tick_params(axis="y", colors=INK, length=0, pad=7)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(MID_GREY)
        if panel_index == 0:
            ax.set_yticks(y_positions, labels=labels)
        ax.set_xlabel("Difference in development cF1", color=INK, labelpad=8)

    fig.suptitle(
        "Paired development-set effects of LLM-labeled silver data",
        x=0.08,
        y=0.975,
        ha="left",
        fontsize=13.0,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.08,
        0.932,
        "Points show cF1 differences; horizontal lines are 95% paired-bootstrap intervals (10,000 resamples).",
        ha="left",
        va="top",
        fontsize=9.0,
        color=MID_GREY,
    )

    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor=BLUE,
            markeredgecolor=BLUE, markersize=6, label="95% CI excludes zero",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="white",
            markeredgecolor=BLUE, markersize=6, label="95% CI crosses zero",
        ),
        Line2D(
            [0], [0], marker="D", linestyle="none", markerfacecolor=BLUE_DARK,
            markeredgecolor=BLUE_DARK, markersize=6, label="Five-domain macro",
        ),
        Line2D(
            [0], [0], marker="$*$", linestyle="none", color=INK,
            markersize=8, label="Holm-adjusted p < .05",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.56, 0.005),
        ncol=4,
        frameon=False,
        fontsize=8.2,
        handletextpad=0.5,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.185, right=0.985, top=0.855, bottom=0.175, wspace=0.12)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "tallip_kd_effect_forest.pdf"
    png_path = output_dir / "tallip_kd_effect_forest.png"
    metadata = {
        "Title": "Paired development-set effects of LLM-labeled silver data",
        "Author": "ABSA-QUAD reproducible figure pipeline",
        "Subject": "TALLIP KD paired-bootstrap effects",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf_path, bbox_inches="tight", metadata=metadata)
    fig.savefig(png_path, dpi=240, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return pdf_path, png_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    statistics = args.statistics if args.statistics.is_absolute() else ROOT / args.statistics
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    report = load_statistics(statistics)
    pdf_path, png_path = render_forest(report, output_dir)
    print(f"wrote {pdf_path.relative_to(ROOT)}")
    print(f"wrote {png_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
