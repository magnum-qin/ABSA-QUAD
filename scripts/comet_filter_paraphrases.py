"""
Score paraphrases against their gold originals using COMET.

For an open (non-gated) reference-based COMET model, we use the trick
src=original, mt=paraphrase, ref=original. Semantic similarity is what matters.

Adds a `comet` field to each paraphrase line. The training pipeline can then
filter on this score to drop semantic-drift samples (e.g. comet < 0.75).

Usage:
    python scripts/comet_filter_paraphrases.py \
        --train_file data/subtask_3/jpn/jpn_hotel_train_alltasks.jsonl \
        --paraphrase_file data/subtask_3/jpn/jpn_hotel_paraphrase_v10.jsonl \
        --output data/subtask_3/jpn/jpn_hotel_paraphrase_v10_scored.jsonl \
        --thresholds 0.65 0.70 0.75 0.80
"""

import json
import argparse
from pathlib import Path

from comet import download_model, load_from_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True,
                        help="Gold training JSONL (source originals)")
    parser.add_argument("--paraphrase_file", required=True,
                        help="Paraphrase JSONL with SourceID field")
    parser.add_argument("--output", required=True,
                        help="Output JSONL with added 'comet' score field")
    parser.add_argument("--model", default="Unbabel/wmt22-comet-da",
                        help="COMET model (default: open reference-based; "
                             "wmt22-cometkiwi-da is reference-free but gated)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.60, 0.65, 0.70, 0.75, 0.80])
    args = parser.parse_args()

    # Load originals
    originals = {}
    with open(args.train_file, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            originals[s["ID"]] = s["Text"]
    print(f"Loaded {len(originals)} source originals")

    # Load paraphrases
    paraphrases = []
    with open(args.paraphrase_file, encoding="utf-8") as f:
        for line in f:
            paraphrases.append(json.loads(line))
    print(f"Loaded {len(paraphrases)} paraphrases")

    # Build COMET inputs.
    # Reference-based models (e.g. wmt22-comet-da) need 'ref'; we set ref=src
    # to measure semantic equivalence between paraphrase and original.
    is_qe = ("kiwi" in args.model.lower() or "qe" in args.model.lower())
    inputs = []
    valid_indices = []
    for i, p in enumerate(paraphrases):
        src_text = originals.get(p.get("SourceID"))
        mt_text = p.get("Text")
        if src_text and mt_text:
            entry = {"src": src_text, "mt": mt_text}
            if not is_qe:
                entry["ref"] = src_text
            inputs.append(entry)
            valid_indices.append(i)
    print(f"Valid input pairs: {len(inputs)} (model={'QE' if is_qe else 'reference-based'})")

    # Load model + score
    print(f"Downloading/loading COMET model: {args.model}")
    model_path = download_model(args.model)
    model = load_from_checkpoint(model_path)
    print("Scoring...")
    output = model.predict(inputs, batch_size=args.batch_size, gpus=args.gpus)
    scores = output.scores
    print(f"Got {len(scores)} scores. System-level: {output.system_score:.4f}")

    # Attach scores back
    for idx, score in zip(valid_indices, scores):
        paraphrases[idx]["comet"] = float(score)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in paraphrases:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote scored paraphrases to: {out_path}")

    # Threshold table
    print("\n=== Threshold sweep ===")
    print(f"{'Threshold':>10} {'Kept':>6} {'%':>7} {'Dropped':>8}")
    print("-" * 35)
    n_total = len(scores)
    for t in args.thresholds:
        n_kept = sum(1 for s in scores if s >= t)
        pct = 100 * n_kept / n_total if n_total else 0
        print(f"{t:>10.2f} {n_kept:>6} {pct:>6.1f}% {n_total - n_kept:>8}")

    # Score distribution
    print("\n=== Score distribution ===")
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    if n > 0:
        for q, name in [(0, "min"), (n // 10, "10%"), (n // 4, "25%"),
                         (n // 2, "median"), (3 * n // 4, "75%"),
                         (9 * n // 10, "90%"), (n - 1, "max")]:
            print(f"  {name:>7}: {sorted_scores[q]:.4f}")
        avg = sum(scores) / n
        print(f"  {'mean':>7}: {avg:.4f}")


if __name__ == "__main__":
    main()
