"""
Inference Script for DimASQP Model
Generates predictions in SemEval-2026 submission format

Supports:
  - Single model inference (--checkpoint)
  - Weight-averaging ensemble of top-k checkpoints (--ensemble --checkpoint_dir)
  - Multi-seed voting ensemble (--multi_seed_ensemble --seed_dirs)
"""

import sys
import json
import yaml
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dimabsa_model import DimASQPModel
from src.data.dataset import DimASQPDataset, collate_fn
from torch.utils.data import DataLoader


def _create_model(config: dict) -> DimASQPModel:
    """Create a DimASQPModel from config (without loading weights)."""
    return DimASQPModel(
        encoder_name=config['model']['encoder_name'],
        num_entity_labels=config['model']['num_entity_labels'],
        num_attribute_labels=config['model']['num_attribute_labels'],
        hidden_dim=config['model']['hidden_size'],
        va_hidden_dim=config['model']['va_hidden_size'],
        label_embedding_dim=config['model']['rethink']['label_embedding_dim'],
        dropout=config['model']['dropout'],
        use_rethink=config['model']['rethink']['enabled'],
        num_rethink_iterations=config['model']['rethink']['num_iterations'],
        va_min=config['model']['va_min'],
        va_max=config['model']['va_max'],
        loss_weights=config['training'].get('loss_weights'),
        entity_labels=config['data'].get('entity_labels'),
        attribute_labels=config['data'].get('attribute_labels'),
        category_embed_dim=config['model'].get('category_embed_dim', 0),
        use_crf=config['model'].get('use_crf', False)
    )


def _apply_ema_weights(model: DimASQPModel, checkpoint: dict) -> bool:
    """Apply EMA shadow weights to model if available. Returns True if applied."""
    ema_state = checkpoint.get('ema_state_dict')
    if ema_state and 'shadow' in ema_state:
        shadow = ema_state['shadow']
        for name, param in model.named_parameters():
            if name in shadow:
                param.data.copy_(shadow[name])
        return True
    return False


def _merge_config_from_checkpoint(config: dict, checkpoint: dict) -> dict:
    """Merge saved config from checkpoint into user config.

    Checkpoint's entity_labels/attribute_labels take priority to avoid
    label index mismatch between training and inference.
    """
    saved_config = checkpoint.get('config', {})
    saved_data = saved_config.get('data', {})
    saved_model = saved_config.get('model', {})

    # Restore encoder name (e.g. mdeberta-v3-base for non-English)
    if saved_model.get('encoder_name'):
        config['model']['encoder_name'] = saved_model['encoder_name']
        print(f"  Using encoder from checkpoint: {saved_model['encoder_name']}")

    if saved_data.get('entity_labels'):
        config['data']['entity_labels'] = saved_data['entity_labels']
        config['model']['num_entity_labels'] = len(saved_data['entity_labels'])
        print(f"  Using entity_labels from checkpoint: {saved_data['entity_labels']}")
    if saved_data.get('attribute_labels'):
        config['data']['attribute_labels'] = saved_data['attribute_labels']
        config['model']['num_attribute_labels'] = len(saved_data['attribute_labels'])
        print(f"  Using attribute_labels from checkpoint: {saved_data['attribute_labels']}")

    return config


def load_model(checkpoint_path: str, config: dict, device: str) -> DimASQPModel:
    """Load trained model from checkpoint, preferring EMA shadow weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = _merge_config_from_checkpoint(config, checkpoint)
    model = _create_model(config)

    if _apply_ema_weights(model, checkpoint):
        print("  Loaded EMA shadow weights (used during validation)")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  Loaded raw model weights (no EMA found)")

    model.to(device)
    model.eval()
    return model


def load_ensemble_model(checkpoint_dir: str, config: dict, device: str) -> DimASQPModel:
    """Load multiple checkpoints and average their weights (Weight Averaging ensemble).

    Prefers EMA shadow weights when available. Only averages top-k epoch checkpoints,
    skipping checkpoint_best.pt and checkpoint_latest.pt to avoid double-counting.
    """
    ckpt_dir = Path(checkpoint_dir)
    # Only use epoch checkpoints (e.g. checkpoint_epoch023.pt), skip best/latest
    ckpt_paths = sorted(ckpt_dir.glob('checkpoint_epoch*.pt'))
    if not ckpt_paths:
        raise FileNotFoundError(f"No epoch checkpoints found in {checkpoint_dir}")

    print(f"  Ensemble: averaging {len(ckpt_paths)} checkpoints:")
    for p in ckpt_paths:
        print(f"    - {p.name}")

    # Accumulate state dicts
    avg_state = None
    n = 0
    for path in ckpt_paths:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        # Prefer EMA shadow weights
        ema_state = ckpt.get('ema_state_dict')
        if ema_state and 'shadow' in ema_state:
            state = ema_state['shadow']
            source = "EMA"
        else:
            state = ckpt['model_state_dict']
            source = "raw"
        print(f"    {path.name}: using {source} weights")

        if avg_state is None:
            avg_state = {k: v.clone().float() for k, v in state.items()}
        else:
            for k in avg_state:
                if k in state:
                    avg_state[k] += state[k].float()
        n += 1

    # Average
    for k in avg_state:
        avg_state[k] /= n

    model = _create_model(config)
    model.load_state_dict(avg_state, strict=False)
    model.to(device)
    model.eval()
    print(f"  Ensemble model loaded ({n} checkpoints averaged)")
    return model


def generate_predictions(
    model: DimASQPModel,
    dataloader: DataLoader,
    device: str,
    pair_threshold: float = 0.3
) -> list:
    """Generate predictions for all samples"""
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            texts = batch['texts']
            sample_ids = batch['sample_ids']
            
            # Get predictions
            predictions = model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                texts=texts,
                offset_mapping=batch['offset_mapping'],
                pair_threshold=pair_threshold
            )
            
            # Format for submission
            for sample_id, quads in zip(sample_ids, predictions):
                all_predictions.append({
                    "ID": sample_id,
                    "Quadruplet": quads
                })
    
    return all_predictions


def multi_seed_ensemble_predictions(
    seed_dirs: list,
    config: dict,
    dataloader: DataLoader,
    device: str,
    pair_threshold: float = 0.3,
    vote_threshold: int = None,
    dual_layer: bool = False
) -> list:
    """Run inference with multiple seed models and merge via majority voting.

    For each sample:
      - Each model independently generates quads
      - Quads with same (Aspect, Category, Opinion) key are grouped
      - Only quads appearing in >= vote_threshold models are kept
      - VA values are averaged across models that predicted the quad

    Args:
        seed_dirs: list of checkpoint directories (one per seed)
        config: base config dict
        dataloader: test dataloader
        device: device string
        pair_threshold: pair confidence threshold
        vote_threshold: minimum votes to keep a quad (default: majority)

    Returns:
        Merged predictions list
    """
    n_models = len(seed_dirs)
    if vote_threshold is None:
        vote_threshold = (n_models // 2) + 1  # majority
    print(f"Multi-seed ensemble: {n_models} models, vote_threshold={vote_threshold}")

    # Collect predictions from each model
    all_model_preds = []
    for i, seed_dir in enumerate(seed_dirs):
        if dual_layer:
            # Dual-layer: average top-k checkpoints within each seed
            print(f"\n--- Model {i+1}/{n_models}: {seed_dir} (dual-layer) ---")
            epoch_ckpts = sorted(Path(seed_dir).glob('checkpoint_epoch*.pt'))
            if epoch_ckpts:
                # Peek first checkpoint to merge config
                peek_ckpt = torch.load(epoch_ckpts[0], map_location='cpu', weights_only=False)
                seed_config = _merge_config_from_checkpoint(config.copy(), peek_ckpt)
                model = load_ensemble_model(seed_dir, seed_config, device)
            else:
                # Fallback to best checkpoint
                print(f"  No epoch checkpoints, falling back to checkpoint_best.pt")
                ckpt_path = Path(seed_dir) / "checkpoint_best.pt"
                model = load_model(str(ckpt_path), config.copy(), device)
        else:
            ckpt_path = Path(seed_dir) / "checkpoint_best.pt"
            print(f"\n--- Model {i+1}/{n_models}: {ckpt_path} ---")
            model = load_model(str(ckpt_path), config.copy(), device)
        preds = generate_predictions(model, dataloader, device, pair_threshold)
        all_model_preds.append(preds)
        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Merge predictions by voting
    n_samples = len(all_model_preds[0])
    merged = []
    for idx in range(n_samples):
        sample_id = all_model_preds[0][idx]["ID"]
        # Group quads by key
        quad_votes = defaultdict(list)  # key -> list of (valence, arousal)
        for model_preds in all_model_preds:
            for quad in model_preds[idx]["Quadruplet"]:
                key = (quad["Aspect"], quad["Category"], quad["Opinion"])
                va_parts = quad["VA"].split("#")
                quad_votes[key].append((float(va_parts[0]), float(va_parts[1])))

        # Filter by vote threshold and average VA
        final_quads = []
        for (aspect, category, opinion), va_list in quad_votes.items():
            if len(va_list) >= vote_threshold:
                avg_v = sum(v for v, a in va_list) / len(va_list)
                avg_a = sum(a for v, a in va_list) / len(va_list)
                final_quads.append({
                    "Aspect": aspect,
                    "Category": category,
                    "Opinion": opinion,
                    "VA": f"{avg_v:.2f}#{avg_a:.2f}"
                })
        merged.append({"ID": sample_id, "Quadruplet": final_quads})

    return merged


def save_predictions(predictions: list, output_path: str):
    """Save predictions in JSONL format"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(predictions)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate DimASQP Predictions')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to single model checkpoint')
    parser.add_argument('--ensemble', action='store_true',
                       help='Use weight-averaging ensemble of top-k checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Directory containing top-k checkpoints (for --ensemble)')
    parser.add_argument('--multi_seed_ensemble', action='store_true',
                       help='Use multi-seed voting ensemble')
    parser.add_argument('--seed_dirs', nargs='+', default=None,
                       help='Checkpoint directories for each seed (for --multi_seed_ensemble)')
    parser.add_argument('--vote_threshold', type=int, default=None,
                       help='Min votes to keep a quad (default: majority)')
    parser.add_argument('--dual_layer', action='store_true',
                       help='Dual-layer ensemble: top-k weight avg within seed + voting across seeds')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--test_file', type=str, required=True,
                       help='Path to test data file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output predictions file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use')
    parser.add_argument('--pair_threshold', type=float, default=0.3,
                       help='Pair validity confidence threshold (default: 0.3)')
    args = parser.parse_args()

    if not args.ensemble and not args.multi_seed_ensemble and not args.checkpoint:
        parser.error("--checkpoint is required when not using --ensemble or --multi_seed_ensemble")
    if args.ensemble and not args.checkpoint_dir:
        parser.error("--checkpoint_dir is required when using --ensemble")
    if args.multi_seed_ensemble and not args.seed_dirs:
        parser.error("--seed_dirs is required when using --multi_seed_ensemble")

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = args.device or config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.multi_seed_ensemble:
        # Multi-seed voting ensemble: peek first checkpoint to get correct encoder_name
        first_ckpt_path = Path(args.seed_dirs[0]) / "checkpoint_best.pt"
        first_ckpt = torch.load(first_ckpt_path, map_location='cpu', weights_only=False)
        config = _merge_config_from_checkpoint(config, first_ckpt)
        del first_ckpt

        print(f"Loading tokenizer: {config['model']['encoder_name']}")
        tokenizer = AutoTokenizer.from_pretrained(config['model']['encoder_name'])

        print(f"Loading test data from: {args.test_file}")
        test_dataset = DimASQPDataset(
            data_path=args.test_file,
            tokenizer=tokenizer,
            max_length=config['data']['max_seq_length']
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=4
        )
        print(f"Test samples: {len(test_dataset)}")

        predictions = multi_seed_ensemble_predictions(
            seed_dirs=args.seed_dirs,
            config=config,
            dataloader=test_dataloader,
            device=device,
            pair_threshold=args.pair_threshold,
            vote_threshold=args.vote_threshold,
            dual_layer=args.dual_layer
        )
    else:
        # Single model or weight-averaging ensemble
        if args.ensemble:
            print(f"Loading ensemble from: {args.checkpoint_dir}")
            ckpt_paths = sorted(Path(args.checkpoint_dir).glob('checkpoint_epoch*.pt'))
            if ckpt_paths:
                peek_ckpt = torch.load(ckpt_paths[0], map_location='cpu', weights_only=False)
                config = _merge_config_from_checkpoint(config, peek_ckpt)
            model = load_ensemble_model(args.checkpoint_dir, config, device)
        else:
            print(f"Loading model from: {args.checkpoint}")
            model = load_model(args.checkpoint, config, device)

        print(f"Loading tokenizer: {config['model']['encoder_name']}")
        tokenizer = AutoTokenizer.from_pretrained(config['model']['encoder_name'])

        print(f"Loading test data from: {args.test_file}")
        test_dataset = DimASQPDataset(
            data_path=args.test_file,
            tokenizer=tokenizer,
            max_length=config['data']['max_seq_length']
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=4
        )
        print(f"Test samples: {len(test_dataset)}")

        predictions = generate_predictions(model, test_dataloader, device, pair_threshold=args.pair_threshold)

    # Save predictions
    save_predictions(predictions, args.output)

    # Print sample predictions
    print("\n📋 Sample predictions:")
    for i, pred in enumerate(predictions[:3]):
        print(f"\nSample {i+1}: {pred['ID']}")
        for quad in pred['Quadruplet']:
            print(f"  - {quad}")


if __name__ == "__main__":
    main()