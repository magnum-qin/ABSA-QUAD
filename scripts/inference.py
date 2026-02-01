"""
Inference Script for DimASQP Model
Generates predictions in SemEval-2026 submission format
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dimabsa_model import DimASQPModel
from src.data.dataset import DimASQPDataset, collate_fn
from torch.utils.data import DataLoader


def load_model(checkpoint_path: str, config: dict, device: str) -> DimASQPModel:
    """Load trained model from checkpoint"""
    model = DimASQPModel(
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
        va_max=config['model']['va_max']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def generate_predictions(
    model: DimASQPModel,
    dataloader: DataLoader,
    device: str
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
                texts=texts
            )
            
            # Format for submission
            for sample_id, quads in zip(sample_ids, predictions):
                all_predictions.append({
                    "ID": sample_id,
                    "Quadruplet": quads
                })
    
    return all_predictions


def save_predictions(predictions: list, output_path: str):
    """Save predictions in JSONL format"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(predictions)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate DimASQP Predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
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
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = args.device or config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {config['model']['encoder_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['encoder_name'])
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)
    
    # Create test dataset
    print(f"Loading test data from: {args.test_file}")
    test_dataset = DimASQPDataset(
        data_path=args.test_file,
        tokenizer=tokenizer,
        max_length=config['data']['max_seq_length']
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Generate predictions
    predictions = generate_predictions(model, test_dataloader, device)
    
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