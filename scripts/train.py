"""
Training Script for DimASQP Model
SemEval-2026 Task 3: Dimensional Aspect Sentiment Quad Prediction
"""

import os
import sys
import yaml
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dimabsa_model import DimASQPModel
from src.data.dataset import DimASQPDataset, create_dataloader, collate_fn
from src.evaluation.metrics import DimASQPEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for DimASQP Model
    """
    
    def __init__(
        self,
        model: DimASQPModel,
        train_dataloader,
        val_dataloader,
        config: dict,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        
        # Optimizer with different learning rates
        encoder_params = list(model.encoder.parameters())
        head_params = [p for n, p in model.named_parameters() if 'encoder' not in n]
        
        encoder_lr = float(config['training']['encoder_lr'])
        head_lr = float(config['training']['head_lr'])
        self.optimizer = AdamW([
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': head_params, 'lr': head_lr}
        ], lr=encoder_lr, weight_decay=config['training']['weight_decay'])
        
        # Scheduler
        num_training_steps = len(train_dataloader) * config['training']['num_epochs']
        num_warmup_steps = int(num_training_steps * config['training']['warmup_ratio'])
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Loss weights
        self.loss_weights = config['training']['loss_weights']
        
        # Training state
        self.best_metric = 0.0
        self.patience_counter = 0
        self.global_step = 0
        
        # Checkpointing
        self.save_dir = Path(config['training']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {
            'bio_loss': 0.0,
            'pair_loss': 0.0,
            'entity_loss': 0.0,
            'attribute_loss': 0.0,
            'va_loss': 0.0,
            'consistency_loss': 0.0
        }
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            leave=True
        )
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            bio_labels = batch['bio_labels'].to(self.device)
            pair_labels = batch['pair_labels']
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bio_labels=bio_labels,
                pair_labels=pair_labels,
                return_loss=True
            )
            
            # Weighted loss
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            for key in loss_components:
                if key in outputs:
                    loss_components[key] += outputs[key].item()
            
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Average losses
        num_batches = len(self.train_dataloader)
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {
            'loss': avg_loss,
            **loss_components
        }
    
    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate on validation set"""
        self.model.eval()
        
        evaluator = DimASQPEvaluator()
        total_loss = 0.0
        
        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            bio_labels = batch['bio_labels'].to(self.device)
            pair_labels = batch['pair_labels']
            texts = batch['texts']
            gold_quads = batch['quadruplets']
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bio_labels=bio_labels,
                pair_labels=pair_labels,
                return_loss=True
            )
            
            total_loss += outputs['loss'].item()
            
            # Generate predictions
            predictions = self.model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                texts=texts
            )
            
            # Convert gold labels to expected format
            gold_formatted = []
            for quads in gold_quads:
                formatted = []
                for q in quads:
                    formatted.append({
                        "Aspect": q.get("aspect", ""),
                        "Category": f"{q.get('entity', 'NULL')}#{q.get('attribute', 'GENERAL')}",
                        "Opinion": q.get("opinion", ""),
                        "VA": f"{q.get('valence', 5.0):.2f}#{q.get('arousal', 5.0):.2f}"
                    })
                gold_formatted.append(formatted)
            
            evaluator.add_batch(predictions, gold_formatted)
        
        # Compute metrics
        metrics = evaluator.compute_all_metrics()
        metrics['val_loss'] = total_loss / len(self.val_dataloader)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, self.save_dir / 'checkpoint_latest.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.save_dir / 'checkpoint_best.pt')
            logger.info(f"✅ Saved new best model with cF1: {metrics['continuous_f1']:.4f}")
    
    def train(self):
        """Main training loop"""
        logger.info("🚀 Starting training...")
        logger.info(f"Config: {json.dumps(self.config['training'], indent=2)}")
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{self.config['training']['num_epochs']}")
            logger.info('='*60)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            
            # Evaluate
            val_metrics = self.evaluate()
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"Val cF1: {val_metrics['continuous_f1']:.4f}")
            logger.info(f"Val Exact F1: {val_metrics['exact_f1']:.4f}")
            logger.info(f"Val VA RMSE: {val_metrics['va_rmse']:.4f}")
            
            # Check for improvement
            is_best = val_metrics['continuous_f1'] > self.best_metric
            if is_best:
                self.best_metric = val_metrics['continuous_f1']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config['training']['patience']:
                logger.info(f"⚠️ Early stopping triggered after {epoch} epochs")
                break
        
        logger.info(f"\n🎉 Training completed! Best cF1: {self.best_metric:.4f}")
        
        return self.best_metric


def main():
    parser = argparse.ArgumentParser(description='Train DimASQP Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--train_file', type=str, default=None,
                       help='Override train file path')
    parser.add_argument('--val_file', type=str, default=None,
                       help='Override validation file path')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args
    if args.train_file:
        config['data']['train_file'] = args.train_file
    if args.val_file:
        config['data']['dev_file'] = args.val_file
    if args.device:
        config['device'] = args.device
    
    # Set device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(config.get('seed', 42))
    if device == 'cuda':
        torch.cuda.manual_seed_all(config.get('seed', 42))
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {config['model']['encoder_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['encoder_name'])
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataloader = create_dataloader(
        data_path=config['data']['train_file'],
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['data']['max_seq_length'],
        shuffle=True
    )
    
    val_dataloader = create_dataloader(
        data_path=config['data']['dev_file'],
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['data']['max_seq_length'],
        shuffle=False
    )
    
    logger.info(f"Train samples: {len(train_dataloader.dataset)}")
    logger.info(f"Val samples: {len(val_dataloader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
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
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        device=device
    )
    
    # Train
    best_metric = trainer.train()
    
    return best_metric


if __name__ == "__main__":
    main()