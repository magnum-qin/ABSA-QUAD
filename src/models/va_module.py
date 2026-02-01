"""
Valence-Arousal Regression Module
Predicts continuous V-A scores in range [1, 9] for SemEval-2026 DimABSA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VARegressor(nn.Module):
    """
    Valence-Arousal Regression Head
    
    Predicts continuous values:
    - Valence: [1, 9] (positive/negative sentiment)
    - Arousal: [1, 9] (intensity/excitement level)
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        va_min: float = 1.0,
        va_max: float = 9.0
    ):
        super().__init__()
        
        self.va_min = va_min
        self.va_max = va_max
        self.va_range = va_max - va_min  # 8.0
        
        # Shared feature extraction
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # Separate heads for Valence and Arousal
        self.valence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, pair_representation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pair_representation: [batch_size, num_pairs, input_dim]
            
        Returns:
            valence: [batch_size, num_pairs] in range [1, 9]
            arousal: [batch_size, num_pairs] in range [1, 9]
        """
        # Shared features
        shared_feat = self.shared_layer(pair_representation)
        
        # Predict valence and arousal
        valence_logits = self.valence_head(shared_feat).squeeze(-1)
        arousal_logits = self.arousal_head(shared_feat).squeeze(-1)
        
        # Apply sigmoid and scale to [1, 9]
        valence = torch.sigmoid(valence_logits) * self.va_range + self.va_min
        arousal = torch.sigmoid(arousal_logits) * self.va_range + self.va_min
        
        return valence, arousal
    
    def compute_loss(
        self,
        pred_valence: torch.Tensor,
        pred_arousal: torch.Tensor,
        target_valence: torch.Tensor,
        target_arousal: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Huber Loss for V-A regression
        
        Args:
            pred_valence, pred_arousal: [batch_size, num_pairs]
            target_valence, target_arousal: [batch_size, num_pairs]
            mask: [batch_size, num_pairs] - valid pair mask
        """
        # Huber loss (robust to outliers)
        valence_loss = F.huber_loss(pred_valence, target_valence, reduction='none', delta=1.0)
        arousal_loss = F.huber_loss(pred_arousal, target_arousal, reduction='none', delta=1.0)
        
        if mask is not None:
            valence_loss = (valence_loss * mask).sum() / (mask.sum() + 1e-8)
            arousal_loss = (arousal_loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            valence_loss = valence_loss.mean()
            arousal_loss = arousal_loss.mean()
        
        return valence_loss + arousal_loss


class VAConsistencyVerifier(nn.Module):
    """
    Verifies consistency between opinion words and predicted V-A values
    Used in Output-level Rethink stage
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # Opinion encoder
        self.opinion_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # VA predictor from opinion
        self.opinion_to_va = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)  # [valence, arousal]
        )
    
    def forward(
        self,
        opinion_repr: torch.Tensor,
        pred_valence: torch.Tensor,
        pred_arousal: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency score between opinion and predicted V-A
        
        Returns:
            consistency_score: [batch_size, num_pairs] in [0, 1]
        """
        opinion_feat = self.opinion_encoder(opinion_repr)
        implied_va = self.opinion_to_va(opinion_feat)  # [B, N, 2]
        
        implied_valence = torch.sigmoid(implied_va[..., 0]) * 8 + 1
        implied_arousal = torch.sigmoid(implied_va[..., 1]) * 8 + 1
        
        # Compute distance-based consistency
        va_distance = torch.sqrt(
            (pred_valence - implied_valence) ** 2 + 
            (pred_arousal - implied_arousal) ** 2
        )
        
        # Convert to consistency score (closer = higher score)
        max_distance = torch.sqrt(torch.tensor(128.0))  # sqrt(8^2 + 8^2)
        consistency_score = 1.0 - va_distance / max_distance
        
        return consistency_score


class VAEmbedding(nn.Module):
    """
    Converts continuous V-A values to embeddings for Rethink module
    """
    
    def __init__(
        self,
        output_dim: int = 64,
        num_bins: int = 18,  # 0.5 intervals from 1-9
        va_min: float = 1.0,
        va_max: float = 9.0
    ):
        super().__init__()
        
        self.va_min = va_min
        self.va_max = va_max
        self.num_bins = num_bins
        
        # Discretized embeddings
        self.valence_embedding = nn.Embedding(num_bins, output_dim // 2)
        self.arousal_embedding = nn.Embedding(num_bins, output_dim // 2)
        
        # Continuous projection (for residual)
        self.continuous_proj = nn.Linear(2, output_dim)
    
    def _discretize(self, value: torch.Tensor) -> torch.Tensor:
        """Convert continuous value to bin index"""
        normalized = (value - self.va_min) / (self.va_max - self.va_min)
        bins = (normalized * (self.num_bins - 1)).long().clamp(0, self.num_bins - 1)
        return bins
    
    def forward(
        self,
        valence: torch.Tensor,
        arousal: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            valence, arousal: [batch_size, num_pairs]
            
        Returns:
            va_embedding: [batch_size, num_pairs, output_dim]
        """
        # Discretized embeddings
        v_bins = self._discretize(valence)
        a_bins = self._discretize(arousal)
        
        v_emb = self.valence_embedding(v_bins)
        a_emb = self.arousal_embedding(a_bins)
        
        discrete_emb = torch.cat([v_emb, a_emb], dim=-1)
        
        # Continuous residual
        va_continuous = torch.stack([valence, arousal], dim=-1)
        continuous_emb = self.continuous_proj(va_continuous)
        
        return discrete_emb + continuous_emb