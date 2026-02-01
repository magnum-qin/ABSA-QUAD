"""
Feature-level Rethink Module
Enhances hidden representations by incorporating Stage 1 predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class LabelEmbeddingModule(nn.Module):
    """
    Converts Stage 1 predictions to learnable embeddings
    """
    
    def __init__(
        self,
        num_bio_labels: int = 5,  # O, B-ASP, I-ASP, B-OPN, I-OPN
        num_entity_labels: int = 7,
        num_attribute_labels: int = 6,
        embedding_dim: int = 64,
        hidden_dim: int = 768,
        pair_dim: int = None
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        pair_dim = pair_dim or hidden_dim * 2

        # BIO label embeddings (for token-level)
        self.bio_embedding = nn.Embedding(num_bio_labels, embedding_dim)

        # Category embeddings (for pair-level)
        self.entity_embedding = nn.Embedding(num_entity_labels, embedding_dim)
        self.attribute_embedding = nn.Embedding(num_attribute_labels, embedding_dim)

        # Pair validity embedding
        self.pair_embedding = nn.Embedding(2, embedding_dim)  # valid/invalid

        # VA embedding projector
        self.va_projector = nn.Linear(2, embedding_dim)

        # Fusion layer
        self.token_fusion = nn.Linear(hidden_dim + embedding_dim, hidden_dim)
        self.pair_fusion = nn.Linear(pair_dim + embedding_dim * 4, hidden_dim)
    
    def embed_bio_labels(self, bio_probs: torch.Tensor) -> torch.Tensor:
        """
        Convert BIO probabilities to embeddings using soft embedding
        
        Args:
            bio_probs: [batch_size, seq_len, num_bio_labels]
            
        Returns:
            bio_emb: [batch_size, seq_len, embedding_dim]
        """
        # Soft embedding: weighted sum of embeddings
        bio_emb = torch.matmul(bio_probs, self.bio_embedding.weight)
        return bio_emb
    
    def embed_pair_labels(
        self,
        pair_valid_probs: torch.Tensor,
        entity_probs: torch.Tensor,
        attribute_probs: torch.Tensor,
        valence: torch.Tensor,
        arousal: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert pair-level predictions to embeddings
        
        Args:
            pair_valid_probs: [batch_size, num_pairs, 2]
            entity_probs: [batch_size, num_pairs, num_entities]
            attribute_probs: [batch_size, num_pairs, num_attributes]
            valence, arousal: [batch_size, num_pairs]
            
        Returns:
            pair_emb: [batch_size, num_pairs, embedding_dim * 4]
        """
        # Soft embeddings for categorical predictions
        pair_emb = torch.matmul(pair_valid_probs, self.pair_embedding.weight)
        entity_emb = torch.matmul(entity_probs, self.entity_embedding.weight)
        attribute_emb = torch.matmul(attribute_probs, self.attribute_embedding.weight)
        
        # VA embedding
        va_values = torch.stack([valence, arousal], dim=-1)
        va_emb = self.va_projector(va_values)
        
        # Concatenate all embeddings
        combined = torch.cat([pair_emb, entity_emb, attribute_emb, va_emb], dim=-1)
        
        return combined
    
    def fuse_token_features(
        self,
        hidden_states: torch.Tensor,
        bio_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse token-level hidden states with BIO embeddings
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            bio_emb: [batch_size, seq_len, embedding_dim]
            
        Returns:
            refined_hidden: [batch_size, seq_len, hidden_dim]
        """
        combined = torch.cat([hidden_states, bio_emb], dim=-1)
        refined = self.token_fusion(combined)
        return refined
    
    def fuse_pair_features(
        self,
        pair_repr: torch.Tensor,
        pair_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse pair-level representations with label embeddings
        
        Args:
            pair_repr: [batch_size, num_pairs, hidden_dim]
            pair_emb: [batch_size, num_pairs, embedding_dim * 4]
            
        Returns:
            refined_pair: [batch_size, num_pairs, hidden_dim]
        """
        combined = torch.cat([pair_repr, pair_emb], dim=-1)
        refined = self.pair_fusion(combined)
        return refined


class FeatureLevelRethink(nn.Module):
    """
    Feature-level Rethink Module
    
    Takes Stage 1 predictions and hidden states, produces refined features
    for Stage 2 re-prediction
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        pair_dim: int = None,
        num_bio_labels: int = 5,
        num_entity_labels: int = 7,
        num_attribute_labels: int = 6,
        label_embedding_dim: int = 64,
        num_attention_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        pair_dim = pair_dim or hidden_dim * 2

        # Label embedding module
        self.label_embedding = LabelEmbeddingModule(
            num_bio_labels=num_bio_labels,
            num_entity_labels=num_entity_labels,
            num_attribute_labels=num_attribute_labels,
            embedding_dim=label_embedding_dim,
            hidden_dim=hidden_dim,
            pair_dim=pair_dim
        )
        
        # Self-attention for refined features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        bio_probs: torch.Tensor,
        pair_repr: torch.Tensor,
        pair_valid_probs: torch.Tensor,
        entity_probs: torch.Tensor,
        attribute_probs: torch.Tensor,
        valence: torch.Tensor,
        arousal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Feature-level Rethink
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
            bio_probs: [batch_size, seq_len, num_bio_labels]
            pair_repr: [batch_size, num_pairs, hidden_dim]
            pair_valid_probs: [batch_size, num_pairs, 2]
            entity_probs: [batch_size, num_pairs, num_entities]
            attribute_probs: [batch_size, num_pairs, num_attributes]
            valence, arousal: [batch_size, num_pairs]
            
        Returns:
            refined_hidden: [batch_size, seq_len, hidden_dim]
            refined_pair: [batch_size, num_pairs, hidden_dim]
        """
        # Get label embeddings
        bio_emb = self.label_embedding.embed_bio_labels(bio_probs)
        pair_emb = self.label_embedding.embed_pair_labels(
            pair_valid_probs, entity_probs, attribute_probs, valence, arousal
        )
        
        # Fuse with original features
        fused_hidden = self.label_embedding.fuse_token_features(hidden_states, bio_emb)
        fused_pair = self.label_embedding.fuse_pair_features(pair_repr, pair_emb)
        
        # Self-attention refinement for token features
        key_padding_mask = (attention_mask == 0)
        attn_output, _ = self.self_attention(
            fused_hidden, fused_hidden, fused_hidden,
            key_padding_mask=key_padding_mask
        )
        
        # Residual connection and layer norm
        fused_hidden = self.norm1(fused_hidden + attn_output)
        
        # FFN
        ffn_output = self.ffn(fused_hidden)
        refined_hidden = self.norm2(fused_hidden + ffn_output)
        
        # Gated fusion with original hidden states
        gate_input = torch.cat([hidden_states, refined_hidden], dim=-1)
        gate_weights = self.gate(gate_input)
        refined_hidden = gate_weights * refined_hidden + (1 - gate_weights) * hidden_states
        
        return refined_hidden, fused_pair


class ConsistencyLoss(nn.Module):
    """
    Ensures Stage 2 predictions are better than Stage 1
    """
    
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        stage1_loss: torch.Tensor,
        stage2_loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Margin loss: stage2_loss should be lower than stage1_loss
        
        L_consistency = max(0, stage2_loss - stage1_loss + margin)
        """
        return F.relu(stage2_loss - stage1_loss + self.margin)