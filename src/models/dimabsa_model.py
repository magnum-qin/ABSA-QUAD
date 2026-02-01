"""
DimASQP Model - Main Architecture
Hybrid model with Feature-level Rethink for SemEval-2026 Task 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Optional, Tuple

from .va_module import VARegressor, VAEmbedding
from .feature_rethink import FeatureLevelRethink, ConsistencyLoss


class SpanExtractor(nn.Module):
    """
    Extracts aspect and opinion spans using BIO tagging
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_labels: int = 5,  # O, B-ASP, I-ASP, B-OPN, I-OPN
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_labels = num_labels
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            
        Returns:
            logits: [batch_size, seq_len, num_labels]
        """
        return self.classifier(hidden_states)
    
    def extract_spans(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[List[Tuple[int, int, str]]]:
        """
        Extract span positions from BIO predictions
        
        Returns:
            List of spans for each sample: [(start, end, type), ...]
        """
        predictions = torch.argmax(logits, dim=-1)  # [B, L]
        batch_spans = []
        
        # Label mapping
        label_names = ["O", "B-ASP", "I-ASP", "B-OPN", "I-OPN"]
        
        for b in range(predictions.size(0)):
            spans = []
            current_span = None
            
            for i in range(predictions.size(1)):
                if attention_mask[b, i] == 0:
                    break
                    
                label = label_names[predictions[b, i].item()]
                
                if label.startswith("B-"):
                    # Save previous span if exists
                    if current_span is not None:
                        spans.append(current_span)
                    
                    span_type = "aspect" if "ASP" in label else "opinion"
                    current_span = (i, i, span_type)
                    
                elif label.startswith("I-"):
                    if current_span is not None:
                        span_type = "aspect" if "ASP" in label else "opinion"
                        if current_span[2] == span_type:
                            current_span = (current_span[0], i, span_type)
                        else:
                            spans.append(current_span)
                            current_span = None
                else:  # O label
                    if current_span is not None:
                        spans.append(current_span)
                        current_span = None
            
            # Don't forget the last span
            if current_span is not None:
                spans.append(current_span)
            
            batch_spans.append(spans)
        
        return batch_spans


class PairClassifier(nn.Module):
    """
    Classifies aspect-opinion pairs and predicts categories
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_entity_labels: int = 7,
        num_attribute_labels: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Pair validity classifier
        self.pair_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # valid / invalid
        )
        
        # Entity classifier
        self.entity_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_entity_labels)
        )
        
        # Attribute classifier
        self.attribute_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_attribute_labels)
        )
    
    def forward(
        self,
        aspect_repr: torch.Tensor,
        opinion_repr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            aspect_repr: [batch_size, num_pairs, hidden_dim]
            opinion_repr: [batch_size, num_pairs, hidden_dim]
            
        Returns:
            pair_logits: [batch_size, num_pairs, 2]
            entity_logits: [batch_size, num_pairs, num_entities]
            attribute_logits: [batch_size, num_pairs, num_attributes]
        """
        # Concatenate aspect and opinion representations
        pair_repr = torch.cat([aspect_repr, opinion_repr], dim=-1)
        
        pair_logits = self.pair_classifier(pair_repr)
        entity_logits = self.entity_classifier(pair_repr)
        attribute_logits = self.attribute_classifier(pair_repr)
        
        return pair_logits, entity_logits, attribute_logits


class DimASQPModel(nn.Module):
    """
    Main DimASQP Model with Feature-level Rethink
    
    Architecture:
    Stage 1: Draft Prediction
        - Encoder (DeBERTa/BERT)
        - Span Extraction (BIO)
        - Pair Classification
        - V-A Regression
    
    Stage 2: Feature-level Rethink
        - Label Embeddings from Stage 1
        - Refined Hidden States
        - All heads re-predict
    """
    
    def __init__(
        self,
        encoder_name: str = "microsoft/deberta-v3-base",
        num_bio_labels: int = 5,
        num_entity_labels: int = 7,
        num_attribute_labels: int = 6,
        hidden_dim: int = 768,
        va_hidden_dim: int = 256,
        label_embedding_dim: int = 64,
        dropout: float = 0.1,
        use_rethink: bool = True,
        num_rethink_iterations: int = 1,
        va_min: float = 1.0,
        va_max: float = 9.0
    ):
        super().__init__()
        
        self.use_rethink = use_rethink
        self.num_rethink_iterations = num_rethink_iterations
        self.num_bio_labels = num_bio_labels
        self.num_entity_labels = num_entity_labels
        self.num_attribute_labels = num_attribute_labels
        
        # Encoder
        self.encoder = AutoModel.from_pretrained(encoder_name)
        config = AutoConfig.from_pretrained(encoder_name)
        self.hidden_dim = config.hidden_size
        
        # Stage 1: Draft Prediction Heads
        self.span_extractor = SpanExtractor(
            hidden_dim=self.hidden_dim,
            num_labels=num_bio_labels,
            dropout=dropout
        )
        
        self.pair_classifier = PairClassifier(
            hidden_dim=self.hidden_dim,
            num_entity_labels=num_entity_labels,
            num_attribute_labels=num_attribute_labels,
            dropout=dropout
        )
        
        self.va_regressor = VARegressor(
            input_dim=self.hidden_dim * 2,
            hidden_dim=va_hidden_dim,
            dropout=dropout,
            va_min=va_min,
            va_max=va_max
        )
        
        # Stage 2: Feature-level Rethink
        if use_rethink:
            self.rethink_module = FeatureLevelRethink(
                hidden_dim=self.hidden_dim,
                num_bio_labels=num_bio_labels,
                num_entity_labels=num_entity_labels,
                num_attribute_labels=num_attribute_labels,
                label_embedding_dim=label_embedding_dim,
                dropout=dropout
            )
            
            # Stage 2 heads (can share weights with Stage 1 or be separate)
            self.span_extractor_stage2 = SpanExtractor(
                hidden_dim=self.hidden_dim,
                num_labels=num_bio_labels,
                dropout=dropout
            )
            
            self.pair_classifier_stage2 = PairClassifier(
                hidden_dim=self.hidden_dim,
                num_entity_labels=num_entity_labels,
                num_attribute_labels=num_attribute_labels,
                dropout=dropout
            )
            
            self.va_regressor_stage2 = VARegressor(
                input_dim=self.hidden_dim * 2,
                hidden_dim=va_hidden_dim,
                dropout=dropout,
                va_min=va_min,
                va_max=va_max
            )
            
            # Consistency loss
            self.consistency_loss = ConsistencyLoss(margin=0.1)
    
    def _get_span_representations(
        self,
        hidden_states: torch.Tensor,
        spans: List[List[Tuple[int, int, str]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get representations for aspect and opinion spans
        
        Returns:
            aspect_repr: [batch_size, max_pairs, hidden_dim]
            opinion_repr: [batch_size, max_pairs, hidden_dim]
            pair_mask: [batch_size, max_pairs]
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        
        # Find max number of pairs
        max_pairs = 0
        for batch_spans in spans:
            aspects = [s for s in batch_spans if s[2] == "aspect"]
            opinions = [s for s in batch_spans if s[2] == "opinion"]
            max_pairs = max(max_pairs, len(aspects) * len(opinions))
        
        max_pairs = max(max_pairs, 1)  # At least 1 to avoid empty tensors
        
        aspect_repr = torch.zeros(batch_size, max_pairs, self.hidden_dim, device=device)
        opinion_repr = torch.zeros(batch_size, max_pairs, self.hidden_dim, device=device)
        pair_mask = torch.zeros(batch_size, max_pairs, device=device)
        
        for b, batch_spans in enumerate(spans):
            aspects = [s for s in batch_spans if s[2] == "aspect"]
            opinions = [s for s in batch_spans if s[2] == "opinion"]
            
            pair_idx = 0
            for asp_start, asp_end, _ in aspects:
                # Average pooling for aspect span
                asp_repr = hidden_states[b, asp_start:asp_end+1].mean(dim=0)
                
                for opn_start, opn_end, _ in opinions:
                    if pair_idx >= max_pairs:
                        break
                    
                    # Average pooling for opinion span
                    opn_repr = hidden_states[b, opn_start:opn_end+1].mean(dim=0)
                    
                    aspect_repr[b, pair_idx] = asp_repr
                    opinion_repr[b, pair_idx] = opn_repr
                    pair_mask[b, pair_idx] = 1
                    pair_idx += 1
        
        return aspect_repr, opinion_repr, pair_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bio_labels: Optional[torch.Tensor] = None,
        pair_labels: Optional[List[List[Dict]]] = None,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            bio_labels: [batch_size, seq_len] - BIO labels for training
            pair_labels: List of pair information for training
            return_loss: Whether to compute and return loss
            
        Returns:
            Dictionary containing predictions and losses
        """
        # Encode
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = encoder_outputs.last_hidden_state.float()  # [B, L, H]
        
        # ================ Stage 1: Draft Prediction ================
        # Span extraction
        bio_logits_s1 = self.span_extractor(hidden_states)
        bio_probs_s1 = F.softmax(bio_logits_s1, dim=-1)
        
        # Extract spans
        spans = self.span_extractor.extract_spans(bio_logits_s1, attention_mask)
        
        # Get span representations
        aspect_repr, opinion_repr, pair_mask = self._get_span_representations(
            hidden_states, spans
        )
        
        # Pair classification
        pair_logits_s1, entity_logits_s1, attr_logits_s1 = self.pair_classifier(
            aspect_repr, opinion_repr
        )
        pair_probs_s1 = F.softmax(pair_logits_s1, dim=-1)
        entity_probs_s1 = F.softmax(entity_logits_s1, dim=-1)
        attr_probs_s1 = F.softmax(attr_logits_s1, dim=-1)
        
        # V-A regression
        pair_repr_s1 = torch.cat([aspect_repr, opinion_repr], dim=-1)
        valence_s1, arousal_s1 = self.va_regressor(pair_repr_s1)
        
        outputs = {
            "bio_logits": bio_logits_s1,
            "pair_logits": pair_logits_s1,
            "entity_logits": entity_logits_s1,
            "attribute_logits": attr_logits_s1,
            "valence": valence_s1,
            "arousal": arousal_s1,
            "pair_mask": pair_mask,
            "spans": spans
        }
        
        # ================ Stage 2: Feature-level Rethink ================
        if self.use_rethink:
            # Apply rethink module
            refined_hidden, refined_pair_repr = self.rethink_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                bio_probs=bio_probs_s1,
                pair_repr=pair_repr_s1,
                pair_valid_probs=pair_probs_s1,
                entity_probs=entity_probs_s1,
                attribute_probs=attr_probs_s1,
                valence=valence_s1,
                arousal=arousal_s1
            )
            
            # Stage 2 predictions
            bio_logits_s2 = self.span_extractor_stage2(refined_hidden)
            
            # Re-extract spans with refined features
            spans_s2 = self.span_extractor.extract_spans(bio_logits_s2, attention_mask)
            aspect_repr_s2, opinion_repr_s2, pair_mask_s2 = self._get_span_representations(
                refined_hidden, spans_s2
            )
            
            pair_logits_s2, entity_logits_s2, attr_logits_s2 = self.pair_classifier_stage2(
                aspect_repr_s2, opinion_repr_s2
            )
            
            pair_repr_s2 = torch.cat([aspect_repr_s2, opinion_repr_s2], dim=-1)
            valence_s2, arousal_s2 = self.va_regressor_stage2(pair_repr_s2)
            
            # Update outputs with Stage 2 predictions
            outputs.update({
                "bio_logits_s1": bio_logits_s1,
                "bio_logits": bio_logits_s2,
                "pair_logits_s1": pair_logits_s1,
                "pair_logits": pair_logits_s2,
                "entity_logits_s1": entity_logits_s1,
                "entity_logits": entity_logits_s2,
                "attribute_logits_s1": attr_logits_s1,
                "attribute_logits": attr_logits_s2,
                "valence_s1": valence_s1,
                "valence": valence_s2,
                "arousal_s1": arousal_s1,
                "arousal": arousal_s2,
                "pair_mask": pair_mask_s2,
                "spans": spans_s2
            })
        
        # ================ Compute Loss ================
        if return_loss and bio_labels is not None:
            loss_dict = self._compute_loss(
                bio_logits=outputs["bio_logits"],
                bio_labels=bio_labels,
                attention_mask=attention_mask,
                pair_labels=pair_labels,
                outputs=outputs
            )
            outputs.update(loss_dict)
        
        return outputs
    
    def _compute_loss(
        self,
        bio_logits: torch.Tensor,
        bio_labels: torch.Tensor,
        attention_mask: torch.Tensor,
        pair_labels: List[List[Dict]],
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses"""
        
        # BIO span loss
        bio_loss = F.cross_entropy(
            bio_logits.view(-1, self.num_bio_labels),
            bio_labels.view(-1),
            ignore_index=-100
        )
        
        # Pair classification and V-A losses
        pair_loss = torch.tensor(0.0, device=bio_logits.device)
        entity_loss = torch.tensor(0.0, device=bio_logits.device)
        attr_loss = torch.tensor(0.0, device=bio_logits.device)
        va_loss = torch.tensor(0.0, device=bio_logits.device)
        
        num_pairs = 0
        
        for b, pairs in enumerate(pair_labels):
            for pair in pairs:
                # This is a valid pair
                # For simplicity, we'll compute losses on the gold pairs
                # In practice, you'd need to match predicted spans to gold spans
                
                entity_target = torch.tensor([pair["entity"]], device=bio_logits.device)
                attr_target = torch.tensor([pair["attribute"]], device=bio_logits.device)
                
                # Approximate: use first pair slot
                if num_pairs < outputs["pair_logits"].size(1):
                    entity_loss += F.cross_entropy(
                        outputs["entity_logits"][b:b+1, num_pairs:num_pairs+1].squeeze(1),
                        entity_target
                    )
                    attr_loss += F.cross_entropy(
                        outputs["attribute_logits"][b:b+1, num_pairs:num_pairs+1].squeeze(1),
                        attr_target
                    )
                    
                    # V-A loss
                    v_target = torch.tensor([pair["valence"]], device=bio_logits.device)
                    a_target = torch.tensor([pair["arousal"]], device=bio_logits.device)
                    
                    va_loss += F.huber_loss(
                        outputs["valence"][b, num_pairs:num_pairs+1], v_target
                    )
                    va_loss += F.huber_loss(
                        outputs["arousal"][b, num_pairs:num_pairs+1], a_target
                    )
                
                num_pairs += 1
        
        # Normalize
        num_pairs = max(num_pairs, 1)
        entity_loss = entity_loss / num_pairs
        attr_loss = attr_loss / num_pairs
        va_loss = va_loss / num_pairs
        
        # Total loss
        total_loss = bio_loss + pair_loss + entity_loss + attr_loss + va_loss
        
        # Consistency loss for Rethink
        if self.use_rethink and "bio_logits_s1" in outputs:
            bio_loss_s1 = F.cross_entropy(
                outputs["bio_logits_s1"].view(-1, self.num_bio_labels),
                bio_labels.view(-1),
                ignore_index=-100
            )
            consistency_loss = self.consistency_loss(bio_loss_s1, bio_loss)
            total_loss = total_loss + 0.5 * consistency_loss
        else:
            consistency_loss = torch.tensor(0.0, device=bio_logits.device)
        
        return {
            "loss": total_loss,
            "bio_loss": bio_loss,
            "pair_loss": pair_loss,
            "entity_loss": entity_loss,
            "attribute_loss": attr_loss,
            "va_loss": va_loss,
            "consistency_loss": consistency_loss
        }
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        texts: List[str]
    ) -> List[List[Dict]]:
        """
        Generate predictions in SemEval output format
        
        Returns:
            List of quadruplets for each sample
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_loss=False
            )
        
        predictions = []
        
        for b in range(input_ids.size(0)):
            quads = []
            spans = outputs["spans"][b]
            
            aspects = [s for s in spans if s[2] == "aspect"]
            opinions = [s for s in spans if s[2] == "opinion"]
            
            pair_idx = 0
            for asp_start, asp_end, _ in aspects:
                for opn_start, opn_end, _ in opinions:
                    if outputs["pair_mask"][b, pair_idx] < 0.5:
                        pair_idx += 1
                        continue
                    
                    # Check pair validity
                    pair_valid = torch.argmax(outputs["pair_logits"][b, pair_idx]).item()
                    if pair_valid == 0:  # Invalid pair
                        pair_idx += 1
                        continue
                    
                    # Get predictions
                    entity_idx = torch.argmax(outputs["entity_logits"][b, pair_idx]).item()
                    attr_idx = torch.argmax(outputs["attribute_logits"][b, pair_idx]).item()
                    valence = outputs["valence"][b, pair_idx].item()
                    arousal = outputs["arousal"][b, pair_idx].item()
                    
                    # Extract text spans
                    # Note: This is simplified; you'd need proper token-to-text mapping
                    entity_labels = ["RESTAURANT", "FOOD", "DRINKS", "AMBIENCE", 
                                    "SERVICE", "LOCATION", "NULL"]
                    attr_labels = ["GENERAL", "PRICES", "QUALITY", "STYLE_OPTIONS", 
                                  "MISCELLANEOUS", "NULL"]
                    
                    quads.append({
                        "Aspect": texts[b][asp_start:asp_end+1] if asp_end < len(texts[b]) else "NULL",
                        "Category": f"{entity_labels[entity_idx]}#{attr_labels[attr_idx]}",
                        "Opinion": texts[b][opn_start:opn_end+1] if opn_end < len(texts[b]) else "NULL",
                        "VA": f"{valence:.2f}#{arousal:.2f}"
                    })
                    
                    pair_idx += 1
            
            predictions.append(quads)
        
        return predictions
