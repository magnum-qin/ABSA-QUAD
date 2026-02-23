"""
DimASQP Model - Main Architecture
Hybrid model with Feature-level Rethink for SemEval-2026 Task 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment
import numpy as np

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
        va_max: float = 9.0,
        loss_weights: dict = None
    ):
        super().__init__()

        self.use_rethink = use_rethink
        assert num_rethink_iterations >= 1, "num_rethink_iterations must be >= 1"
        self.num_rethink_iterations = num_rethink_iterations
        self.num_bio_labels = num_bio_labels
        self.loss_weights = loss_weights or {
            'span': 1.0, 'pair': 1.0, 'entity': 1.0,
            'attribute': 1.0, 'va': 1.0, 'consistency': 0.5
        }
        self.num_entity_labels = num_entity_labels
        self.num_attribute_labels = num_attribute_labels
        
        # Encoder (force float32 to avoid fp16 gradient overflow)
        self.encoder = AutoModel.from_pretrained(encoder_name, dtype=torch.float32)
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
        hidden_states = encoder_outputs.last_hidden_state  # [B, L, H]
        
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
        
        # ================ Stage 2: Feature-level Rethink (num_rethink_iterations loops) ================
        if self.use_rethink:
            # Initialize iterative state from Stage 1 outputs
            cur_hidden = hidden_states
            cur_bio_probs = bio_probs_s1
            cur_pair_repr = pair_repr_s1
            cur_pair_valid_probs = pair_probs_s1
            cur_entity_probs = entity_probs_s1
            cur_attr_probs = attr_probs_s1
            cur_valence = valence_s1
            cur_arousal = arousal_s1

            for _ in range(self.num_rethink_iterations):
                refined_hidden, _ = self.rethink_module(
                    hidden_states=cur_hidden,
                    attention_mask=attention_mask,
                    bio_probs=cur_bio_probs,
                    pair_repr=cur_pair_repr,
                    pair_valid_probs=cur_pair_valid_probs,
                    entity_probs=cur_entity_probs,
                    attribute_probs=cur_attr_probs,
                    valence=cur_valence,
                    arousal=cur_arousal
                )

                bio_logits_s2 = self.span_extractor_stage2(refined_hidden)
                bio_probs_s2 = F.softmax(bio_logits_s2, dim=-1)

                # Re-extract spans with refined features (fix: use stage2 extractor)
                spans_s2 = self.span_extractor_stage2.extract_spans(bio_logits_s2, attention_mask)
                aspect_repr_s2, opinion_repr_s2, pair_mask_s2 = self._get_span_representations(
                    refined_hidden, spans_s2
                )

                pair_logits_s2, entity_logits_s2, attr_logits_s2 = self.pair_classifier_stage2(
                    aspect_repr_s2, opinion_repr_s2
                )
                pair_probs_s2 = F.softmax(pair_logits_s2, dim=-1)
                entity_probs_s2 = F.softmax(entity_logits_s2, dim=-1)
                attr_probs_s2 = F.softmax(attr_logits_s2, dim=-1)

                pair_repr_s2 = torch.cat([aspect_repr_s2, opinion_repr_s2], dim=-1)
                valence_s2, arousal_s2 = self.va_regressor_stage2(pair_repr_s2)

                # Feed this iteration's outputs into the next
                cur_hidden = refined_hidden
                cur_bio_probs = bio_probs_s2
                cur_pair_repr = pair_repr_s2
                cur_pair_valid_probs = pair_probs_s2
                cur_entity_probs = entity_probs_s2
                cur_attr_probs = attr_probs_s2
                cur_valence = valence_s2
                cur_arousal = arousal_s2

            # Update outputs with final iteration's predictions
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
    
    def _compute_pair_cost_matrix(
        self,
        pred_spans: List[Tuple[int, int, str]],
        gold_pairs: List[Dict],
        entity_logits: torch.Tensor,
        attr_logits: torch.Tensor,
        valence: torch.Tensor,
        arousal: torch.Tensor
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Compute cost matrix for Hungarian matching between predicted and gold pairs.

        Args:
            pred_spans: List of (start, end, type) from span extraction
            gold_pairs: List of gold pair dicts with aspect_start, opinion_start, etc.
            entity_logits: [num_pred_pairs, num_entities]
            attr_logits: [num_pred_pairs, num_attributes]
            valence: [num_pred_pairs]
            arousal: [num_pred_pairs]

        Returns:
            cost_matrix: [num_gold_pairs, num_pred_pairs]
            pred_pair_info: List of (asp_start, asp_end, opn_start, opn_end) for each pred pair
        """
        # Extract aspect and opinion spans from predictions
        aspects = [(s[0], s[1]) for s in pred_spans if s[2] == "aspect"]
        opinions = [(s[0], s[1]) for s in pred_spans if s[2] == "opinion"]

        # Build predicted pairs (cartesian product)
        pred_pair_info = []
        for asp_start, asp_end in aspects:
            for opn_start, opn_end in opinions:
                pred_pair_info.append((asp_start, asp_end, opn_start, opn_end))

        num_gold = len(gold_pairs)
        num_pred = len(pred_pair_info)

        if num_gold == 0 or num_pred == 0:
            return np.zeros((max(num_gold, 1), max(num_pred, 1))), pred_pair_info

        # Compute cost matrix
        cost_matrix = np.zeros((num_gold, num_pred))

        for g_idx, gold in enumerate(gold_pairs):
            gold_asp = (gold["aspect_start"], gold["aspect_end"])
            gold_opn = (gold["opinion_start"], gold["opinion_end"])

            for p_idx, (asp_s, asp_e, opn_s, opn_e) in enumerate(pred_pair_info):
                # Span matching cost (IoU-based)
                asp_iou = self._compute_span_iou(gold_asp, (asp_s, asp_e))
                opn_iou = self._compute_span_iou(gold_opn, (opn_s, opn_e))
                span_cost = 2.0 - asp_iou - opn_iou  # Lower is better

                # Classification cost
                if p_idx < entity_logits.size(0):
                    entity_prob = F.softmax(entity_logits[p_idx], dim=-1)
                    attr_prob = F.softmax(attr_logits[p_idx], dim=-1)
                    entity_cost = -torch.log(entity_prob[gold["entity"]] + 1e-8).item()
                    attr_cost = -torch.log(attr_prob[gold["attribute"]] + 1e-8).item()

                    # V-A cost
                    v_cost = abs(valence[p_idx].item() - gold["valence"])
                    a_cost = abs(arousal[p_idx].item() - gold["arousal"])
                    va_cost = (v_cost + a_cost) / 8.0  # Normalize to ~[0, 1]
                else:
                    entity_cost = 10.0
                    attr_cost = 10.0
                    va_cost = 1.0

                # Total cost (weighted sum)
                cost_matrix[g_idx, p_idx] = (
                    2.0 * span_cost +      # Span matching is most important
                    1.0 * entity_cost +
                    1.0 * attr_cost +
                    0.5 * va_cost
                )

        return cost_matrix, pred_pair_info

    def _compute_span_iou(
        self,
        span1: Tuple[int, int],
        span2: Tuple[int, int]
    ) -> float:
        """Compute IoU between two spans"""
        start1, end1 = span1
        start2, end2 = span2

        # Intersection
        inter_start = max(start1, start2)
        inter_end = min(end1, end2)
        intersection = max(0, inter_end - inter_start + 1)

        # Union
        len1 = end1 - start1 + 1
        len2 = end2 - start2 + 1
        union = len1 + len2 - intersection

        if union == 0:
            return 0.0
        return intersection / union

    def _compute_loss(
        self,
        bio_logits: torch.Tensor,
        bio_labels: torch.Tensor,
        attention_mask: torch.Tensor,
        pair_labels: List[List[Dict]],
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses with Hungarian matching for pairs"""

        device = bio_logits.device

        # BIO span loss (only on non-padded tokens) with label smoothing
        active_mask = attention_mask.view(-1) == 1
        active_logits = bio_logits.view(-1, self.num_bio_labels)[active_mask]
        active_labels = bio_labels.view(-1)[active_mask]
        bio_loss = F.cross_entropy(active_logits, active_labels, label_smoothing=0.1)

        # Pair classification and V-A losses with Hungarian matching
        pair_loss = torch.tensor(0.0, device=device)
        entity_loss = torch.tensor(0.0, device=device)
        attr_loss = torch.tensor(0.0, device=device)
        va_loss = torch.tensor(0.0, device=device)

        num_matched_pairs = 0

        num_pair_samples = 0  # count samples that contribute to pair_loss

        for b, gold_pairs in enumerate(pair_labels):
            num_pred_pairs = outputs["pair_logits"].size(1)

            # Defect 1 fix: even when gold_pairs is empty, train validity head
            # to output "all invalid" for predicted pairs (negative signal)
            if len(gold_pairs) == 0:
                valid_indices = (outputs["pair_mask"][b] > 0.5).nonzero(as_tuple=True)[0]
                if len(valid_indices) > 0:
                    pair_logits_valid = outputs["pair_logits"][b, valid_indices]
                    # All targets are 0 (invalid) since no gold pairs exist
                    pair_targets_b = torch.zeros(len(valid_indices), dtype=torch.long, device=device)
                    pair_loss += F.cross_entropy(
                        pair_logits_valid, pair_targets_b, label_smoothing=0.1
                    )
                    num_pair_samples += 1
                continue

            pred_spans = outputs["spans"][b]

            if num_pred_pairs == 0:
                continue

            # Get predictions for this sample
            entity_logits_b = outputs["entity_logits"][b]  # [num_pairs, num_entities]
            attr_logits_b = outputs["attribute_logits"][b]  # [num_pairs, num_attributes]
            valence_b = outputs["valence"][b]  # [num_pairs]
            arousal_b = outputs["arousal"][b]  # [num_pairs]

            # Compute cost matrix
            cost_matrix, pred_pair_info = self._compute_pair_cost_matrix(
                pred_spans, gold_pairs,
                entity_logits_b, attr_logits_b,
                valence_b, arousal_b
            )

            # Hungarian matching
            if cost_matrix.size > 0 and len(pred_pair_info) > 0:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)

                # Defect 4 fix: track actually used pairs (not filtered by bounds)
                actually_matched_pred_set = set()

                # Compute losses for matched pairs
                for g_idx, p_idx in zip(row_indices, col_indices):
                    if g_idx >= len(gold_pairs) or p_idx >= len(pred_pair_info):
                        continue
                    if p_idx >= num_pred_pairs:
                        continue

                    gold = gold_pairs[g_idx]

                    # Entity classification loss with label smoothing
                    entity_target = torch.tensor([gold["entity"]], device=device)
                    entity_loss += F.cross_entropy(
                        entity_logits_b[p_idx].unsqueeze(0),
                        entity_target,
                        label_smoothing=0.1
                    )

                    # Attribute classification loss with label smoothing
                    attr_target = torch.tensor([gold["attribute"]], device=device)
                    attr_loss += F.cross_entropy(
                        attr_logits_b[p_idx].unsqueeze(0),
                        attr_target,
                        label_smoothing=0.1
                    )

                    # V-A regression loss
                    v_target = torch.tensor([gold["valence"]], device=device, dtype=torch.float)
                    a_target = torch.tensor([gold["arousal"]], device=device, dtype=torch.float)

                    va_loss += F.huber_loss(valence_b[p_idx:p_idx+1], v_target)
                    va_loss += F.huber_loss(arousal_b[p_idx:p_idx+1], a_target)

                    num_matched_pairs += 1
                    actually_matched_pred_set.add(int(p_idx))

                # Pair validity loss: only pairs that passed bounds check are marked valid
                valid_indices = (outputs["pair_mask"][b] > 0.5).nonzero(as_tuple=True)[0]
                if len(valid_indices) > 0:
                    pair_logits_valid = outputs["pair_logits"][b, valid_indices]
                    pair_targets_b = torch.zeros(len(valid_indices), dtype=torch.long, device=device)
                    for local_idx, global_idx in enumerate(valid_indices.tolist()):
                        if global_idx in actually_matched_pred_set:
                            pair_targets_b[local_idx] = 1
                    pair_loss += F.cross_entropy(
                        pair_logits_valid, pair_targets_b, label_smoothing=0.1
                    )
                    num_pair_samples += 1

        # Normalize losses (Defect 2 fix: normalize pair_loss by actual contributing samples)
        num_matched_pairs = max(num_matched_pairs, 1)
        entity_loss = entity_loss / num_matched_pairs
        attr_loss = attr_loss / num_matched_pairs
        va_loss = va_loss / num_matched_pairs
        pair_loss = pair_loss / max(num_pair_samples, 1)

        # Total loss with configurable weights (Bug fix: apply loss_weights from config)
        lw = self.loss_weights
        total_loss = (
            lw['span']      * bio_loss    +
            lw['pair']      * pair_loss   +
            lw['entity']    * entity_loss +
            lw['attribute'] * attr_loss   +
            lw['va']        * va_loss
        )

        # Consistency loss for Rethink (detach s1 to stabilize)
        if self.use_rethink and "bio_logits_s1" in outputs:
            active_logits_s1 = outputs["bio_logits_s1"].view(-1, self.num_bio_labels)[active_mask]
            bio_loss_s1 = F.cross_entropy(active_logits_s1, active_labels)
            consistency_loss = self.consistency_loss(bio_loss_s1.detach(), bio_loss)
            total_loss = total_loss + lw['consistency'] * consistency_loss
        else:
            consistency_loss = torch.tensor(0.0, device=device)

        return {
            "loss": total_loss,
            "bio_loss": bio_loss,
            "pair_loss": pair_loss,
            "entity_loss": entity_loss,
            "attribute_loss": attr_loss,
            "va_loss": va_loss,
            "consistency_loss": consistency_loss
        }
    
    def _tokens_to_text(self, text: str, offset_mapping: torch.Tensor, start_tok: int, end_tok: int) -> str:
        """Convert token span indices to original text substring using offset_mapping."""
        offsets = offset_mapping.tolist()
        # Get char start from first token and char end from last token
        char_start = offsets[start_tok][0]
        char_end = offsets[end_tok][1]
        if char_start == 0 and char_end == 0:
            # Special token, no valid mapping
            return "NULL"
        return text[char_start:char_end].strip()

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        texts: List[str],
        offset_mapping: torch.Tensor = None
    ) -> List[List[Dict]]:
        """
        Generate predictions in SemEval output format

        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            texts: list of original text strings
            offset_mapping: [B, L, 2] token-to-char mapping from tokenizer

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

        entity_labels = ["RESTAURANT", "FOOD", "DRINKS", "AMBIENCE",
                        "SERVICE", "LOCATION", "NULL"]
        attr_labels = ["GENERAL", "PRICES", "QUALITY", "STYLE_OPTIONS",
                      "MISCELLANEOUS", "NULL"]

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

                    # Check pair validity (index 1 = valid, index 0 = invalid)
                    pair_logits = outputs["pair_logits"][b, pair_idx]
                    pair_valid = torch.argmax(pair_logits).item()
                    # Also check confidence - skip low confidence predictions
                    pair_prob = torch.softmax(pair_logits, dim=-1)
                    if pair_valid == 0 or pair_prob[1] < 0.3:  # Invalid or low confidence
                        pair_idx += 1
                        continue

                    # Get predictions
                    entity_idx = torch.argmax(outputs["entity_logits"][b, pair_idx]).item()
                    attr_idx = torch.argmax(outputs["attribute_logits"][b, pair_idx]).item()
                    valence = outputs["valence"][b, pair_idx].item()
                    arousal = outputs["arousal"][b, pair_idx].item()

                    # Extract text spans using offset_mapping
                    if offset_mapping is not None:
                        aspect_text = self._tokens_to_text(texts[b], offset_mapping[b], asp_start, asp_end)
                        opinion_text = self._tokens_to_text(texts[b], offset_mapping[b], opn_start, opn_end)
                    else:
                        aspect_text = "NULL"
                        opinion_text = "NULL"

                    quads.append({
                        "Aspect": aspect_text,
                        "Category": f"{entity_labels[entity_idx]}#{attr_labels[attr_idx]}",
                        "Opinion": opinion_text,
                        "VA": f"{valence:.2f}#{arousal:.2f}"
                    })

                    pair_idx += 1

            predictions.append(quads)

        return predictions
