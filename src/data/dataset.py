"""
DimASQP Data Module
Handles data loading, preprocessing, and dataset creation for SemEval-2026 Task 3
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional
import re


class DimASQPDataset(Dataset):
    """
    Dataset for Dimensional Aspect Sentiment Quad Prediction

    Each sample contains:
    - text: input sentence
    - quadruplets: list of (aspect, category, opinion, va) tuples
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        entity_labels: List[str] = None,
        attribute_labels: List[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.entity_labels = entity_labels or [
            "RESTAURANT", "FOOD", "DRINKS", "AMBIENCE",
            "SERVICE", "LOCATION", "NULL"
        ]
        self.attribute_labels = attribute_labels or [
            "GENERAL", "PRICES", "QUALITY", "STYLE_OPTIONS",
            "MISCELLANEOUS", "NULL"
        ]

        # Build label mappings
        self.entity2id = {label: i for i, label in enumerate(self.entity_labels)}
        self.attribute2id = {label: i for i, label in enumerate(self.attribute_labels)}
        self.bio_labels = ["O", "B-ASP", "I-ASP", "B-OPN", "I-OPN"]
        self.bio2id = {label: i for i, label in enumerate(self.bio_labels)}

        # Track used character positions per sample to handle duplicate spans
        self._used_char_positions = set()
        
        # Load data
        self.samples = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load JSONL data file"""
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                samples.append(self._process_item(item))
        return samples
    
    def _process_item(self, item: Dict) -> Dict:
        """Process a single data item"""
        text = item["Text"]
        sample_id = item["ID"]
        
        # Parse quadruplets if present (for training/dev)
        quadruplets = []
        if "Quadruplet" in item:
            for quad in item["Quadruplet"]:
                aspect = quad["Aspect"]
                category = quad["Category"]
                opinion = quad["Opinion"]
                va_str = quad["VA"]
                
                # Parse V#A format
                valence, arousal = self._parse_va(va_str)
                
                # Parse ENTITY#ATTRIBUTE
                entity, attribute = self._parse_category(category)
                
                quadruplets.append({
                    "aspect": aspect,
                    "opinion": opinion,
                    "entity": entity,
                    "attribute": attribute,
                    "valence": valence,
                    "arousal": arousal
                })
        
        return {
            "id": sample_id,
            "text": text,
            "quadruplets": quadruplets
        }
    
    def _parse_va(self, va_str: str) -> Tuple[float, float]:
        """Parse V#A format string"""
        parts = va_str.split("#")
        valence = float(parts[0])
        arousal = float(parts[1])
        return valence, arousal
    
    def _parse_category(self, category: str) -> Tuple[str, str]:
        """Parse ENTITY#ATTRIBUTE format"""
        parts = category.split("#")
        entity = parts[0]
        attribute = parts[1] if len(parts) > 1 else "GENERAL"
        return entity, attribute
    
    def _find_all_occurrences(self, text: str, span: str) -> List[Tuple[int, int]]:
        """Find all occurrences of span in text (case-insensitive)"""
        occurrences = []
        text_lower = text.lower()
        span_lower = span.lower().strip()

        if not span_lower:
            return occurrences

        start = 0
        while True:
            pos = text_lower.find(span_lower, start)
            if pos == -1:
                break
            occurrences.append((pos, pos + len(span_lower)))
            start = pos + 1

        return occurrences

    def _find_span_positions(
        self, text: str, span: str, encoding, used_positions: set
    ) -> Tuple[int, int]:
        """
        Find token positions of a span in the text, avoiding already-used positions.

        Args:
            text: Original text
            span: Span text to find
            encoding: Tokenizer encoding result
            used_positions: Set of (start_char, end_char) tuples already used

        Returns:
            (start_token, end_token) or (-1, -1) if not found
        """
        # Find all occurrences
        occurrences = self._find_all_occurrences(text, span)

        if not occurrences:
            return -1, -1

        # Try each occurrence, prefer unused ones
        for start_char, end_char in occurrences:
            # Check if this position overlaps with any used position
            is_used = False
            for used_start, used_end in used_positions:
                # Check for overlap
                if not (end_char <= used_start or start_char >= used_end):
                    is_used = True
                    break

            if is_used:
                continue

            # Convert to token positions
            start_token = encoding.char_to_token(start_char)
            end_token = encoding.char_to_token(end_char - 1)

            if start_token is not None and end_token is not None:
                # Mark this position as used
                used_positions.add((start_char, end_char))
                return start_token, end_token

        # If all occurrences are used, fall back to first valid occurrence
        for start_char, end_char in occurrences:
            start_token = encoding.char_to_token(start_char)
            end_token = encoding.char_to_token(end_char - 1)
            if start_token is not None and end_token is not None:
                return start_token, end_token

        return -1, -1
    
    def _create_bio_labels(self, seq_len: int, spans: List[Tuple[int, int, str]]) -> List[int]:
        """Create BIO labels for aspect and opinion spans"""
        labels = [self.bio2id["O"]] * seq_len
        
        for start, end, span_type in spans:
            if start < 0 or end < 0 or start >= seq_len or end >= seq_len:
                continue
            
            if span_type == "aspect":
                labels[start] = self.bio2id["B-ASP"]
                for i in range(start + 1, end + 1):
                    if i < seq_len:
                        labels[i] = self.bio2id["I-ASP"]
            elif span_type == "opinion":
                labels[start] = self.bio2id["B-OPN"]
                for i in range(start + 1, end + 1):
                    if i < seq_len:
                        labels[i] = self.bio2id["I-OPN"]
        
        return labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        text = sample["text"]
        quadruplets = sample["quadruplets"]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels
        seq_len = self.max_length

        # Track used character positions for this sample
        used_aspect_positions = set()
        used_opinion_positions = set()

        # BIO labels for spans
        spans = []
        pair_labels = []  # (aspect_idx, opinion_idx, entity, attribute, valence, arousal)

        for quad in quadruplets:
            aspect_pos = self._find_span_positions(
                text, quad["aspect"], encoding, used_aspect_positions
            )
            opinion_pos = self._find_span_positions(
                text, quad["opinion"], encoding, used_opinion_positions
            )

            if aspect_pos[0] >= 0:
                spans.append((aspect_pos[0], aspect_pos[1], "aspect"))
            if opinion_pos[0] >= 0:
                spans.append((opinion_pos[0], opinion_pos[1], "opinion"))

            # Store pair information
            if aspect_pos[0] >= 0 and opinion_pos[0] >= 0:
                pair_labels.append({
                    "aspect_start": aspect_pos[0],
                    "aspect_end": aspect_pos[1],
                    "opinion_start": opinion_pos[0],
                    "opinion_end": opinion_pos[1],
                    "entity": self.entity2id.get(quad["entity"], 0),
                    "attribute": self.attribute2id.get(quad["attribute"], 0),
                    "valence": quad["valence"],
                    "arousal": quad["arousal"]
                })

        bio_labels = self._create_bio_labels(seq_len, spans)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": encoding["offset_mapping"].squeeze(0),
            "bio_labels": torch.tensor(bio_labels, dtype=torch.long),
            "pair_labels": pair_labels,
            "sample_id": sample["id"],
            "text": text,
            "quadruplets": quadruplets
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    bio_labels = torch.stack([item["bio_labels"] for item in batch])
    
    offset_mapping = torch.stack([item["offset_mapping"] for item in batch])

    # Pair labels need special handling (variable length)
    pair_labels = [item["pair_labels"] for item in batch]
    sample_ids = [item["sample_id"] for item in batch]
    texts = [item["text"] for item in batch]
    quadruplets = [item["quadruplets"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offset_mapping": offset_mapping,
        "bio_labels": bio_labels,
        "pair_labels": pair_labels,
        "sample_ids": sample_ids,
        "texts": texts,
        "quadruplets": quadruplets
    }


def create_dataloader(
    data_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_length: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    entity_labels: list = None,
    attribute_labels: list = None
) -> DataLoader:
    """Create a DataLoader for DimASQP"""
    import platform
    dataset = DimASQPDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        entity_labels=entity_labels,
        attribute_labels=attribute_labels
    )

    use_pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory
    )