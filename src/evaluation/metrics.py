"""
Evaluation Metrics for DimASQP
Implements continuous F1 as defined in SemEval-2026 Task 3
"""

import math
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class DimASQPEvaluator:
    """
    Evaluator for Dimensional Aspect Sentiment Quad Prediction
    
    Implements:
    - Continuous F1 (cF1) for quadruplet extraction
    - RMSE for V-A regression
    - Standard Exact Match F1
    """
    
    def __init__(
        self,
        va_min: float = 1.0,
        va_max: float = 9.0
    ):
        self.va_min = va_min
        self.va_max = va_max
        self.d_max = math.sqrt(2 * (va_max - va_min) ** 2)  # sqrt(128) ≈ 11.31
        
        self.reset()
    
    def reset(self):
        """Reset all counters"""
        self.predictions = []
        self.gold_labels = []
        
        # For RMSE calculation
        self.valence_errors = []
        self.arousal_errors = []
    
    def add_batch(
        self,
        predictions: List[List[Dict]],
        gold_labels: List[List[Dict]]
    ):
        """
        Add a batch of predictions and gold labels
        
        Args:
            predictions: List of predicted quadruplets per sample
            gold_labels: List of gold quadruplets per sample
        """
        self.predictions.extend(predictions)
        self.gold_labels.extend(gold_labels)
    
    def _parse_va(self, va_str: str) -> Tuple[float, float]:
        """Parse V#A format string"""
        parts = va_str.split("#")
        return float(parts[0]), float(parts[1])
    
    def _normalize_quad(self, quad: Dict) -> Tuple[str, str, str]:
        """Normalize quadruplet for categorical matching"""
        aspect = quad.get("Aspect", quad.get("aspect", "")).lower().strip()
        category = quad.get("Category", quad.get("category", "")).upper().strip()
        opinion = quad.get("Opinion", quad.get("opinion", "")).lower().strip()
        return (aspect, category, opinion)
    
    def _compute_va_distance(
        self,
        pred_va: Tuple[float, float],
        gold_va: Tuple[float, float]
    ) -> float:
        """
        Compute normalized Euclidean distance in VA space
        
        dist = sqrt((V_p - V_g)^2 + (A_p - A_g)^2) / D_max
        """
        v_diff = pred_va[0] - gold_va[0]
        a_diff = pred_va[1] - gold_va[1]
        distance = math.sqrt(v_diff ** 2 + a_diff ** 2)
        normalized = distance / self.d_max
        return min(normalized, 1.0)  # Cap at 1.0
    
    def compute_continuous_f1(self) -> Dict[str, float]:
        """
        Compute continuous F1 score as defined in SemEval-2026
        
        cTP = 1 - dist(VA_pred, VA_gold) for categorically matched quads
        cRecall = sum(cTP) / (TP_cat + FN_cat)
        cPrecision = sum(cTP) / (TP_cat + FP_cat)
        cF1 = 2 * cRecall * cPrecision / (cRecall + cPrecision)
        """
        total_ctp = 0.0
        total_tp_cat = 0
        total_fn_cat = 0
        total_fp_cat = 0
        
        for pred_quads, gold_quads in zip(self.predictions, self.gold_labels):
            # Create normalized representations
            pred_set = {}
            for quad in pred_quads:
                key = self._normalize_quad(quad)
                va_str = quad.get("VA", quad.get("va", "5.0#5.0"))
                pred_set[key] = self._parse_va(va_str)
            
            gold_set = {}
            for quad in gold_quads:
                key = self._normalize_quad(quad)
                va_str = quad.get("VA", quad.get("va", "5.0#5.0"))
                gold_set[key] = self._parse_va(va_str)
            
            # Find matches
            matched_gold = set()
            matched_pred = set()
            
            for pred_key, pred_va in pred_set.items():
                if pred_key in gold_set:
                    gold_va = gold_set[pred_key]
                    
                    # Compute continuous TP
                    distance = self._compute_va_distance(pred_va, gold_va)
                    ctp = 1.0 - distance
                    total_ctp += ctp
                    
                    matched_gold.add(pred_key)
                    matched_pred.add(pred_key)
                    total_tp_cat += 1
                    
                    # Store V-A errors for RMSE
                    self.valence_errors.append(pred_va[0] - gold_va[0])
                    self.arousal_errors.append(pred_va[1] - gold_va[1])
            
            # Count FN (gold quads not matched)
            total_fn_cat += len(gold_set) - len(matched_gold)
            
            # Count FP (predicted quads not matched)
            total_fp_cat += len(pred_set) - len(matched_pred)
        
        # Compute metrics
        c_recall = total_ctp / (total_tp_cat + total_fn_cat) if (total_tp_cat + total_fn_cat) > 0 else 0.0
        c_precision = total_ctp / (total_tp_cat + total_fp_cat) if (total_tp_cat + total_fp_cat) > 0 else 0.0
        c_f1 = 2 * c_recall * c_precision / (c_recall + c_precision) if (c_recall + c_precision) > 0 else 0.0
        
        return {
            "continuous_f1": c_f1,
            "continuous_recall": c_recall,
            "continuous_precision": c_precision,
            "tp_categorical": total_tp_cat,
            "fn_categorical": total_fn_cat,
            "fp_categorical": total_fp_cat
        }
    
    def compute_exact_f1(self) -> Dict[str, float]:
        """
        Compute exact match F1 (ignoring V-A values)
        """
        total_tp = 0
        total_fn = 0
        total_fp = 0
        
        for pred_quads, gold_quads in zip(self.predictions, self.gold_labels):
            pred_set = {self._normalize_quad(q) for q in pred_quads}
            gold_set = {self._normalize_quad(q) for q in gold_quads}
            
            tp = len(pred_set & gold_set)
            fn = len(gold_set - pred_set)
            fp = len(pred_set - gold_set)
            
            total_tp += tp
            total_fn += fn
            total_fp += fp
        
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
        
        return {
            "exact_f1": f1,
            "exact_recall": recall,
            "exact_precision": precision
        }
    
    def compute_va_rmse(self) -> Dict[str, float]:
        """
        Compute RMSE for V-A predictions (on matched pairs)
        """
        if not self.valence_errors:
            return {
                "valence_rmse": 0.0,
                "arousal_rmse": 0.0,
                "va_rmse": 0.0
            }
        
        v_errors = np.array(self.valence_errors)
        a_errors = np.array(self.arousal_errors)
        
        v_rmse = np.sqrt(np.mean(v_errors ** 2))
        a_rmse = np.sqrt(np.mean(a_errors ** 2))
        
        # Combined RMSE as defined in SemEval
        va_rmse = np.sqrt(np.mean(v_errors ** 2 + a_errors ** 2))
        
        return {
            "valence_rmse": v_rmse,
            "arousal_rmse": a_rmse,
            "va_rmse": va_rmse,
            "valence_mae": np.mean(np.abs(v_errors)),
            "arousal_mae": np.mean(np.abs(a_errors))
        }
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all metrics"""
        metrics = {}
        metrics.update(self.compute_continuous_f1())
        metrics.update(self.compute_exact_f1())
        metrics.update(self.compute_va_rmse())
        return metrics
    
    def print_report(self):
        """Print a formatted evaluation report"""
        metrics = self.compute_all_metrics()
        
        print("\n" + "=" * 60)
        print("DimASQP Evaluation Report")
        print("=" * 60)
        
        print("\n📊 Continuous F1 (Main Metric):")
        print(f"  cF1:        {metrics['continuous_f1']:.4f}")
        print(f"  cRecall:    {metrics['continuous_recall']:.4f}")
        print(f"  cPrecision: {metrics['continuous_precision']:.4f}")
        
        print("\n📊 Exact Match F1 (Categorical Only):")
        print(f"  F1:        {metrics['exact_f1']:.4f}")
        print(f"  Recall:    {metrics['exact_recall']:.4f}")
        print(f"  Precision: {metrics['exact_precision']:.4f}")
        
        print("\n📊 V-A Regression Metrics:")
        print(f"  VA RMSE:   {metrics['va_rmse']:.4f}")
        print(f"  V RMSE:    {metrics['valence_rmse']:.4f}")
        print(f"  A RMSE:    {metrics['arousal_rmse']:.4f}")
        print(f"  V MAE:     {metrics['valence_mae']:.4f}")
        print(f"  A MAE:     {metrics['arousal_mae']:.4f}")
        
        print("\n📊 Counts:")
        print(f"  TP (categorical): {metrics['tp_categorical']}")
        print(f"  FN (categorical): {metrics['fn_categorical']}")
        print(f"  FP (categorical): {metrics['fp_categorical']}")
        
        print("=" * 60)
        
        return metrics


def evaluate_predictions(
    pred_file: str,
    gold_file: str
) -> Dict[str, float]:
    """
    Evaluate predictions from JSONL files
    
    Args:
        pred_file: Path to predictions JSONL file
        gold_file: Path to gold labels JSONL file
    """
    import json
    
    # Load predictions
    predictions = {}
    with open(pred_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            predictions[item["ID"]] = item.get("Quadruplet", [])
    
    # Load gold labels
    gold_labels = {}
    with open(gold_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            gold_labels[item["ID"]] = item.get("Quadruplet", [])
    
    # Align and evaluate
    evaluator = DimASQPEvaluator()
    
    pred_list = []
    gold_list = []
    
    for sample_id in gold_labels:
        pred_list.append(predictions.get(sample_id, []))
        gold_list.append(gold_labels[sample_id])
    
    evaluator.add_batch(pred_list, gold_list)
    
    return evaluator.compute_all_metrics()


if __name__ == "__main__":
    # Example usage
    evaluator = DimASQPEvaluator()
    
    # Example predictions
    predictions = [
        [
            {"Aspect": "thai food", "Category": "FOOD#QUALITY", "Opinion": "average to good", "VA": "6.50#6.00"},
            {"Aspect": "delivery", "Category": "SERVICE#GENERAL", "Opinion": "terrible", "VA": "3.00#7.00"}
        ]
    ]
    
    # Example gold labels
    gold_labels = [
        [
            {"Aspect": "thai food", "Category": "FOOD#QUALITY", "Opinion": "average to good", "VA": "6.75#6.38"},
            {"Aspect": "delivery", "Category": "SERVICE#GENERAL", "Opinion": "terrible", "VA": "2.88#6.62"}
        ]
    ]
    
    evaluator.add_batch(predictions, gold_labels)
    evaluator.print_report()