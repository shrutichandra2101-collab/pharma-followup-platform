"""
Medical Named Entity Recognition - Evaluation Metrics Module
Calculate comprehensive NER performance metrics

Step 3: Evaluate NER model performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import json


class NERMetrics:
    """Calculate NER-specific metrics."""
    
    @staticmethod
    def calculate_entity_overlap(pred_entity: Dict, true_entity: Dict) -> float:
        """Calculate overlap between two entities (IoU - Intersection over Union)."""
        pred_start, pred_end = pred_entity['start'], pred_entity['end']
        true_start, true_end = true_entity['start'], true_entity['end']
        
        # Calculate intersection
        inter_start = max(pred_start, true_start)
        inter_end = min(pred_end, true_end)
        
        if inter_end <= inter_start:
            return 0.0  # No overlap
        
        intersection = inter_end - inter_start
        union = max(pred_end, true_end) - min(pred_start, true_start)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def match_entities(predicted: List[Dict], true: List[Dict], threshold: float = 0.8) -> Tuple[int, List[Dict]]:
        """Match predicted entities to true entities."""
        matches = 0
        match_details = []
        used_true = set()
        
        for pred in predicted:
            best_match = None
            best_overlap = 0.0
            best_idx = -1
            
            for idx, true_ent in enumerate(true):
                if idx in used_true:
                    continue
                
                if pred['type'] != true_ent['type']:
                    continue
                
                overlap = NERMetrics.calculate_entity_overlap(pred, true_ent)
                if overlap > best_overlap and overlap >= threshold:
                    best_overlap = overlap
                    best_match = true_ent
                    best_idx = idx
            
            if best_match:
                matches += 1
                used_true.add(best_idx)
                match_details.append({
                    'predicted': pred,
                    'true': best_match,
                    'overlap': best_overlap,
                    'match': True
                })
            else:
                match_details.append({
                    'predicted': pred,
                    'true': None,
                    'match': False
                })
        
        return matches, match_details
    
    @staticmethod
    def calculate_per_entity_metrics(predictions: List[Dict], true_labels: List[Dict], 
                                     entity_types: List[str]) -> Dict[str, Any]:
        """Calculate metrics per entity type."""
        metrics = {}
        
        for entity_type in entity_types:
            pred_type = [e for e in predictions if e['type'] == entity_type]
            true_type = [e for e in true_labels if e['type'] == entity_type]
            
            tp = 0
            fp = len(pred_type)
            fn = len(true_type)
            
            # Find true positives
            used_true = set()
            for pred in pred_type:
                for idx, true_ent in enumerate(true_type):
                    if idx not in used_true:
                        if pred['text'].lower() == true_ent['text'].lower():
                            tp += 1
                            fp -= 1
                            fn -= 1
                            used_true.add(idx)
                            break
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[entity_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'support': len(true_type)
            }
        
        return metrics
    
    @staticmethod
    def calculate_confusion_matrix(predictions: List[Dict], true_labels: List[Dict],
                                   entity_types: List[str]) -> np.ndarray:
        """Create confusion matrix for entity types."""
        matrix = np.zeros((len(entity_types) + 1, len(entity_types) + 1))  # +1 for 'O' (no entity)
        
        entity_to_idx = {e: i for i, e in enumerate(entity_types)}
        entity_to_idx['O'] = len(entity_types)  # Outside any entity
        
        # Group by position in text
        pred_dict = {(e['start'], e['end']): e for e in predictions}
        true_dict = {(e['start'], e['end']): e for e in true_labels}
        
        all_positions = set(pred_dict.keys()) | set(true_dict.keys())
        
        for pos in all_positions:
            pred_type = pred_dict.get(pos, {}).get('type', 'O')
            true_type = true_dict.get(pos, {}).get('type', 'O')
            
            pred_idx = entity_to_idx.get(pred_type, len(entity_types))
            true_idx = entity_to_idx.get(true_type, len(entity_types))
            
            matrix[true_idx, pred_idx] += 1
        
        return matrix


class NERPerformanceAnalysis:
    """Analyze NER performance across different aspects."""
    
    @staticmethod
    def analyze_by_complexity(test_df: pd.DataFrame, predictions: List[Dict],
                             entity_types: List[str]) -> Dict[str, Any]:
        """Analyze performance by narrative complexity."""
        analysis = {}
        
        for complexity in ['simple', 'moderate', 'complex']:
            mask = test_df['complexity'] == complexity
            subset_df = test_df[mask]
            
            if len(subset_df) == 0:
                continue
            
            # Calculate metrics for this subset
            total_true = sum(subset_df['entity_count'])
            total_pred = sum(len(self.extract_entities(text)) for text in subset_df['narrative'])
            
            analysis[complexity] = {
                'count': len(subset_df),
                'avg_entities': total_true / len(subset_df) if len(subset_df) > 0 else 0,
                'avg_narrative_length': subset_df['narrative_length'].mean(),
                'total_true_entities': total_true,
                'total_predicted': total_pred
            }
        
        return analysis
    
    @staticmethod
    def analyze_extraction_quality(test_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze extraction quality metrics."""
        entity_coverage = []
        extraction_completeness = []
        
        for _, row in test_df.iterrows():
            entities = row['entities']
            entity_count = len(entities)
            
            if entity_count > 0:
                coverage = entity_count / max(entity_count, 1)
                entity_coverage.append(coverage)
                extraction_completeness.append(coverage)
        
        return {
            'avg_coverage': np.mean(entity_coverage) if entity_coverage else 0.0,
            'coverage_std': np.std(entity_coverage) if entity_coverage else 0.0,
            'median_coverage': np.median(entity_coverage) if entity_coverage else 0.0,
            'min_coverage': np.min(entity_coverage) if entity_coverage else 0.0,
            'max_coverage': np.max(entity_coverage) if entity_coverage else 1.0
        }
    
    @staticmethod
    def analyze_entity_distribution(test_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze entity type distribution."""
        distribution = {
            'DRUG': 0, 'DOSAGE': 0, 'ROUTE': 0, 'DURATION': 0,
            'CONDITION': 0, 'OUTCOME': 0, 'FREQUENCY': 0, 'SEVERITY': 0
        }
        
        for entities_list in test_df['entities']:
            for entity in entities_list:
                entity_type = entity['type']
                if entity_type in distribution:
                    distribution[entity_type] += 1
        
        total = sum(distribution.values())
        percentages = {k: (v / total * 100) if total > 0 else 0 for k, v in distribution.items()}
        
        return {
            'counts': distribution,
            'percentages': percentages,
            'total': total,
            'avg_per_narrative': total / len(test_df) if len(test_df) > 0 else 0
        }


def generate_evaluation_report(metrics: Dict[str, Any], entity_types: List[str]) -> str:
    """Generate comprehensive evaluation report."""
    report = []
    
    report.append("\n" + "="*70)
    report.append("NER MODEL EVALUATION REPORT")
    report.append("="*70 + "\n")
    
    # Overall metrics
    if 'overall' in metrics:
        overall = metrics['overall']
        report.append("OVERALL PERFORMANCE:")
        report.append("-" * 70)
        report.append(f"Precision: {overall.get('precision', 0):.4f}")
        report.append(f"Recall:    {overall.get('recall', 0):.4f}")
        report.append(f"F1-Score:  {overall.get('f1', 0):.4f}")
        report.append(f"Total Correct Extractions: {overall.get('total_correct', 0)}")
        report.append(f"Total Extracted: {overall.get('total_extracted', 0)}")
        report.append(f"Total True: {overall.get('total_true', 0)}\n")
    
    # Per-entity metrics
    if 'by_entity_type' in metrics:
        report.append("PER-ENTITY-TYPE METRICS:")
        report.append("-" * 70)
        
        for entity_type in entity_types:
            if entity_type in metrics['by_entity_type']:
                m = metrics['by_entity_type'][entity_type]
                report.append(f"\n{entity_type}:")
                report.append(f"  Precision: {m.get('precision', 0):.4f}")
                report.append(f"  Recall:    {m.get('recall', 0):.4f}")
                report.append(f"  F1-Score:  {m.get('f1', 0):.4f}")
                report.append(f"  Support:   {m.get('support', 0)}")
    
    return "\n".join(report)


if __name__ == "__main__":
    print("NER Metrics module ready")
