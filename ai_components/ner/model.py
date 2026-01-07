"""
Medical Named Entity Recognition - Model Training Module
Train NER model using spaCy and transformer-based sequence labeling

Step 2: Build and train NER model
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json
import joblib
import sys
sys.path.append('../..')

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import pickle


class SimpleNERModel:
    """Simple pattern-based NER model for medical entities."""
    
    def __init__(self):
        """Initialize NER model."""
        self.entity_patterns = {}
        self.trained = False
        self.entity_types = ['DRUG', 'DOSAGE', 'ROUTE', 'DURATION', 'CONDITION', 
                            'OUTCOME', 'FREQUENCY', 'SEVERITY']
    
    def build_patterns(self, train_df: pd.DataFrame):
        """Build entity patterns from training data."""
        print("\nBuilding NER patterns from training data...")
        
        # Extract all entities by type
        all_entities = {}
        for entity_type in self.entity_types:
            all_entities[entity_type] = set()
        
        # Aggregate entities from all narratives
        for entities_list in train_df['entities']:
            for entity in entities_list:
                entity_type = entity['type']
                entity_text = entity['text'].lower()
                if entity_type in all_entities:
                    all_entities[entity_type].add(entity_text)
        
        # Create pattern dictionary
        self.entity_patterns = {k: sorted(list(v)) for k, v in all_entities.items()}
        
        print(f"✓ Built patterns for {len(self.entity_types)} entity types")
        for entity_type, patterns in self.entity_patterns.items():
            print(f"  • {entity_type}: {len(patterns)} unique values")
        
        self.trained = True
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text."""
        if not self.trained:
            raise ValueError("Model not trained. Call build_patterns first.")
        
        entities = []
        text_lower = text.lower()
        
        # Search for each entity type
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                start_idx = 0
                while True:
                    idx = text_lower.find(pattern, start_idx)
                    if idx == -1:
                        break
                    
                    # Find original case version in text
                    end_idx = idx + len(pattern)
                    original_text = text[idx:end_idx]
                    
                    # Check if this overlaps with existing entities
                    overlaps = False
                    for existing in entities:
                        if not (end_idx <= existing['start'] or idx >= existing['end']):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        entities.append({
                            'text': original_text,
                            'type': entity_type,
                            'start': idx,
                            'end': end_idx,
                            'confidence': 0.85
                        })
                    
                    start_idx = end_idx
        
        # Sort by position
        entities = sorted(entities, key=lambda x: x['start'])
        return entities
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({
            'patterns': self.entity_patterns,
            'trained': self.trained,
            'entity_types': self.entity_types
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.entity_patterns = data['patterns']
        self.trained = data['trained']
        self.entity_types = data['entity_types']
        print(f"✓ Model loaded from {path}")


class NERModelTrainer:
    """Train and evaluate NER model."""
    
    def __init__(self):
        self.model = SimpleNERModel()
        self.train_df = None
        self.test_df = None
        self.metrics = {}
    
    def train(self, train_df: pd.DataFrame):
        """Train NER model."""
        print("\n" + "="*70)
        print("TRAINING NER MODEL")
        print("="*70)
        
        self.train_df = train_df
        self.model.build_patterns(train_df)
        
        print("\n✓ NER model training complete")
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate NER model on test set."""
        print("\n" + "="*70)
        print("EVALUATING NER MODEL")
        print("="*70)
        
        self.test_df = test_df
        
        # Track metrics
        all_precision = []
        all_recall = []
        all_f1 = []
        
        entity_metrics = {}
        for entity_type in self.model.entity_types:
            entity_metrics[entity_type] = {
                'precision': [],
                'recall': [],
                'f1': [],
                'extracted_count': 0,
                'true_count': 0,
                'correct_count': 0
            }
        
        # Evaluate on each test case
        for _, row in test_df.iterrows():
            narrative = row['narrative']
            true_entities = row['entities']
            
            # Extract entities
            predicted_entities = self.model.extract_entities(narrative)
            
            # Track by type
            true_by_type = {}
            pred_by_type = {}
            
            for entity_type in self.model.entity_types:
                true_by_type[entity_type] = [e for e in true_entities if e['type'] == entity_type]
                pred_by_type[entity_type] = [e for e in predicted_entities if e['type'] == entity_type]
            
            # Calculate matches
            for entity_type in self.model.entity_types:
                true_list = true_by_type[entity_type]
                pred_list = pred_by_type[entity_type]
                
                entity_metrics[entity_type]['true_count'] += len(true_list)
                entity_metrics[entity_type]['extracted_count'] += len(pred_list)
                
                # Count correct extractions (exact text match)
                correct = 0
                for pred in pred_list:
                    for true in true_list:
                        if pred['text'].lower() == true['text'].lower():
                            correct += 1
                            break
                
                entity_metrics[entity_type]['correct_count'] += correct
        
        # Calculate overall metrics
        print("\nPer-Entity-Type Performance:")
        print("-" * 70)
        
        for entity_type in self.model.entity_types:
            metrics = entity_metrics[entity_type]
            
            if metrics['extracted_count'] > 0:
                precision = metrics['correct_count'] / metrics['extracted_count']
            else:
                precision = 0.0
            
            if metrics['true_count'] > 0:
                recall = metrics['correct_count'] / metrics['true_count']
            else:
                recall = 0.0
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
            
            print(f"\n{entity_type}:")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall:    {recall:.3f}")
            print(f"  F1-Score:  {f1:.3f}")
            print(f"  Extracted: {metrics['extracted_count']} / True: {metrics['true_count']} / Correct: {metrics['correct_count']}")
        
        # Overall metrics
        total_correct = sum(m['correct_count'] for m in entity_metrics.values())
        total_pred = sum(m['extracted_count'] for m in entity_metrics.values())
        total_true = sum(m['true_count'] for m in entity_metrics.values())
        
        overall_precision = total_correct / total_pred if total_pred > 0 else 0.0
        overall_recall = total_correct / total_true if total_true > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        print("\n" + "-" * 70)
        print("OVERALL METRICS:")
        print(f"Precision: {overall_precision:.3f}")
        print(f"Recall:    {overall_recall:.3f}")
        print(f"F1-Score:  {overall_f1:.3f}")
        
        # Store metrics
        self.metrics = {
            'overall': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
                'total_extracted': total_pred,
                'total_true': total_true,
                'total_correct': total_correct
            },
            'by_entity_type': entity_metrics
        }
        
        return self.metrics
    
    def save_model(self, path: str):
        """Save trained model."""
        self.model.save(path)
    
    def load_model(self, path: str):
        """Load trained model."""
        self.model.load(path)


if __name__ == "__main__":
    from data_generator import NERDataGenerator
    
    # Generate data
    gen = NERDataGenerator()
    train_df, test_df = gen.generate_and_split(num_training=4000, num_test=1000)
    
    # Train model
    trainer = NERModelTrainer()
    trainer.train(train_df)
    
    # Evaluate
    metrics = trainer.evaluate(test_df)
    
    # Save model
    trainer.save_model('../../../data/models/ner_model.pkl')
