"""
Medical Named Entity Recognition - Main Orchestrator Module
Orchestrates complete NER pipeline end-to-end

Step 5: Run complete NER pipeline
"""

import pandas as pd
import json
import sys
import os
sys.path.append('../..')
sys.path.insert(0, '.')

from data_generator import NERDataGenerator
from model import NERModelTrainer
from evaluation_metrics import NERMetrics, generate_evaluation_report
from visualizer import NERVisualizer


class NERPipeline:
    """Complete NER pipeline orchestrator."""
    
    def __init__(self):
        self.data_generator = NERDataGenerator()
        self.model_trainer = NERModelTrainer()
        self.visualizer = NERVisualizer()
        
        self.train_df = None
        self.test_df = None
        self.metrics = None
    
    def run_full_pipeline(self, num_training: int = 4000, num_test: int = 1000, output_json: bool = True):
        """
        Run complete NER pipeline.
        
        Args:
            num_training: Number of training cases
            num_test: Number of test cases
            output_json: Save results to JSON
        """
        print("\n" + "="*70)
        print("MEDICAL NAMED ENTITY RECOGNITION PIPELINE")
        print("="*70 + "\n")
        
        # Step 1: Generate training data
        print("STEP 1: Generating Training Data")
        print("-" * 70)
        self.train_df, self.test_df = self.data_generator.generate_and_split(num_training, num_test)
        
        # Step 2: Train NER model
        print("\n\nSTEP 2: Training NER Model")
        print("-" * 70)
        self.model_trainer.train(self.train_df)
        
        # Step 3: Evaluate model
        print("\n\nSTEP 3: Evaluating NER Model")
        print("-" * 70)
        self.metrics = self.model_trainer.evaluate(self.test_df)
        
        # Step 4: Generate visualizations
        print("\n\nSTEP 4: Generating Visualizations")
        print("-" * 70)
        self.visualizer.generate_all_visualizations(self.test_df, self.metrics)
        
        # Step 5: Save results
        print("\n\nSTEP 5: Saving Results")
        print("-" * 70)
        self._save_results(output_json)
        
        # Step 6: Display summary
        print("\n\nSTEP 6: Pipeline Summary")
        print("-" * 70)
        self._print_summary()
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")
        
        return {
            'train_data': self.train_df,
            'test_data': self.test_df,
            'metrics': self.metrics,
            'model': self.model_trainer.model
        }
    
    def _save_results(self, output_json: bool = True):
        """Save results to files."""
        # Create output directories
        os.makedirs('../../../data/models', exist_ok=True)
        os.makedirs('../../../data/processed', exist_ok=True)
        os.makedirs('../../../evaluation', exist_ok=True)
        
        # Save training data
        self.train_df.to_csv('../../../data/processed/ner_train.csv', index=False)
        print("✓ Training data saved: data/processed/ner_train.csv")
        
        # Save test data
        self.test_df.to_csv('../../../data/processed/ner_test.csv', index=False)
        print("✓ Test data saved: data/processed/ner_test.csv")
        
        # Save model
        self.model_trainer.save_model('../../../data/models/ner_model.pkl')
        
        # Save metrics to JSON
        if output_json and self.metrics:
            metrics_dict = {
                'overall': self.metrics.get('overall', {}),
                'by_entity_type': {}
            }
            
            for entity_type, m in self.metrics.get('by_entity_type', {}).items():
                metrics_dict['by_entity_type'][entity_type] = {
                    'precision': m.get('precision', 0.0),
                    'recall': m.get('recall', 0.0),
                    'f1': m.get('f1', 0.0),
                    'support': m.get('support', 0)
                }
            
            with open('../../../evaluation/ner_metrics.json', 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            
            print("✓ Metrics saved: evaluation/ner_metrics.json")
        
        # Save evaluation report
        report = generate_evaluation_report(self.metrics, self.model_trainer.model.entity_types)
        with open('../../../evaluation/NER_ENGINE_REPORT.txt', 'w') as f:
            f.write(report)
        
        print("✓ Report saved: evaluation/NER_ENGINE_REPORT.txt")
    
    def _print_summary(self):
        """Print pipeline execution summary."""
        print(f"\nData Generation:")
        print(f"  Training samples: {len(self.train_df):,}")
        print(f"  Test samples: {len(self.test_df):,}")
        print(f"  Entity types: {len(self.model_trainer.model.entity_types)}")
        
        # Complexity distribution
        print(f"\nComplexity Distribution:")
        for complexity in ['simple', 'moderate', 'complex']:
            count = (self.test_df['complexity'] == complexity).sum()
            pct = count / len(self.test_df) * 100
            print(f"  • {complexity}: {count} ({pct:.1f}%)")
        
        # Average metrics
        if self.metrics and 'overall' in self.metrics:
            overall = self.metrics['overall']
            print(f"\nOverall Performance:")
            print(f"  Precision: {overall['precision']:.4f}")
            print(f"  Recall:    {overall['recall']:.4f}")
            print(f"  F1-Score:  {overall['f1']:.4f}")
        
        # Per-entity metrics
        if self.metrics and 'by_entity_type' in self.metrics:
            print(f"\nPer-Entity-Type F1 Scores:")
            for entity_type, m in self.metrics['by_entity_type'].items():
                print(f"  • {entity_type}: {m['f1']:.4f}")


if __name__ == "__main__":
    pipeline = NERPipeline()
    results = pipeline.run_full_pipeline(num_training=4000, num_test=1000)
