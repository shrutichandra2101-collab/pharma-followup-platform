"""
Data Validation & Gap Detection Engine - Main Integration Module
Orchestrates all validation components end-to-end
"""

import pandas as pd
import numpy as np
import json
import sys
sys.path.append('../..')

from data_generator import ValidationDataGenerator
from rule_validator import BatchValidator
from anomaly_detector import CompositeAnomalyDetector
from completeness_scorer import CompletenessScorer
from evaluation_metrics import generate_evaluation_report
from visualizer import ValidationVisualizer


class ValidationEngine:
    """Complete validation pipeline."""
    
    def __init__(self):
        self.data_generator = ValidationDataGenerator()
        self.batch_validator = BatchValidator()
        self.anomaly_detector = CompositeAnomalyDetector()
        self.completeness_scorer = CompletenessScorer()
        self.visualizer = ValidationVisualizer()
        
        self.validation_results = None
        self.anomaly_results = None
        self.combined_results = None
        self.metrics = None
    
    def run_full_pipeline(self, num_reports=10000, error_rate=0.35, output_csv=True):
        """
        Run complete validation pipeline.
        
        Args:
            num_reports: Number of synthetic reports to generate
            error_rate: Percentage of reports to have errors
            output_csv: Save results to CSV
            
        Returns:
            dict with results and metrics
        """
        print("\n" + "="*70)
        print("DATA VALIDATION & GAP DETECTION ENGINE")
        print("="*70 + "\n")
        
        # Step 1: Generate synthetic data
        print("STEP 1: Generating Synthetic Data")
        print("-" * 70)
        print(f"Generating {num_reports} adverse event reports...")
        # Reinitialize generator with desired parameters
        self.data_generator = ValidationDataGenerator(num_samples=num_reports, error_rate=error_rate)
        df = self.data_generator.generate_dataset()
        print(f"✓ Generated {len(df)} reports")
        print(f"  Reports with errors: {(df['has_errors'] == 1).sum()} ({(df['has_errors'] == 1).sum()/len(df)*100:.1f}%)")
        print(f"  Clean reports: {(df['has_errors'] == 0).sum()} ({(df['has_errors'] == 0).sum()/len(df)*100:.1f}%)")
        
        # Step 2: Rule-based validation
        print("\n\nSTEP 2: Rule-Based Validation")
        print("-" * 70)
        print("Validating reports against ICH E2B(R3) rules...")
        self.validation_results = self.batch_validator.validate_dataset(df)
        validation_summary = self.batch_validator.get_summary_report(self.validation_results, df)
        
        print(f"✓ Validation complete")
        print(f"  Valid reports: {validation_summary['valid_count']} ({validation_summary['valid_count']/len(self.validation_results)*100:.1f}%)")
        print(f"  Invalid reports: {validation_summary['invalid_count']} ({validation_summary['invalid_count']/len(self.validation_results)*100:.1f}%)")
        print(f"  Average quality score: {validation_summary['avg_quality_score']:.2f}/100")
        
        # Step 3: Anomaly detection
        print("\n\nSTEP 3: Anomaly Detection")
        print("-" * 70)
        print("Training Isolation Forest on clean patterns...")
        clean_df = df[df['has_errors'] == 0]
        if len(clean_df) > 0:
            self.anomaly_detector.statistical_detector.train(clean_df)
            print(f"✓ Trained on {len(clean_df)} clean reports")
        else:
            self.anomaly_detector.statistical_detector.train(df)
            print(f"✓ Trained on all {len(df)} reports (no clean subset)")
        
        print("Detecting anomalies...")
        self.anomaly_results = self.anomaly_detector.detect_anomalies(df, self.validation_results)
        
        high_risk = (self.anomaly_results['anomaly_risk'] == 'High').sum()
        medium_risk = (self.anomaly_results['anomaly_risk'] == 'Medium').sum()
        low_risk = (self.anomaly_results['anomaly_risk'] == 'Low').sum()
        
        print(f"✓ Anomaly detection complete")
        print(f"  High risk: {high_risk} ({high_risk/len(self.anomaly_results)*100:.1f}%)")
        print(f"  Medium risk: {medium_risk} ({medium_risk/len(self.anomaly_results)*100:.1f}%)")
        print(f"  Low risk: {low_risk} ({low_risk/len(self.anomaly_results)*100:.1f}%)")
        
        # Step 4: Completeness scoring
        print("\n\nSTEP 4: Completeness Scoring")
        print("-" * 70)
        print("Calculating field completeness...")
        completeness_scores = self.completeness_scorer.calculate_scores_batch(df)
        completeness_report = self.completeness_scorer.generate_completeness_report(df)
        
        print(f"✓ Completeness scoring complete")
        print(f"  Average completeness: {completeness_report['average_completeness']:.2f}%")
        print(f"  Excellent (≥80%): {completeness_report['distribution']['excellent']}")
        print(f"  Good (60-80%): {completeness_report['distribution']['good']}")
        print(f"  Fair (40-60%): {completeness_report['distribution']['fair']}")
        print(f"  Poor (20-40%): {completeness_report['distribution']['poor']}")
        print(f"  Critical (<20%): {completeness_report['distribution']['critical']}")
        
        # Step 5: Combine all results
        print("\n\nSTEP 5: Combining Results")
        print("-" * 70)
        self.combined_results = self._combine_results(df, completeness_scores)
        
        print(f"✓ Results combined")
        status_counts = self.combined_results['overall_status'].value_counts()
        for status in ['ACCEPT', 'CONDITIONAL_ACCEPT', 'REVIEW', 'REJECT']:
            count = status_counts.get(status, 0)
            pct = count / len(self.combined_results) * 100
            print(f"  {status}: {count} ({pct:.1f}%)")
        
        # Step 6: Calculate evaluation metrics
        print("\n\nSTEP 6: Calculating Metrics")
        print("-" * 70)
        self.metrics = generate_evaluation_report(
            self.validation_results,
            self.anomaly_results,
            self.combined_results,
            df
        )
        
        val_metrics = self.metrics.get('validation_metrics', {})
        print(f"✓ Validation Metrics:")
        print(f"  Precision: {val_metrics.get('precision', 0):.3f}")
        print(f"  Recall: {val_metrics.get('recall', 0):.3f}")
        print(f"  F1-Score: {val_metrics.get('f1', 0):.3f}")
        print(f"  False Positive Rate: {val_metrics.get('false_positive_rate', 0):.3f}")
        
        anom_metrics = self.metrics.get('anomaly_metrics', {})
        print(f"\n✓ Anomaly Detection Metrics:")
        print(f"  Precision: {anom_metrics.get('precision', 0):.3f}")
        print(f"  Recall: {anom_metrics.get('recall', 0):.3f}")
        print(f"  F1-Score: {anom_metrics.get('f1', 0):.3f}")
        
        # Step 7: Generate visualizations
        print("\n\nSTEP 7: Generating Visualizations")
        print("-" * 70)
        self.visualizer.generate_all_visualizations(
            self.validation_results,
            self.anomaly_results,
            self.combined_results,
            df,
            self.metrics
        )
        
        # Step 8: Save results
        print("\n\nSTEP 8: Saving Results")
        print("-" * 70)
        if output_csv:
            self._save_results(df)
        
        return {
            'combined_results': self.combined_results,
            'metrics': self.metrics,
            'completeness_report': completeness_report,
            'validation_summary': validation_summary,
        }
    
    def _combine_results(self, original_df, completeness_scores):
        """Combine all validation aspects."""
        combined = pd.DataFrame({
            'case_id': original_df.get('case_id', range(len(original_df))),
            'is_valid': self.validation_results['is_valid'].values,
            'error_count': self.validation_results['error_count'].values,
            'quality_score': self.validation_results['quality_score'].values,
            'completeness_score': completeness_scores.values,
            'anomaly_score': self.anomaly_results['anomaly_score'].values,
            'anomaly_risk': self.anomaly_results['anomaly_risk'].values,
        })
        
        # Overall decision
        combined['overall_status'] = combined.apply(
            lambda row: self._determine_status(row),
            axis=1
        )
        
        # Priority for human review
        combined['review_priority'] = combined.apply(
            lambda row: self._calculate_priority(row),
            axis=1
        )
        
        return combined
    
    def _determine_status(self, row):
        """Determine overall validation status."""
        if not row['is_valid'] or row['error_count'] > 2:
            return 'REJECT'
        elif row['quality_score'] < 40 or row['anomaly_risk'] == 'High':
            return 'REVIEW'
        elif row['quality_score'] < 70 or row['anomaly_risk'] == 'Medium':
            return 'CONDITIONAL_ACCEPT'
        else:
            return 'ACCEPT'
    
    def _calculate_priority(self, row):
        """Calculate review priority (1=highest)."""
        priority = 0
        
        if row['overall_status'] == 'REJECT':
            priority += 40
        elif row['overall_status'] == 'REVIEW':
            priority += 30
        elif row['overall_status'] == 'CONDITIONAL_ACCEPT':
            priority += 15
        
        if row['anomaly_risk'] == 'High':
            priority += 20
        elif row['anomaly_risk'] == 'Medium':
            priority += 10
        
        if row['error_count'] > 3:
            priority += 10
        
        if row['completeness_score'] < 30:
            priority += 10
        
        return max(1, priority)
    
    def _save_results(self, original_df):
        """Save all results to files."""
        import os
        
        # Create output directories
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(base_dir, 'data', 'processed')
        eval_dir = os.path.join(base_dir, 'evaluation')
        
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        
        # Combined results
        output_path = os.path.join(data_dir, 'validation_results.csv')
        self.combined_results.to_csv(output_path, index=False)
        print(f"✓ Saved combined results: {output_path}")
        
        # Detailed metrics to JSON
        metrics_path = os.path.join(eval_dir, 'validation_metrics.json')
        metrics_json = {}
        for key, value in self.metrics.items():
            if not isinstance(value, (dict, list)) or key == 'validation_metrics' or key == 'anomaly_metrics':
                # Skip non-serializable objects like arrays
                if isinstance(value, dict):
                    # Clean up non-serializable items
                    cleaned = {}
                    for k, v in value.items():
                        if isinstance(v, (int, float, str, bool, type(None))):
                            cleaned[k] = v
                        elif isinstance(v, np.ndarray):
                            cleaned[k] = v.tolist()
                    metrics_json[key] = cleaned
                else:
                    metrics_json[key] = value
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        print(f"✓ Saved metrics: {metrics_path}")
        
        # Summary report
        report_path = os.path.join(eval_dir, 'VALIDATION_ENGINE_REPORT.txt')
        with open(report_path, 'w') as f:
            f.write("DATA VALIDATION & GAP DETECTION ENGINE - REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("DATASET OVERVIEW\n")
            f.write("-"*70 + "\n")
            f.write(f"Total reports: {len(original_df)}\n")
            f.write(f"Reports with errors: {(original_df['has_errors']==1).sum()}\n")
            f.write(f"Clean reports: {(original_df['has_errors']==0).sum()}\n\n")
            
            f.write("VALIDATION RESULTS\n")
            f.write("-"*70 + "\n")
            for key, value in self.metrics.get('error_detection_analysis', {}).items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("ANOMALY DETECTION RESULTS\n")
            f.write("-"*70 + "\n")
            for key, value in self.metrics.get('anomaly_detection_analysis', {}).items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("VALIDATION METRICS\n")
            f.write("-"*70 + "\n")
            val_metrics = self.metrics.get('validation_metrics', {})
            for key in ['precision', 'recall', 'f1', 'accuracy', 'false_positive_rate']:
                f.write(f"{key}: {val_metrics.get(key, 'N/A')}\n")
            f.write("\n")
            
            f.write("OVERALL STATUS DISTRIBUTION\n")
            f.write("-"*70 + "\n")
            status_counts = self.combined_results['overall_status'].value_counts()
            for status in ['ACCEPT', 'CONDITIONAL_ACCEPT', 'REVIEW', 'REJECT']:
                count = status_counts.get(status, 0)
                pct = count / len(self.combined_results) * 100
                f.write(f"{status}: {count} ({pct:.1f}%)\n")
        
        print(f"✓ Saved report: {report_path}")


if __name__ == "__main__":
    # Run full pipeline
    engine = ValidationEngine()
    results = engine.run_full_pipeline(num_reports=10000, error_rate=0.35)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
