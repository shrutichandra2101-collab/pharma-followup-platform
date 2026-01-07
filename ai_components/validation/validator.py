"""
Integrated validation system combining rule-based and statistical approaches.
Main entry point for data validation and quality assessment.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('../..')
from rule_validator import RuleBasedValidator, BatchValidator
from anomaly_detector import CompositeAnomalyDetector


class ComprehensiveValidator:
    """Unified validation system for adverse event reports."""
    
    def __init__(self):
        self.batch_validator = BatchValidator()
        self.anomaly_detector = CompositeAnomalyDetector()
        
    def validate_dataset(self, df):
        """
        Run complete validation on dataset:
        1. Rule-based validation (mandatory fields, ranges, logic)
        2. Anomaly detection (unusual patterns)
        3. Quality scoring
        
        Args:
            df: DataFrame with adverse event reports
            
        Returns:
            dict with validation results and summary
        """
        print("\n" + "="*70)
        print("RUNNING COMPREHENSIVE VALIDATION")
        print("="*70 + "\n")
        
        # Step 1: Rule-based validation
        print("STEP 1: Rule-Based Validation")
        print("-" * 70)
        validation_results = self.batch_validator.validate_dataset(df)
        validation_summary = self.batch_validator.get_summary_report(validation_results, df)
        
        # Step 2: Anomaly detection
        print("\n\nSTEP 2: Anomaly Detection")
        print("-" * 70)
        print("Training Isolation Forest on normal patterns...")
        
        # Train on clean reports only (if available)
        if 'has_errors' in df.columns:
            training_df = df[df['has_errors'] == 0]
            if len(training_df) > 0:
                self.anomaly_detector.statistical_detector.train(training_df)
                print(f"✓ Trained on {len(training_df)} clean reports")
            else:
                self.anomaly_detector.statistical_detector.train(df)
                print(f"✓ Trained on all {len(df)} reports (no clean subset available)")
        else:
            self.anomaly_detector.statistical_detector.train(df)
            print(f"✓ Trained on all {len(df)} reports")
        
        # Detect anomalies
        print("Detecting anomalies...")
        anomaly_results = self.anomaly_detector.detect_anomalies(df, validation_results)
        
        print(f"Anomalies detected:")
        print(f"  High risk: {(anomaly_results['anomaly_risk'] == 'High').sum()}")
        print(f"  Medium risk: {(anomaly_results['anomaly_risk'] == 'Medium').sum()}")
        print(f"  Low risk: {(anomaly_results['anomaly_risk'] == 'Low').sum()}")
        
        # Step 3: Combine results
        print("\n\nSTEP 3: Combining Results")
        print("-" * 70)
        combined_results = self._combine_results(validation_results, anomaly_results, df)
        
        # Step 4: Summary
        summary = self._generate_summary(combined_results, validation_summary)
        
        return {
            'combined_results': combined_results,
            'validation_results': validation_results,
            'anomaly_results': anomaly_results,
            'summary': summary
        }
    
    def _combine_results(self, validation_results, anomaly_results, original_df):
        """Combine validation and anomaly detection results."""
        combined = pd.DataFrame({
            'case_id': anomaly_results['case_id'],
            'is_valid': validation_results['is_valid'].values,
            'error_count': validation_results['error_count'].values,
            'warning_count': validation_results['warning_count'].values,
            'quality_score': validation_results['quality_score'].values,
            'anomaly_score': anomaly_results['anomaly_score'].values,
            'anomaly_risk': anomaly_results['anomaly_risk'].values,
            'has_errors': original_df['has_errors'].values if 'has_errors' in original_df.columns else None,
        })
        
        # Overall assessment
        combined['overall_status'] = combined.apply(
            lambda row: self._determine_overall_status(row),
            axis=1
        )
        
        return combined
    
    def _determine_overall_status(self, row):
        """Determine overall validation status."""
        if not row['is_valid'] or row['error_count'] > 2:
            return 'REJECT'
        elif row['quality_score'] < 40 or row['anomaly_risk'] == 'High':
            return 'REVIEW'
        elif row['quality_score'] < 70 or row['anomaly_risk'] == 'Medium':
            return 'CONDITIONAL_ACCEPT'
        else:
            return 'ACCEPT'
    
    def _generate_summary(self, combined_results, validation_summary):
        """Generate comprehensive summary statistics."""
        summary = validation_summary.copy()
        
        # Add anomaly statistics
        summary['anomalies_detected'] = (combined_results['anomaly_risk'] == 'High').sum()
        summary['anomalies_pct'] = summary['anomalies_detected'] / len(combined_results) * 100
        
        # Add overall status distribution
        summary['status_distribution'] = {
            'ACCEPT': (combined_results['overall_status'] == 'ACCEPT').sum(),
            'CONDITIONAL_ACCEPT': (combined_results['overall_status'] == 'CONDITIONAL_ACCEPT').sum(),
            'REVIEW': (combined_results['overall_status'] == 'REVIEW').sum(),
            'REJECT': (combined_results['overall_status'] == 'REJECT').sum(),
        }
        
        # Add recommendations
        summary['recommendations'] = self._get_recommendations(combined_results)
        
        return summary
    
    def _get_recommendations(self, combined_results):
        """Generate recommendations based on validation results."""
        recommendations = []
        
        reject_count = (combined_results['overall_status'] == 'REJECT').sum()
        review_count = (combined_results['overall_status'] == 'REVIEW').sum()
        anomaly_count = (combined_results['anomaly_risk'] == 'High').sum()
        
        if reject_count / len(combined_results) > 0.3:
            recommendations.append("High rejection rate - review data collection process")
        
        if anomaly_count / len(combined_results) > 0.15:
            recommendations.append("Significant anomaly detection - investigate unusual patterns")
        
        avg_quality = combined_results['quality_score'].mean()
        if avg_quality < 50:
            recommendations.append("Low average quality score - focus on data completeness")
        
        if combined_results['error_count'].max() > 5:
            recommendations.append("Some reports have many errors - implement stricter input validation")
        
        if len(recommendations) == 0:
            recommendations.append("Dataset quality is good - continue current processes")
        
        return recommendations


def validate_file(csv_path):
    """Convenience function to validate a CSV file of reports."""
    df = pd.read_csv(csv_path)
    validator = ComprehensiveValidator()
    results = validator.validate_dataset(df)
    return results


if __name__ == "__main__":
    print("Comprehensive Validation Module")
    print("Usage: validator = ComprehensiveValidator()")
    print("       results = validator.validate_dataset(df)")
