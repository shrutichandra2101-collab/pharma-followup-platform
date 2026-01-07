"""
Evaluation metrics for validation and anomaly detection.
Calculates precision, recall, F1-score, AUC-ROC, and false positive rate.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc,
    confusion_matrix, classification_report
)


class ValidationMetrics:
    """Calculate metrics for validation accuracy."""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Calculate precision, recall, F1, and accuracy.
        
        Args:
            y_true: true labels (0=valid, 1=invalid)
            y_pred: predicted labels (0=valid, 1=invalid)
            
        Returns:
            dict with all metrics
        """
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': np.mean(y_true == y_pred),
            'true_negatives': ((y_true == 0) & (y_pred == 0)).sum(),
            'false_positives': ((y_true == 0) & (y_pred == 1)).sum(),
            'false_negatives': ((y_true == 1) & (y_pred == 0)).sum(),
            'true_positives': ((y_true == 1) & (y_pred == 1)).sum(),
        }
        
        # False Positive Rate
        if metrics['true_negatives'] + metrics['false_positives'] > 0:
            metrics['false_positive_rate'] = metrics['false_positives'] / (
                metrics['true_negatives'] + metrics['false_positives']
            )
        else:
            metrics['false_positive_rate'] = 0
        
        # True Positive Rate
        if metrics['true_positives'] + metrics['false_negatives'] > 0:
            metrics['true_positive_rate'] = metrics['true_positives'] / (
                metrics['true_positives'] + metrics['false_negatives']
            )
        else:
            metrics['true_positive_rate'] = 0
        
        return metrics
    
    @staticmethod
    def calculate_metrics_from_validation_results(validation_results, original_df):
        """
        Calculate metrics using validation results against ground truth.
        Ground truth: has_errors column in original_df (0=valid, 1=invalid)
        
        Args:
            validation_results: DataFrame with is_valid column
            original_df: DataFrame with has_errors ground truth
            
        Returns:
            dict with metrics and confusion matrix
        """
        # Convert to binary: is_valid True/False -> 0/1 (valid/invalid)
        y_true = original_df['has_errors'].values
        y_pred = (~validation_results['is_valid']).astype(int).values
        
        metrics = ValidationMetrics.calculate_metrics(y_true, y_pred)
        
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        return metrics


class AnomalyMetrics:
    """Calculate metrics for anomaly detection."""
    
    @staticmethod
    def calculate_metrics_from_scores(y_true, anomaly_scores):
        """
        Calculate AUC-ROC and other metrics using anomaly scores.
        
        Args:
            y_true: true labels (0=normal, 1=anomalous)
            anomaly_scores: continuous anomaly scores (0-1, higher=more anomalous)
            
        Returns:
            dict with metrics including AUC-ROC and optimal threshold
        """
        # Calculate AUC-ROC
        try:
            auc_roc = roc_auc_score(y_true, anomaly_scores)
        except:
            auc_roc = None
        
        # Find optimal threshold (Youden's index)
        fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores)
        youden = tpr - fpr
        optimal_idx = np.argmax(youden)
        optimal_threshold = thresholds[optimal_idx]
        
        # Predictions with optimal threshold
        y_pred = (anomaly_scores >= optimal_threshold).astype(int)
        
        metrics = {
            'auc_roc': auc_roc,
            'optimal_threshold': optimal_threshold,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
        }
        
        # Add binary metrics with optimal threshold
        if len(np.unique(y_true)) > 1:
            metrics.update(ValidationMetrics.calculate_metrics(y_true, y_pred))
        
        return metrics
    
    @staticmethod
    def calculate_metrics_from_risk_levels(y_true, risk_levels):
        """
        Calculate metrics using risk level classifications.
        Low -> 0 (normal), Medium/High -> 1 (anomalous)
        
        Args:
            y_true: true labels (0=normal, 1=anomalous)
            risk_levels: risk level predictions ['Low', 'Medium', 'High']
            
        Returns:
            dict with metrics
        """
        # Convert risk levels to binary
        y_pred = (risk_levels.isin(['Medium', 'High'])).astype(int).values
        
        metrics = ValidationMetrics.calculate_metrics(y_true, y_pred)
        
        return metrics


class PerformanceAnalysis:
    """Comprehensive performance analysis across multiple perspectives."""
    
    @staticmethod
    def analyze_error_detection(validation_results, original_df):
        """
        Analyze how well validation detects actual errors.
        
        Args:
            validation_results: DataFrame with validation results
            original_df: DataFrame with has_errors ground truth
            
        Returns:
            dict with detailed analysis
        """
        # Split by error presence
        with_errors = original_df['has_errors'] == 1
        without_errors = original_df['has_errors'] == 0
        
        analysis = {
            'error_detection_rate': {
                'detected': (
                    validation_results.loc[with_errors, 'error_count'] > 0
                ).sum() / with_errors.sum() if with_errors.sum() > 0 else 0,
                'missed': (
                    validation_results.loc[with_errors, 'error_count'] == 0
                ).sum() / with_errors.sum() if with_errors.sum() > 0 else 0,
            },
            'false_positive_rate': {
                'rate': (
                    validation_results.loc[without_errors, 'error_count'] > 0
                ).sum() / without_errors.sum() if without_errors.sum() > 0 else 0,
                'count': (
                    validation_results.loc[without_errors, 'error_count'] > 0
                ).sum()
            },
            'error_count_distribution': {
                'mean_errors_in_bad_reports': validation_results.loc[with_errors, 'error_count'].mean() if with_errors.sum() > 0 else 0,
                'mean_errors_in_good_reports': validation_results.loc[without_errors, 'error_count'].mean() if without_errors.sum() > 0 else 0,
            }
        }
        
        return analysis
    
    @staticmethod
    def analyze_anomaly_detection(combined_results, original_df):
        """
        Analyze anomaly detection performance.
        
        Args:
            combined_results: DataFrame with anomaly detection results
            original_df: DataFrame with has_errors ground truth
            
        Returns:
            dict with detailed analysis
        """
        has_errors = original_df['has_errors'] == 1
        
        # Anomaly detection vs ground truth
        detected_as_anomaly = combined_results['anomaly_risk'] == 'High'
        
        analysis = {
            'anomalies_in_error_reports': {
                'detected': (detected_as_anomaly & has_errors).sum(),
                'total': has_errors.sum(),
                'rate': (detected_as_anomaly & has_errors).sum() / has_errors.sum() if has_errors.sum() > 0 else 0
            },
            'false_anomalies_in_clean_reports': {
                'detected': (detected_as_anomaly & ~has_errors).sum(),
                'total': (~has_errors).sum(),
                'rate': (detected_as_anomaly & ~has_errors).sum() / (~has_errors).sum() if (~has_errors).sum() > 0 else 0
            },
            'anomaly_score_distribution': {
                'mean_score_with_errors': combined_results.loc[has_errors, 'anomaly_score'].mean() if has_errors.sum() > 0 else 0,
                'mean_score_without_errors': combined_results.loc[~has_errors, 'anomaly_score'].mean() if (~has_errors).sum() > 0 else 0,
            }
        }
        
        return analysis


def generate_evaluation_report(validation_results, anomaly_results, combined_results, original_df):
    """
    Generate comprehensive evaluation report.
    
    Args:
        validation_results: Rule-based validation results
        anomaly_results: Anomaly detection results
        combined_results: Combined assessment results
        original_df: Original data with ground truth
        
    Returns:
        dict with complete evaluation
    """
    report = {
        'validation_metrics': ValidationMetrics.calculate_metrics_from_validation_results(
            validation_results, original_df
        ),
        'anomaly_metrics': AnomalyMetrics.calculate_metrics_from_risk_levels(
            original_df['has_errors'].values,
            combined_results['anomaly_risk']
        ),
        'error_detection_analysis': PerformanceAnalysis.analyze_error_detection(
            validation_results, original_df
        ),
        'anomaly_detection_analysis': PerformanceAnalysis.analyze_anomaly_detection(
            combined_results, original_df
        ),
    }
    
    return report


if __name__ == "__main__":
    print("Evaluation Metrics Module")
    print("Calculates performance metrics for validation and anomaly detection")
