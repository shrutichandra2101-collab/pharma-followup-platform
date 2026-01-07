"""
Smart Follow-Up Questionnaire Generator - Evaluation Metrics Module
Calculate questionnaire effectiveness and performance metrics

Step 6: Evaluate questionnaire quality and coverage
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score


class QuestionnaireMetrics:
    """Calculate questionnaire effectiveness metrics."""
    
    @staticmethod
    def calculate_effectiveness(actual_effectiveness: float, predicted_effectiveness: float) -> Dict[str, float]:
        """Calculate effectiveness metrics."""
        error = abs(actual_effectiveness - predicted_effectiveness)
        mae = error
        rmse = error ** 2
        
        return {
            'mean_absolute_error': mae,
            'root_mean_squared_error': np.sqrt(rmse),
            'accuracy_within_10': 1 if error < 10 else 0,
        }
    
    @staticmethod
    def calculate_coverage_metrics(test_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate field coverage metrics."""
        # Average % of missing fields actually addressed
        coverage_pct = (test_df['critical_fields_obtained'] / test_df['num_missing_fields']).mean()
        
        # % of cases where all critical fields addressed
        full_coverage = (test_df['critical_fields_obtained'] == test_df['num_missing_fields']).mean()
        
        return {
            'average_field_coverage': coverage_pct,
            'full_coverage_rate': full_coverage,
        }
    
    @staticmethod
    def calculate_roi_metrics(test_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate questionnaire ROI (information value vs time cost)."""
        # Information value: effectiveness × coverage
        test_df['info_value'] = test_df['questionnaire_effectiveness'] * (
            test_df['critical_fields_obtained'] / test_df['num_missing_fields']
        )
        
        # Time cost normalized (minutes / 15 minute reference)
        test_df['time_cost'] = test_df['actual_completion_time'] / 900
        
        # ROI = info_value / time_cost
        test_df['roi'] = test_df['info_value'] / (test_df['time_cost'] + 0.1)
        
        return {
            'average_roi': test_df['roi'].mean(),
            'median_roi': test_df['roi'].median(),
            'roi_std': test_df['roi'].std(),
            'high_roi_percentage': (test_df['roi'] > 70).mean(),  # ROI > 70
        }
    
    @staticmethod
    def calculate_response_quality_metrics(test_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate response quality metrics."""
        return {
            'average_completion_rate': test_df['response_completion_rate'].mean(),
            'average_response_quality': test_df['avg_response_quality'].mean(),
            'average_satisfaction': test_df['user_satisfaction'].mean(),
            'high_quality_percentage': (test_df['avg_response_quality'] > 3.5).mean(),
        }
    
    @staticmethod
    def calculate_selection_precision(test_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate precision of question selection."""
        # Did selected questions actually lead to good effectiveness?
        
        # Cases with many questions but low effectiveness = poor precision
        selection_accuracy = test_df['questionnaire_effectiveness'].mean() / 100
        
        # Precision: % of selected questions that were useful
        useful_rate = test_df['critical_fields_obtained'].sum() / test_df['num_selected_questions'].sum()
        
        return {
            'selection_accuracy': selection_accuracy,
            'useful_question_rate': useful_rate,
        }


class PerformanceAnalysis:
    """Analyze overall performance by case profile."""
    
    @staticmethod
    def analyze_by_profile(test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze performance by case profile."""
        results = {}
        
        for profile in test_df['profile'].unique():
            profile_df = test_df[test_df['profile'] == profile]
            
            results[profile] = {
                'count': len(profile_df),
                'avg_effectiveness': profile_df['questionnaire_effectiveness'].mean(),
                'avg_completion_rate': profile_df['response_completion_rate'].mean(),
                'avg_satisfaction': profile_df['user_satisfaction'].mean(),
                'avg_time_minutes': (profile_df['actual_completion_time'].mean() / 60),
                'high_quality_rate': (profile_df['avg_response_quality'] > 3.5).mean(),
            }
        
        return results
    
    @staticmethod
    def analyze_by_status(test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze performance by validation status."""
        results = {}
        
        for status in test_df['validation_status'].unique():
            status_df = test_df[test_df['validation_status'] == status]
            
            results[status] = {
                'count': len(status_df),
                'avg_effectiveness': status_df['questionnaire_effectiveness'].mean(),
                'avg_time_minutes': (status_df['actual_completion_time'].mean() / 60),
                'avg_satisfaction': status_df['user_satisfaction'].mean(),
            }
        
        return results
    
    @staticmethod
    def analyze_by_difficulty(test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze time by number of missing fields."""
        # Group by num_missing_fields quartiles
        quartiles = pd.qcut(test_df['num_missing_fields'], q=4, duplicates='drop')
        
        results = {}
        for label in quartiles.unique():
            subset = test_df[quartiles == label]
            
            results[f"{label.left:.0f}-{label.right:.0f}"] = {
                'count': len(subset),
                'avg_effectiveness': subset['questionnaire_effectiveness'].mean(),
                'avg_time_minutes': (subset['actual_completion_time'].mean() / 60),
            }
        
        return results


def generate_evaluation_report(test_df: pd.DataFrame, predictions_df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report.
    
    Args:
        test_df: Test dataset
        predictions_df: Optional predictions for comparison
        
    Returns:
        Dict with all evaluation metrics
    """
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'test_set_size': len(test_df),
        'num_profiles': len(test_df['profile'].unique()),
        'num_statuses': len(test_df['validation_status'].unique()),
    }
    
    # Overall metrics
    report['overall_metrics'] = {
        'effectiveness': QuestionnaireMetrics.calculate_effectiveness(
            test_df['questionnaire_effectiveness'].mean(),
            test_df['questionnaire_effectiveness'].mean()
        ),
        'coverage': QuestionnaireMetrics.calculate_coverage_metrics(test_df),
        'roi': QuestionnaireMetrics.calculate_roi_metrics(test_df),
        'response_quality': QuestionnaireMetrics.calculate_response_quality_metrics(test_df),
        'selection_precision': QuestionnaireMetrics.calculate_selection_precision(test_df),
    }
    
    # Performance by segment
    report['analysis_by_profile'] = PerformanceAnalysis.analyze_by_profile(test_df)
    report['analysis_by_status'] = PerformanceAnalysis.analyze_by_status(test_df)
    report['analysis_by_difficulty'] = PerformanceAnalysis.analyze_by_difficulty(test_df)
    
    return report


if __name__ == "__main__":
    from data_generator import QuestionnaireDataGenerator
    
    print("Generating test data...")
    generator = QuestionnaireDataGenerator(num_samples=1000)
    test_df = generator.generate_dataset()
    
    print("\nGenerating evaluation report...")
    report = generate_evaluation_report(test_df)
    
    print("\n" + "="*70)
    print("QUESTIONNAIRE EVALUATION REPORT")
    print("="*70)
    
    print(f"\nTest Set Size: {report['test_set_size']}")
    print(f"Case Profiles: {report['num_profiles']}")
    print(f"Validation Statuses: {report['num_statuses']}")
    
    print("\n\nOVERALL METRICS")
    print("-"*70)
    
    metrics = report['overall_metrics']
    
    print("\nCoverage:")
    print(f"  Average Field Coverage: {metrics['coverage']['average_field_coverage']:.1%}")
    print(f"  Full Coverage Rate: {metrics['coverage']['full_coverage_rate']:.1%}")
    
    print("\nROI Analysis:")
    roi = metrics['roi']
    print(f"  Average ROI: {roi['average_roi']:.1f}")
    print(f"  Median ROI: {roi['median_roi']:.1f}")
    print(f"  High ROI (>70): {roi['high_roi_percentage']:.1%}")
    
    print("\nResponse Quality:")
    quality = metrics['response_quality']
    print(f"  Avg Completion Rate: {quality['average_completion_rate']:.1%}")
    print(f"  Avg Response Quality: {quality['average_response_quality']:.2f}/5")
    print(f"  Avg Satisfaction: {quality['average_satisfaction']:.2f}/5")
    print(f"  High Quality Rate: {quality['high_quality_percentage']:.1%}")
    
    print("\n\nPERFORMANCE BY PROFILE")
    print("-"*70)
    for profile, metrics in report['analysis_by_profile'].items():
        print(f"\n{profile}:")
        print(f"  Count: {metrics['count']}")
        print(f"  Avg Effectiveness: {metrics['avg_effectiveness']:.1f}/100")
        print(f"  Completion Rate: {metrics['avg_completion_rate']:.1%}")
        print(f"  Time: {metrics['avg_time_minutes']:.1f} min")
        print(f"  Satisfaction: {metrics['avg_satisfaction']:.2f}/5")
    
    print("\n\nPERFORMANCE BY VALIDATION STATUS")
    print("-"*70)
    for status, metrics in report['analysis_by_status'].items():
        print(f"\n{status}:")
        print(f"  Count: {metrics['count']}")
        print(f"  Avg Effectiveness: {metrics['avg_effectiveness']:.1f}/100")
        print(f"  Time: {metrics['avg_time_minutes']:.1f} min")
        print(f"  Satisfaction: {metrics['avg_satisfaction']:.2f}/5")
