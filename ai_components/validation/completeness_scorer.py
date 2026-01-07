"""
Field completeness scoring module.
Calculates data completeness as percentage of important fields filled.
Uses weighted scoring where critical fields have higher importance.
"""

import pandas as pd
import numpy as np
from validation_constants import MANDATORY_FIELDS, OPTIONAL_FIELDS, FIELD_WEIGHTS


class CompletenessScorer:
    """Calculate field completeness with weighted importance."""
    
    def __init__(self):
        self.mandatory_fields = MANDATORY_FIELDS
        self.optional_fields = OPTIONAL_FIELDS
        self.field_weights = FIELD_WEIGHTS
        self.total_weight = sum(FIELD_WEIGHTS.values())
    
    def calculate_score(self, report):
        """
        Calculate completeness score for a single report.
        Returns 0-100 based on fields filled and their weights.
        
        Args:
            report: dict or Series representing one adverse event report
            
        Returns:
            float: Completeness score (0-100)
        """
        filled_weight = 0.0
        
        # Convert Series to dict if needed
        if isinstance(report, pd.Series):
            report = report.to_dict()
        
        # Check each field
        for field, weight in self.field_weights.items():
            if field in report:
                value = report[field]
                # Check if field is filled (not None, not empty string, not NaN)
                if pd.notna(value) and value != '' and value != 'NULL':
                    filled_weight += weight
        
        # Calculate percentage
        score = (filled_weight / self.total_weight) * 100
        return round(score, 2)
    
    def calculate_scores_batch(self, df):
        """
        Calculate completeness scores for entire DataFrame.
        
        Args:
            df: DataFrame with adverse event reports
            
        Returns:
            Series with completeness score for each report
        """
        scores = df.apply(self.calculate_score, axis=1)
        return scores
    
    def get_missing_fields(self, report):
        """
        Identify missing fields for a report.
        
        Args:
            report: dict or Series representing one report
            
        Returns:
            dict with missing mandatory and optional fields
        """
        if isinstance(report, pd.Series):
            report = report.to_dict()
        
        missing = {
            'mandatory': [],
            'optional': [],
            'total_missing': 0
        }
        
        # Check mandatory fields
        for field in self.mandatory_fields:
            if field not in report or pd.isna(report[field]) or report[field] == '':
                missing['mandatory'].append(field)
        
        # Check optional fields (those with weights)
        for field in self.optional_fields:
            if field not in report or pd.isna(report[field]) or report[field] == '':
                missing['optional'].append(field)
        
        missing['total_missing'] = len(missing['mandatory']) + len(missing['optional'])
        return missing
    
    def get_field_contribution(self, report):
        """
        Analyze which fields contribute how much to the score.
        
        Args:
            report: dict or Series
            
        Returns:
            DataFrame with field contributions
        """
        if isinstance(report, pd.Series):
            report = report.to_dict()
        
        contributions = []
        
        for field, weight in self.field_weights.items():
            is_filled = field in report and pd.notna(report[field]) and report[field] != ''
            contribution = weight if is_filled else 0
            contribution_pct = (contribution / self.total_weight) * 100
            
            contributions.append({
                'field': field,
                'weight': weight,
                'is_filled': is_filled,
                'contribution': contribution,
                'contribution_pct': round(contribution_pct, 2)
            })
        
        return pd.DataFrame(contributions)
    
    def score_interpretation(self, score):
        """
        Interpret a completeness score.
        
        Args:
            score: float (0-100)
            
        Returns:
            dict with interpretation and recommendations
        """
        if score >= 80:
            return {
                'level': 'EXCELLENT',
                'description': 'All critical fields present',
                'risk': 'Low'
            }
        elif score >= 60:
            return {
                'level': 'GOOD',
                'description': 'Most important fields present',
                'risk': 'Low-Medium'
            }
        elif score >= 40:
            return {
                'level': 'FAIR',
                'description': 'Some important fields missing',
                'risk': 'Medium'
            }
        elif score >= 20:
            return {
                'level': 'POOR',
                'description': 'Many important fields missing',
                'risk': 'Medium-High'
            }
        else:
            return {
                'level': 'CRITICAL',
                'description': 'Most fields missing',
                'risk': 'High'
            }
    
    def generate_completeness_report(self, df):
        """
        Generate completeness analysis report for dataset.
        
        Args:
            df: DataFrame with reports
            
        Returns:
            dict with summary statistics
        """
        scores = self.calculate_scores_batch(df)
        
        return {
            'total_reports': len(df),
            'average_completeness': round(scores.mean(), 2),
            'min_completeness': round(scores.min(), 2),
            'max_completeness': round(scores.max(), 2),
            'std_completeness': round(scores.std(), 2),
            'distribution': {
                'excellent': (scores >= 80).sum(),
                'good': ((scores >= 60) & (scores < 80)).sum(),
                'fair': ((scores >= 40) & (scores < 60)).sum(),
                'poor': ((scores >= 20) & (scores < 40)).sum(),
                'critical': (scores < 20).sum()
            },
            'scores': scores
        }


if __name__ == "__main__":
    print("Completeness Scorer Module")
    print("Calculates field completeness with weighted importance")
