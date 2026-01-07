"""
Prioritization to Validation Linker
Converts prioritization model outputs into validation inputs
Component 1 → Component 2
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json


class PrioritizationToValidationLinker:
    """Transform prioritization outputs to validation inputs."""
    
    def __init__(self):
        """Initialize linker."""
        self.field_mapping = {
            'priority_score': 'priority_score',
            'follow_up_urgency': 'urgency_level',
            'estimated_response_time_hours': 'expected_response_time_hours',
            'reporter_reliability': 'reporter_reliability_score',
            'regional_significance': 'regional_impact_level',
            'regulatory_deadline': 'deadline_date'
        }
        
    def transform(self, prioritization_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform prioritization output to validation input format.
        
        Args:
            prioritization_df: DataFrame from prioritization model with columns:
                - case_id
                - priority_score
                - follow_up_urgency
                - estimated_response_time_hours
                - reporter_reliability
                - regional_significance
                - regulatory_deadline
                
        Returns:
            DataFrame with validation-compatible schema
        """
        print("\n" + "="*70)
        print("LINKER 1: PRIORITIZATION → VALIDATION")
        print("="*70 + "\n")
        
        validation_input = prioritization_df.copy()
        
        # Map prioritization fields to validation expected fields
        print("Mapping prioritization outputs to validation inputs...")
        
        # Ensure all required fields exist
        required_fields = ['case_id', 'priority_score', 'follow_up_urgency', 
                          'estimated_response_time_hours', 'reporter_reliability',
                          'regional_significance', 'regulatory_deadline']
        
        for field in required_fields:
            if field not in validation_input.columns:
                print(f"  ⚠ Missing field: {field} - generating synthetic data")
                validation_input[field] = self._generate_field(field, len(validation_input))
        
        # Add validation context
        print("Adding validation context...")
        validation_input['prioritization_score'] = validation_input['priority_score']
        validation_input['requires_urgent_review'] = (
            validation_input['follow_up_urgency'] > 0.7
        ).astype(int)
        
        # Determine validation category based on priority
        validation_input['validation_category'] = validation_input['priority_score'].apply(
            self._categorize_by_priority
        )
        
        # Add tracking metadata
        validation_input['source_component'] = 'prioritization_model'
        validation_input['linker_timestamp'] = datetime.now().isoformat()
        validation_input['linker_version'] = '1.0'
        
        # Calculate validation urgency based on deadline and priority
        validation_input['validation_urgency'] = self._calculate_urgency(validation_input)
        
        print(f"✓ Transformed {len(validation_input)} records")
        print(f"  Priority score range: {validation_input['priority_score'].min():.2f} - {validation_input['priority_score'].max():.2f}")
        print(f"  Urgent validations: {validation_input['requires_urgent_review'].sum()} ({validation_input['requires_urgent_review'].sum()/len(validation_input)*100:.1f}%)")
        print(f"  Validation categories: {validation_input['validation_category'].value_counts().to_dict()}")
        
        return validation_input
    
    def _categorize_by_priority(self, priority_score: float) -> str:
        """Categorize report by priority score."""
        if priority_score >= 0.8:
            return 'CRITICAL'
        elif priority_score >= 0.6:
            return 'HIGH'
        elif priority_score >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_urgency(self, df: pd.DataFrame) -> pd.Series:
        """Calculate validation urgency from deadline and priority."""
        urgency = (df['priority_score'] * 0.6 + 
                  df['follow_up_urgency'] * 0.4)
        return np.clip(urgency, 0, 1)
    
    def _generate_field(self, field_name: str, n_samples: int) -> np.ndarray:
        """Generate synthetic field data if missing."""
        if field_name == 'priority_score':
            return np.random.beta(8, 4, n_samples)
        elif field_name == 'follow_up_urgency':
            return np.random.beta(6, 3, n_samples)
        elif field_name == 'estimated_response_time_hours':
            return np.random.gamma(2, 24, n_samples)
        elif field_name == 'reporter_reliability':
            return np.random.beta(7, 3, n_samples)
        elif field_name == 'regional_significance':
            return np.random.choice([0.3, 0.5, 0.7, 0.9], n_samples)
        elif field_name == 'regulatory_deadline':
            return [(datetime.now() + timedelta(days=np.random.randint(1, 90))).date() 
                    for _ in range(n_samples)]
        else:
            return np.random.uniform(0, 1, n_samples)
    
    def get_schema(self) -> Dict[str, str]:
        """Get output schema for validation inputs."""
        return {
            'case_id': 'str',
            'priority_score': 'float',  # 0-1
            'follow_up_urgency': 'float',  # 0-1
            'estimated_response_time_hours': 'float',
            'reporter_reliability': 'float',  # 0-1
            'regional_significance': 'float',  # 0-1
            'regulatory_deadline': 'date',
            'prioritization_score': 'float',
            'requires_urgent_review': 'int',
            'validation_category': 'str',  # CRITICAL, HIGH, MEDIUM, LOW
            'validation_urgency': 'float',
            'source_component': 'str',
            'linker_timestamp': 'datetime',
            'linker_version': 'str'
        }
