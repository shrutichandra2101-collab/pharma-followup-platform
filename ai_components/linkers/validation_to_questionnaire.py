"""
Validation to Questionnaire Linker
Converts validation model outputs into questionnaire generator inputs
Component 2 → Component 3
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json


class ValidationToQuestionnaireLinker:
    """Transform validation outputs to questionnaire inputs."""
    
    def __init__(self):
        """Initialize linker."""
        self.field_to_question_category = {
            'patient_demographics': 'Patient',
            'medication_details': 'Medication',
            'adverse_event_details': 'Safety',
            'event_outcome': 'Efficacy',
            'medical_history': 'Medical History',
            'causality_assessment': 'Causality',
            'concomitant_medications': 'Medication',
            'dosage_info': 'Medication',
            'route_of_administration': 'Medication',
            'date_of_onset': 'Safety',
            'severity_assessment': 'Safety',
            'hospitalization_info': 'Patient',
            'recovery_status': 'Efficacy'
        }
        
        self.status_to_questionnaire_type = {
            'ACCEPT': 'quick_verification',
            'CONDITIONAL_ACCEPT': 'targeted_followup',
            'REVIEW': 'comprehensive_followup',
            'REJECT': 'detailed_investigation'
        }
    
    def transform(self, validation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform validation output to questionnaire generator input format.
        
        Args:
            validation_df: DataFrame from validation model with columns:
                - case_id
                - validation_status (ACCEPT/CONDITIONAL_ACCEPT/REVIEW/REJECT)
                - quality_score
                - completeness_score
                - missing_fields (list)
                - anomaly_risk (Low/Medium/High)
                - detected_issues (dict)
                - validation_timestamp
                
        Returns:
            DataFrame with questionnaire-compatible schema
        """
        print("\n" + "="*70)
        print("LINKER 2: VALIDATION → QUESTIONNAIRE")
        print("="*70 + "\n")
        
        questionnaire_input = validation_df.copy()
        
        # Ensure all required fields exist
        print("Validating input schema...")
        required_fields = ['case_id', 'validation_status', 'quality_score', 
                          'completeness_score', 'anomaly_risk']
        
        for field in required_fields:
            if field not in questionnaire_input.columns:
                print(f"  ⚠ Missing field: {field} - generating synthetic data")
                questionnaire_input[field] = self._generate_field(field, len(questionnaire_input))
        
        # Process missing fields - convert from validation format to questionnaire format
        print("Processing missing fields...")
        if 'missing_fields' not in questionnaire_input.columns:
            questionnaire_input['missing_fields'] = questionnaire_input.apply(
                lambda x: self._infer_missing_fields(x['quality_score'], x['completeness_score']),
                axis=1
            )
        else:
            # Ensure missing_fields is list format
            questionnaire_input['missing_fields'] = questionnaire_input['missing_fields'].apply(
                lambda x: x if isinstance(x, list) else [x] if isinstance(x, str) else []
            )
        
        # Map validation status to questionnaire type
        print("Mapping validation status to questionnaire strategy...")
        questionnaire_input['questionnaire_type'] = questionnaire_input['validation_status'].map(
            self.status_to_questionnaire_type
        )
        
        # Categorize missing fields by question category
        print("Categorizing fields for question selection...")
        questionnaire_input['fields_by_category'] = questionnaire_input['missing_fields'].apply(
            self._categorize_fields
        )
        
        # Calculate expected questionnaire difficulty
        questionnaire_input['expected_difficulty'] = questionnaire_input.apply(
            self._estimate_difficulty, axis=1
        )
        
        # Calculate expected completion time
        questionnaire_input['expected_completion_minutes'] = questionnaire_input.apply(
            self._estimate_completion_time, axis=1
        )
        
        # Add anomaly context for targeting harder questions
        questionnaire_input['anomaly_risk_numeric'] = questionnaire_input['anomaly_risk'].map({
            'Low': 0, 'Medium': 1, 'High': 2
        }).fillna(1)
        
        # Add questionnaire generation metadata
        questionnaire_input['source_component'] = 'validation_model'
        questionnaire_input['linker_timestamp'] = datetime.now().isoformat()
        questionnaire_input['linker_version'] = '1.0'
        
        # Calculate priority for questionnaire distribution
        questionnaire_input['questionnaire_priority'] = self._calculate_priority(questionnaire_input)
        
        # Summary statistics
        print(f"✓ Transformed {len(questionnaire_input)} records")
        print(f"  Quality scores: {questionnaire_input['quality_score'].mean():.2f} ± {questionnaire_input['quality_score'].std():.2f}")
        print(f"  Completeness: {questionnaire_input['completeness_score'].mean():.2f} ± {questionnaire_input['completeness_score'].std():.2f}")
        print(f"  Questionnaire types: {questionnaire_input['questionnaire_type'].value_counts().to_dict()}")
        print(f"  Average fields needing questions: {questionnaire_input['missing_fields'].str.len().mean():.2f}")
        print(f"  Expected completion time: {questionnaire_input['expected_completion_minutes'].mean():.2f} minutes")
        
        return questionnaire_input
    
    def _infer_missing_fields(self, quality_score: float, completeness_score: float) -> List[str]:
        """Infer missing fields based on quality and completeness scores."""
        missing = []
        
        # Quality < 60 suggests safety/severity issues
        if quality_score < 60:
            missing.extend(['severity_assessment', 'date_of_onset'])
        
        # Completeness < 70 suggests patient/medication details
        if completeness_score < 70:
            missing.extend(['patient_demographics', 'medication_details', 'dosage_info'])
        
        # Very low scores suggest multiple gaps
        if completeness_score < 40:
            missing.extend(['medical_history', 'concomitant_medications', 'hospitalization_info'])
        
        return list(set(missing))  # Remove duplicates
    
    def _categorize_fields(self, missing_fields: List[str]) -> Dict[str, List[str]]:
        """Categorize missing fields by question category."""
        categorized = {
            'Safety': [],
            'Efficacy': [],
            'Patient': [],
            'Medication': [],
            'Medical History': [],
            'Causality': []
        }
        
        for field in missing_fields:
            category = self.field_to_question_category.get(field, 'Safety')
            categorized[category].append(field)
        
        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}
    
    def _estimate_difficulty(self, row: pd.Series) -> str:
        """Estimate questionnaire difficulty."""
        num_missing = len(row.get('missing_fields', []))
        anomaly_numeric = row.get('anomaly_risk_numeric', 1)
        
        difficulty_score = (num_missing * 0.4 + 
                          (1 - row.get('quality_score', 50)/100) * 0.3 +
                          anomaly_numeric * 0.3)
        
        if difficulty_score >= 2.0:
            return 'Hard'
        elif difficulty_score >= 1.0:
            return 'Medium'
        else:
            return 'Easy'
    
    def _estimate_completion_time(self, row: pd.Series) -> float:
        """Estimate questionnaire completion time in minutes."""
        base_time = 3  # Base 3 minutes
        per_field_time = 1.5  # 1.5 minutes per missing field
        anomaly_multiplier = 1.0 + (row.get('anomaly_risk_numeric', 1) * 0.3)
        
        num_missing = len(row.get('missing_fields', []))
        estimated_time = (base_time + per_field_time * num_missing) * anomaly_multiplier
        
        return min(estimated_time, 20)  # Cap at 20 minutes
    
    def _calculate_priority(self, df: pd.DataFrame) -> pd.Series:
        """Calculate questionnaire distribution priority."""
        priority = (
            (df['completeness_score'] < 50).astype(int) * 3 +  # Critical gaps
            (df['quality_score'] < 60).astype(int) * 2 +        # Quality issues
            (df['anomaly_risk_numeric'] == 2).astype(int) * 2   # High anomaly risk
        )
        
        # Normalize to 0-1 range
        return priority / priority.max() if priority.max() > 0 else 0.5
    
    def _generate_field(self, field_name: str, n_samples: int) -> Any:
        """Generate synthetic field data if missing."""
        if field_name == 'validation_status':
            return np.random.choice(['ACCEPT', 'CONDITIONAL_ACCEPT', 'REVIEW', 'REJECT'],
                                  n_samples, p=[0.35, 0.25, 0.25, 0.15])
        elif field_name == 'quality_score':
            return np.random.beta(8, 2, n_samples) * 100
        elif field_name == 'completeness_score':
            return np.random.beta(7, 2, n_samples) * 100
        elif field_name == 'anomaly_risk':
            return np.random.choice(['Low', 'Medium', 'High'],
                                  n_samples, p=[0.6, 0.3, 0.1])
        else:
            return np.random.uniform(0, 1, n_samples)
    
    def get_schema(self) -> Dict[str, str]:
        """Get output schema for questionnaire inputs."""
        return {
            'case_id': 'str',
            'validation_status': 'str',  # ACCEPT, CONDITIONAL_ACCEPT, REVIEW, REJECT
            'quality_score': 'float',  # 0-100
            'completeness_score': 'float',  # 0-100
            'missing_fields': 'list',
            'anomaly_risk': 'str',  # Low, Medium, High
            'anomaly_risk_numeric': 'int',
            'questionnaire_type': 'str',
            'fields_by_category': 'dict',
            'expected_difficulty': 'str',  # Easy, Medium, Hard
            'expected_completion_minutes': 'float',
            'questionnaire_priority': 'float',  # 0-1
            'source_component': 'str',
            'linker_timestamp': 'datetime',
            'linker_version': 'str'
        }
