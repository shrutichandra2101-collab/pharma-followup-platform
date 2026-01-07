"""
Smart Follow-Up Questionnaire Generator - Data Generator Module
Generate synthetic follow-up cases with gaps and response data

Step 2: Create training dataset of follow-up scenarios with realistic gaps
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import json


class QuestionnaireDataGenerator:
    """Generate synthetic follow-up cases with questionnaire responses."""
    
    def __init__(self, num_samples: int = 5000, random_seed: int = 42):
        """
        Initialize data generator.
        
        Args:
            num_samples: Number of synthetic cases to generate
            random_seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Validation profile types (from validation component output)
        self.validation_statuses = ['ACCEPT', 'CONDITIONAL_ACCEPT', 'REVIEW', 'REJECT']
        self.anomaly_risks = ['Low', 'Medium', 'High']
        self.case_profiles = ['Complete', 'Missing_Safety', 'Missing_Efficacy', 
                             'Missing_Patient', 'Missing_Multiple', 'Anomalous']
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate synthetic follow-up dataset.
        
        Returns:
            DataFrame with case_id, validation info, missing fields, and response data
        """
        print(f"\nGenerating {self.num_samples} synthetic follow-up cases...")
        
        cases = []
        for i in range(self.num_samples):
            case = self._generate_case(i)
            cases.append(case)
        
        df = pd.DataFrame(cases)
        print(f"✓ Generated {len(df)} cases")
        return df
    
    def _generate_case(self, case_id: int) -> Dict[str, Any]:
        """Generate a single case."""
        # Determine case profile (what's missing)
        profile = np.random.choice(self.case_profiles, p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10])
        
        # Generate validation results
        quality_score = np.random.beta(8, 2) * 100  # Skewed towards high quality
        completeness = np.random.beta(7, 2) * 100
        
        # Adjust scores based on profile
        if profile == 'Missing_Safety':
            quality_score *= 0.6
            completeness *= 0.5
        elif profile == 'Missing_Multiple':
            quality_score *= 0.4
            completeness *= 0.3
        elif profile == 'Anomalous':
            quality_score *= 0.7
        
        quality_score = np.clip(quality_score, 0, 100)
        completeness = np.clip(completeness, 0, 100)
        
        # Determine validation status based on scores
        if quality_score < 40 or completeness < 30:
            status = 'REJECT'
            anomaly_risk = np.random.choice(['High', 'Medium'], p=[0.7, 0.3])
        elif quality_score < 70:
            status = np.random.choice(['REVIEW', 'CONDITIONAL_ACCEPT'], p=[0.6, 0.4])
            anomaly_risk = np.random.choice(['Medium', 'Low'], p=[0.6, 0.4])
        else:
            status = np.random.choice(['ACCEPT', 'CONDITIONAL_ACCEPT'], p=[0.8, 0.2])
            anomaly_risk = np.random.choice(['Low', 'Medium'], p=[0.9, 0.1])
        
        # Generate missing fields based on profile
        missing_fields = self._generate_missing_fields(profile)
        
        # Determine which questions would be selected
        num_questions = len(missing_fields) + np.random.randint(1, 4)
        num_questions = np.clip(num_questions, 3, 12)
        
        # Generate response characteristics
        response_completion = np.random.beta(8, 3)  # High completion bias
        avg_response_quality = np.random.beta(6, 2) * 5
        
        # Adjust response quality based on case difficulty
        if profile in ['Missing_Multiple', 'Anomalous']:
            avg_response_quality *= 0.7
            response_completion *= 0.8
        
        completion_time = int(np.random.gamma(5, 60) + 120)  # 2-20 minutes
        
        # Calculate effectiveness score
        effectiveness = self._calculate_effectiveness(
            missing_fields, 
            num_questions, 
            response_completion,
            avg_response_quality,
            status
        )
        
        case = {
            'case_id': case_id,
            'profile': profile,
            'validation_status': status,
            'quality_score': round(quality_score, 2),
            'completeness_score': round(completeness, 2),
            'anomaly_risk': anomaly_risk,
            'num_missing_fields': len(missing_fields),
            'missing_fields': missing_fields,
            'num_selected_questions': num_questions,
            'response_completion_rate': round(response_completion, 3),
            'avg_response_quality': round(avg_response_quality, 2),
            'estimated_completion_time': completion_time,
            'actual_completion_time': int(completion_time * np.random.uniform(0.7, 1.3)),
            'questionnaire_effectiveness': round(effectiveness, 2),
            'critical_fields_obtained': int(len(missing_fields) * response_completion),
            'user_satisfaction': round(np.random.uniform(2, 5), 2),
        }
        
        return case
    
    def _generate_missing_fields(self, profile: str) -> List[str]:
        """Generate list of missing fields based on profile."""
        all_fields = [
            'event_severity', 'event_outcome', 'hospitalization_flag',
            'days_to_event', 'event_date', 'dechallenge_result',
            'rechallenge_result', 'causality', 'concomitant_medications',
            'drug_interactions', 'medical_history', 'comorbidities',
            'efficacy_outcome', 'treatment_response', 'onset_time',
            'dose_adequacy', 'patient_age', 'patient_weight',
            'pregnancy_status', 'renal_function', 'hepatic_function',
            'smoking_status', 'alcohol_consumption', 'allergy_history',
            'prior_adrs', 'dosage', 'frequency', 'duration',
            'dose_adjustment', 'discontinuation_reason', 'indication'
        ]
        
        if profile == 'Complete':
            return []
        elif profile == 'Missing_Safety':
            safety_fields = ['event_severity', 'hospitalization_flag', 'dechallenge_result',
                           'concomitant_medications', 'medical_history']
            return list(np.random.choice(safety_fields, size=np.random.randint(2, 4), replace=False))
        elif profile == 'Missing_Efficacy':
            efficacy_fields = ['efficacy_outcome', 'treatment_response', 'onset_time',
                             'dose_adequacy', 'treatment_duration']
            return list(np.random.choice(efficacy_fields, size=np.random.randint(2, 4), replace=False))
        elif profile == 'Missing_Patient':
            patient_fields = ['patient_age', 'patient_weight', 'pregnancy_status',
                            'smoking_status', 'alcohol_consumption', 'renal_function']
            return list(np.random.choice(patient_fields, size=np.random.randint(2, 5), replace=False))
        elif profile == 'Missing_Multiple':
            # Missing from multiple categories
            sample_size = np.random.randint(6, 10)
            return list(np.random.choice(all_fields, size=sample_size, replace=False))
        else:  # Anomalous
            # Unusual pattern but not necessarily many missing fields
            sample_size = np.random.randint(2, 5)
            return list(np.random.choice(all_fields, size=sample_size, replace=False))
    
    def _calculate_effectiveness(self, missing_fields: List[str], num_questions: int,
                                completion: float, quality: float, status: str) -> float:
        """Calculate questionnaire effectiveness score."""
        base_score = 50
        
        # Higher effectiveness if more fields were missing (bigger potential gain)
        coverage_bonus = min(len(missing_fields) * 5, 20)
        
        # Higher effectiveness if responses were complete and good quality
        response_bonus = (completion * 0.5 + (quality / 5) * 0.5) * 20
        
        # Better status = harder to improve (diminishing returns)
        status_penalty = {'ACCEPT': 5, 'CONDITIONAL_ACCEPT': 0, 'REVIEW': -5, 'REJECT': -10}
        
        effectiveness = base_score + coverage_bonus + response_bonus + status_penalty.get(status, 0)
        return np.clip(effectiveness, 0, 100)
    
    def generate_question_response_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate question-level response data.
        
        Args:
            df: Dataset from generate_dataset()
            
        Returns:
            DataFrame with question-level response information
        """
        responses = []
        
        for _, case in df.iterrows():
            for q_idx in range(case['num_selected_questions']):
                question_id = f"Q_{case['case_id']}_{q_idx}"
                
                # Probability question gets answered
                answered = np.random.random() < case['response_completion_rate']
                
                response_quality = 0
                if answered:
                    response_quality = np.random.beta(4, 1) * case['avg_response_quality']
                    response_quality = np.clip(response_quality, 0, 5)
                
                response = {
                    'case_id': case['case_id'],
                    'question_sequence': q_idx,
                    'question_id': question_id,
                    'was_answered': int(answered),
                    'response_quality': round(response_quality, 2),
                    'useful_for_field': bool(np.random.random() < 0.7 if answered else False),
                    'clarity_rating': int(np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.3, 0.35, 0.2])),
                    'time_to_answer': int(np.random.gamma(3, 20)) if answered else 0,
                }
                responses.append(response)
        
        return pd.DataFrame(responses)
    
    def generate_training_test_split(self, df: pd.DataFrame, test_size: float = 0.2
                                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and test sets.
        
        Args:
            df: Full dataset
            test_size: Proportion for test set
            
        Returns:
            (train_df, test_df)
        """
        test_count = int(len(df) * test_size)
        test_indices = np.random.choice(len(df), size=test_count, replace=False)
        
        test_df = df.iloc[test_indices].reset_index(drop=True)
        train_df = df.drop(test_indices).reset_index(drop=True)
        
        return train_df, test_df
    
    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics."""
        print("\n" + "="*70)
        print("QUESTIONNAIRE DATASET SUMMARY")
        print("="*70)
        
        print(f"\nTotal Cases: {len(df)}")
        print(f"\nCASE PROFILE DISTRIBUTION")
        print("-"*70)
        for profile in self.case_profiles:
            count = (df['profile'] == profile).sum()
            pct = count / len(df) * 100
            print(f"{profile:20s}: {count:5d} ({pct:5.1f}%)")
        
        print(f"\n\nVALIDATION STATUS DISTRIBUTION")
        print("-"*70)
        for status in self.validation_statuses:
            count = (df['validation_status'] == status).sum()
            pct = count / len(df) * 100
            print(f"{status:20s}: {count:5d} ({pct:5.1f}%)")
        
        print(f"\n\nQUESTIONNAIRE CHARACTERISTICS")
        print("-"*70)
        print(f"Avg Missing Fields:        {df['num_missing_fields'].mean():.2f}")
        print(f"Avg Questions Selected:    {df['num_selected_questions'].mean():.2f}")
        print(f"Avg Completion Rate:       {df['response_completion_rate'].mean():.1%}")
        print(f"Avg Response Quality:      {df['avg_response_quality'].mean():.2f}/5")
        print(f"Avg Completion Time:       {df['actual_completion_time'].mean():.0f}s ({df['actual_completion_time'].mean()/60:.1f} min)")
        print(f"Avg Effectiveness:         {df['questionnaire_effectiveness'].mean():.2f}/100")
        
        print(f"\n\nSCORE STATISTICS")
        print("-"*70)
        print(f"Quality Score:             {df['quality_score'].mean():.1f} (std: {df['quality_score'].std():.1f})")
        print(f"Completeness Score:        {df['completeness_score'].mean():.1f} (std: {df['completeness_score'].std():.1f})")
        print(f"User Satisfaction:         {df['user_satisfaction'].mean():.2f}/5")
        
        print(f"\n\nANOMOLY RISK DISTRIBUTION")
        print("-"*70)
        for risk in self.anomaly_risks:
            count = (df['anomaly_risk'] == risk).sum()
            pct = count / len(df) * 100
            print(f"{risk:10s}: {count:5d} ({pct:5.1f}%)")
        
        print("\n")


if __name__ == "__main__":
    # Generate dataset
    generator = QuestionnaireDataGenerator(num_samples=5000)
    df = generator.generate_dataset()
    
    # Print summary
    generator.print_summary(df)
    
    # Generate question-level responses
    print("\nGenerating question-level response data...")
    response_df = generator.generate_question_response_data(df)
    print(f"✓ Generated {len(response_df)} question-level responses")
    
    # Show sample responses
    print("\nSample responses:")
    print(response_df.head(10).to_string())
    
    # Split into train/test
    train_df, test_df = generator.generate_training_test_split(df)
    print(f"\n✓ Train/Test Split: {len(train_df)} / {len(test_df)}")
