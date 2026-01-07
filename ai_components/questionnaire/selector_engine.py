"""
Smart Follow-Up Questionnaire Generator - Selector Engine Module
Smart selection logic for choosing relevant questions based on validation gaps

Step 3: Select the most relevant questions based on missing fields and case profile
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib


class GapAnalyzer:
    """Analyze validation gaps and identify critical missing fields."""
    
    def __init__(self):
        """Initialize gap analyzer."""
        self.field_to_category = {
            # Safety
            'event_severity': 'Safety', 'hospitalization_flag': 'Safety',
            'dechallenge_result': 'Safety', 'concomitant_medications': 'Safety',
            'medical_history': 'Safety', 'drug_interactions': 'Safety',
            # Efficacy
            'efficacy_outcome': 'Efficacy', 'treatment_response': 'Efficacy',
            'onset_time': 'Efficacy', 'dose_adequacy': 'Efficacy',
            # Patient
            'patient_age': 'Patient Info', 'patient_weight': 'Patient Info',
            'pregnancy_status': 'Patient Info', 'smoking_status': 'Patient Info',
            'renal_function': 'Patient Info', 'hepatic_function': 'Patient Info',
            # Medication
            'dosage': 'Medication', 'frequency': 'Medication',
            'duration': 'Medication', 'discontinuation_reason': 'Medication',
            'indication': 'Medication',
            # Causality
            'causality': 'Causality',
        }
        
        self.critical_fields = ['event_severity', 'hospitalization_flag', 'patient_age',
                               'dosage', 'indication', 'causality', 'dechallenge_result']
    
    def analyze_gaps(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze validation gaps in a case.
        
        Args:
            case: Dict with validation results (quality_score, completeness, missing_fields, etc.)
            
        Returns:
            Dict with gap analysis results
        """
        missing = case.get('missing_fields', [])
        
        # Categorize missing fields
        gap_by_category = {}
        for field in missing:
            category = self.field_to_category.get(field, 'Other')
            if category not in gap_by_category:
                gap_by_category[category] = []
            gap_by_category[category].append(field)
        
        # Identify critical gaps
        critical_gaps = [f for f in missing if f in self.critical_fields]
        
        # Calculate gap severity
        num_gaps = len(missing)
        gap_severity = min(1.0, num_gaps / 8)  # Normalize by typical # of gaps
        
        return {
            'missing_fields': missing,
            'gap_by_category': gap_by_category,
            'critical_gaps': critical_gaps,
            'num_missing': num_gaps,
            'gap_severity': gap_severity,
            'most_important_category': max(gap_by_category.keys(), 
                                          key=lambda k: len(gap_by_category[k])) 
                                       if gap_by_category else None,
        }


class QuestionSelector:
    """Select relevant questions based on validation gaps and case characteristics."""
    
    def __init__(self, question_bank=None):
        """
        Initialize question selector.
        
        Args:
            question_bank: QuestionBank object
        """
        self.question_bank = question_bank
        self.gap_analyzer = GapAnalyzer()
        self.decision_tree = None
        self.relevance_scorer = None
        
        # Relevant questions for each field (field -> list of question IDs)
        self.field_to_questions = self._initialize_field_to_questions()
    
    def _initialize_field_to_questions(self) -> Dict[str, List[str]]:
        """Initialize mapping from fields to question IDs."""
        # If question_bank available, use actual data
        # Otherwise use predefined mapping
        return {
            'event_severity': ['S001', 'S008', 'C001'],
            'hospitalization_flag': ['S003', 'S008'],
            'dechallenge_result': ['S005', 'S006'],
            'concomitant_medications': ['S007', 'S009'],
            'medical_history': ['S010', 'H001', 'H002'],
            'patient_age': ['P001', 'P006'],
            'patient_weight': ['P001'],
            'pregnancy_status': ['P002'],
            'smoking_status': ['P007'],
            'dosage': ['M001'],
            'frequency': ['M002'],
            'indication': ['M009'],
            'causality': ['C001', 'C002', 'C003'],
            'treatment_response': ['E001', 'E004'],
            'drug_interactions': ['S007'],
        }
    
    def select_questions(self, case: Dict[str, Any], max_questions: int = 10,
                        difficulty_mix: str = 'balanced') -> List[Dict[str, Any]]:
        """
        Select the most relevant questions for a case.
        
        Args:
            case: Case information from validation/prioritization
            max_questions: Maximum number of questions to select
            difficulty_mix: 'easy', 'hard', or 'balanced'
            
        Returns:
            List of selected questions with metadata
        """
        # Analyze gaps
        gap_analysis = self.gap_analyzer.analyze_gaps(case)
        missing_fields = gap_analysis['missing_fields']
        
        if not missing_fields:
            return []
        
        # Collect candidate questions by field
        candidates = {}
        for field in missing_fields:
            questions = self.field_to_questions.get(field, [])
            for q_id in questions:
                if q_id not in candidates:
                    candidates[q_id] = {
                        'question_id': q_id,
                        'field_targets': [field],
                        'is_critical': field in gap_analysis['critical_gaps'],
                    }
                else:
                    candidates[q_id]['field_targets'].append(field)
        
        # Rank candidates by relevance
        ranked = self._rank_questions(candidates, gap_analysis, case)
        
        # Apply difficulty filter
        selected = self._apply_difficulty_filter(ranked, difficulty_mix)
        
        # Trim to max_questions
        selected = selected[:max_questions]
        
        return selected
    
    def _rank_questions(self, candidates: Dict[str, Dict], gap_analysis: Dict,
                       case: Dict) -> List[Dict[str, Any]]:
        """Rank candidate questions by relevance score."""
        ranked = []
        
        for q_id, q_info in candidates.items():
            # Calculate relevance score
            score = 0.0
            
            # Higher score for critical fields (50% weight)
            if q_info['is_critical']:
                score += 50
            else:
                score += 30
            
            # Higher score for addressing multiple missing fields (20% weight)
            num_fields = len(q_info['field_targets'])
            score += min(20, num_fields * 10)
            
            # Higher score for low gap severity (30% weight)
            # Easier cases need less thorough questionnaires
            difficulty_multiplier = 1.0 if gap_analysis['gap_severity'] > 0.5 else 0.8
            score *= difficulty_multiplier
            
            q_info['relevance_score'] = score
            ranked.append(q_info)
        
        # Sort by relevance score (highest first)
        ranked.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return ranked
    
    def _apply_difficulty_filter(self, questions: List[Dict], 
                                 difficulty_mix: str) -> List[Dict]:
        """
        Adjust selection based on difficulty preference.
        
        Args:
            questions: Ranked questions
            difficulty_mix: 'easy', 'hard', or 'balanced'
            
        Returns:
            Filtered questions
        """
        # In actual implementation, would filter by question difficulty
        # For now, just return ranked list
        return questions
    
    def train_selector_model(self, train_df: pd.DataFrame):
        """
        Train decision tree to learn question selection patterns.
        
        Args:
            train_df: Training dataset with cases and question selections
        """
        print("Training question selector model...")
        
        # Prepare features
        features = []
        labels = []
        
        for _, row in train_df.iterrows():
            # Features: case characteristics
            feature = {
                'quality_score': row['quality_score'],
                'completeness_score': row['completeness_score'],
                'num_missing_fields': row['num_missing_fields'],
                'anomaly_risk_value': {'Low': 0, 'Medium': 1, 'High': 2}.get(row['anomaly_risk'], 0),
                'status_value': {'ACCEPT': 0, 'CONDITIONAL_ACCEPT': 1, 'REVIEW': 2, 'REJECT': 3}.get(
                    row['validation_status'], 0),
            }
            features.append(list(feature.values()))
            
            # Label: effectiveness (high if > 70)
            labels.append(1 if row['questionnaire_effectiveness'] > 70 else 0)
        
        X = np.array(features)
        y = np.array(labels)
        
        # Train decision tree
        self.decision_tree = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=20,
            random_state=42
        )
        self.decision_tree.fit(X, y)
        
        accuracy = self.decision_tree.score(X, y)
        print(f"✓ Decision tree trained - Accuracy: {accuracy:.3f}")
    
    def train_relevance_scorer(self, train_df: pd.DataFrame):
        """
        Train logistic regression for relevance scoring.
        
        Args:
            train_df: Training dataset
        """
        print("Training relevance scorer model...")
        
        features = []
        labels = []
        
        for _, row in train_df.iterrows():
            feature = {
                'num_missing': row['num_missing_fields'],
                'quality': row['quality_score'],
                'completion_rate': row['response_completion_rate'],
            }
            features.append(list(feature.values()))
            
            # Label: 1 if good quality responses (> 3/5)
            labels.append(1 if row['avg_response_quality'] > 3 else 0)
        
        X = np.array(features)
        y = np.array(labels)
        
        self.relevance_scorer = LogisticRegression(random_state=42)
        self.relevance_scorer.fit(X, y)
        
        accuracy = self.relevance_scorer.score(X, y)
        print(f"✓ Relevance scorer trained - Accuracy: {accuracy:.3f}")
    
    def predict_question_quality(self, case: Dict[str, Any]) -> float:
        """
        Predict how useful this case's questions will be.
        
        Args:
            case: Case information
            
        Returns:
            Predicted quality (0-1)
        """
        if self.relevance_scorer is None:
            return 0.5
        
        X = np.array([[
            case.get('num_missing_fields', 0),
            case.get('quality_score', 50),
            case.get('response_completion_rate', 0.5),
        ]])
        
        probability = self.relevance_scorer.predict_proba(X)[0][1]
        return probability
    
    def save_models(self, path_tree: str = None, path_scorer: str = None):
        """Save trained models."""
        if path_tree and self.decision_tree:
            joblib.dump(self.decision_tree, path_tree)
            print(f"✓ Saved decision tree to {path_tree}")
        
        if path_scorer and self.relevance_scorer:
            joblib.dump(self.relevance_scorer, path_scorer)
            print(f"✓ Saved relevance scorer to {path_scorer}")
    
    def load_models(self, path_tree: str = None, path_scorer: str = None):
        """Load pre-trained models."""
        if path_tree:
            self.decision_tree = joblib.load(path_tree)
            print(f"✓ Loaded decision tree from {path_tree}")
        
        if path_scorer:
            self.relevance_scorer = joblib.load(path_scorer)
            print(f"✓ Loaded relevance scorer from {path_scorer}")


if __name__ == "__main__":
    # Test selector
    selector = QuestionSelector()
    
    # Test case
    test_case = {
        'missing_fields': ['event_severity', 'hospitalization_flag', 'patient_age', 'dosage'],
        'quality_score': 55,
        'completeness_score': 60,
        'validation_status': 'REVIEW',
        'anomaly_risk': 'Medium',
    }
    
    print("\nTest Case:")
    print(f"  Missing Fields: {test_case['missing_fields']}")
    print(f"  Quality: {test_case['quality_score']}")
    print(f"  Status: {test_case['validation_status']}")
    
    selected = selector.select_questions(test_case, max_questions=8)
    
    print(f"\n\nSelected {len(selected)} Questions:")
    print("-" * 70)
    for i, q in enumerate(selected, 1):
        print(f"{i}. [{q['question_id']}] Score: {q['relevance_score']:.1f}")
        print(f"   Targets: {', '.join(q['field_targets'])}")
        print(f"   Critical: {q['is_critical']}")
