"""
Smart Follow-Up Questionnaire Generator - Response Predictor Module
Machine learning model for predicting response quality

Step 5: Build ML model to predict how useful responses will be
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib


class ResponseQualityPredictor:
    """Predict response quality and completion likelihood."""
    
    def __init__(self):
        """Initialize predictor."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'quality_score', 'completeness_score', 'num_missing_fields',
            'anomaly_risk_numeric', 'num_selected_questions'
        ]
    
    def prepare_features(self, case: Dict[str, Any]) -> np.ndarray:
        """
        Prepare features for prediction.
        
        Args:
            case: Case dictionary
            
        Returns:
            Feature array
        """
        anomaly_map = {'Low': 0, 'Medium': 1, 'High': 2}
        
        features = np.array([[
            case.get('quality_score', 50),
            case.get('completeness_score', 50),
            case.get('num_missing_fields', 3),
            anomaly_map.get(case.get('anomaly_risk', 'Low'), 1),
            case.get('num_selected_questions', 8),
        ]])
        
        return features
    
    def train(self, train_df: pd.DataFrame):
        """
        Train logistic regression model.
        
        Args:
            train_df: Training data from questionnaire dataset
        """
        print("Training response quality predictor...")
        
        # Prepare features
        X = train_df[[
            'quality_score', 'completeness_score', 'num_missing_fields',
            'num_selected_questions'
        ]].values.copy()
        
        # Map anomaly risk to numeric
        anomaly_map = {'Low': 0, 'Medium': 1, 'High': 2}
        anomaly_numeric = train_df['anomaly_risk'].map(anomaly_map).values.reshape(-1, 1)
        X = np.hstack([X, anomaly_numeric])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Binary label: good response quality (> 3/5)
        y = (train_df['avg_response_quality'] > 3).astype(int).values
        
        # Train logistic regression
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_scaled, y)
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        
        train_accuracy = self.model.score(X_scaled, y)
        print(f"✓ Model trained")
        print(f"  Training Accuracy: {train_accuracy:.3f}")
        print(f"  Cross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        return train_accuracy
    
    def predict_probability(self, case: Dict[str, Any]) -> float:
        """
        Predict probability of good quality response.
        
        Args:
            case: Case information
            
        Returns:
            Probability (0-1)
        """
        if self.model is None:
            return 0.5
        
        features = self.prepare_features(case)
        features_scaled = self.scaler.transform(features)
        probability = self.model.predict_proba(features_scaled)[0][1]
        
        return probability
    
    def predict_completion_rate(self, case: Dict[str, Any], num_questions: int) -> float:
        """
        Predict question completion rate.
        
        Args:
            case: Case information
            num_questions: Number of questions in questionnaire
            
        Returns:
            Estimated completion rate (0-1)
        """
        base_rate = 0.85  # 85% baseline
        
        # Adjust by case difficulty
        quality_factor = case.get('quality_score', 50) / 100
        completion_adjustment = (quality_factor - 0.5) * 0.2
        
        # Adjust by questionnaire length
        # Longer questionnaires have lower completion
        length_adjustment = -0.01 * (num_questions - 8)
        
        estimated_rate = np.clip(base_rate + completion_adjustment + length_adjustment, 0.5, 0.95)
        
        return estimated_rate
    
    def predict_response_time(self, case: Dict[str, Any], num_questions: int,
                             estimated_seconds: int) -> int:
        """
        Predict actual response time.
        
        Args:
            case: Case information
            num_questions: Number of questions
            estimated_seconds: Estimated time from questionnaire
            
        Returns:
            Predicted actual time in seconds
        """
        # User expertise adjustment
        quality_score = case.get('quality_score', 50)
        if quality_score > 80:
            # High quality = likely expert, faster response
            time_multiplier = 0.8
        elif quality_score < 40:
            # Low quality = likely needs more time
            time_multiplier = 1.3
        else:
            time_multiplier = 1.0
        
        predicted_time = int(estimated_seconds * time_multiplier)
        
        return predicted_time
    
    def predict_field_coverage(self, case: Dict[str, Any], missing_fields: List[str]) -> float:
        """
        Predict % of critical missing fields that will be addressed.
        
        Args:
            case: Case information
            missing_fields: List of missing fields
            
        Returns:
            Estimated coverage (0-1)
        """
        # Critical fields more likely to be covered
        critical_fields = ['event_severity', 'hospitalization_flag', 'patient_age', 'dosage']
        
        num_critical = len([f for f in missing_fields if f in critical_fields])
        
        quality_factor = case.get('quality_score', 50) / 100
        
        # Higher quality = better coverage
        base_coverage = 0.6 + (quality_factor * 0.3)
        
        # Critical fields increase coverage likelihood
        critical_bonus = min(num_critical * 0.05, 0.15)
        
        coverage = np.clip(base_coverage + critical_bonus, 0.3, 0.95)
        
        return coverage
    
    def save_model(self, path: str):
        """Save trained model and scaler."""
        joblib.dump(self.model, path.replace('.pkl', '_model.pkl'))
        joblib.dump(self.scaler, path.replace('.pkl', '_scaler.pkl'))
        print(f"✓ Saved response predictor to {path}")
    
    def load_model(self, path: str):
        """Load trained model and scaler."""
        self.model = joblib.load(path.replace('.pkl', '_model.pkl'))
        self.scaler = joblib.load(path.replace('.pkl', '_scaler.pkl'))
        print(f"✓ Loaded response predictor from {path}")


if __name__ == "__main__":
    # Test predictor
    from data_generator import QuestionnaireDataGenerator
    
    print("Generating training data...")
    generator = QuestionnaireDataGenerator(num_samples=5000)
    df = generator.generate_dataset()
    
    predictor = ResponseQualityPredictor()
    accuracy = predictor.train(df)
    
    print("\n" + "="*70)
    print("PREDICTOR EVALUATION")
    print("="*70)
    
    # Test on different case types
    test_cases = [
        {
            'name': 'High Quality Case',
            'quality_score': 85,
            'completeness_score': 90,
            'num_missing_fields': 2,
            'anomaly_risk': 'Low',
            'num_selected_questions': 8,
        },
        {
            'name': 'Low Quality Case',
            'quality_score': 35,
            'completeness_score': 40,
            'num_missing_fields': 8,
            'anomaly_risk': 'High',
            'num_selected_questions': 12,
        },
    ]
    
    print("\nPREDICTIONS ON TEST CASES:")
    print("-"*70)
    for test_case in test_cases:
        case_name = test_case.pop('name')
        
        prob = predictor.predict_probability(test_case)
        completion = predictor.predict_completion_rate(test_case, num_questions=8)
        coverage = predictor.predict_field_coverage(test_case, 
                                                    ['event_severity', 'patient_age', 'dosage'])
        
        print(f"\n{case_name}:")
        print(f"  Quality Score: {test_case['quality_score']}")
        print(f"  Response Quality Probability: {prob:.1%}")
        print(f"  Estimated Completion Rate: {completion:.1%}")
        print(f"  Field Coverage: {coverage:.1%}")
