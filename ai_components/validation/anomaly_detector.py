"""
Anomaly detection using Isolation Forest to identify unusual report patterns.
Detects unusual combinations of features (e.g., extreme dosages, rare event combos).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import sys
sys.path.append('../..')
from validation_constants import ANOMALY_THRESHOLDS


class AnomalyDetector:
    """Detect unusual patterns in adverse event reports."""
    
    def __init__(self, contamination=0.1):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected fraction of anomalies in training data
        """
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def prepare_features(self, df, fit=False):
        """
        Prepare features for anomaly detection.
        
        Features include:
        - Numeric: age, dose
        - Encoded categorical: gender, route, event_type, outcome, reporter_type
        - Derived: days_between_start_and_event, report_lag
        """
        df_prepared = df.copy()
        
        # Numeric features
        numeric_features = ['patient_age', 'dose']
        
        # Sanitize numeric features (convert non-numeric to NaN)
        for feature in numeric_features:
            if feature in df_prepared.columns:
                df_prepared[feature] = pd.to_numeric(df_prepared[feature], errors='coerce')
                df_prepared[feature] = df_prepared[feature].fillna(df_prepared[feature].median() or 0)
        
        # Categorical features to encode
        categorical_features = ['patient_gender', 'route', 'event_type', 'outcome', 'reporter_type']
        
        # Encode categorical features
        for feature in categorical_features:
            if feature in df_prepared.columns:
                if fit:
                    self.label_encoders[feature] = LabelEncoder()
                    df_prepared[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(
                        df_prepared[feature].fillna('Unknown')
                    )
                else:
                    # Handle unknown categories by mapping them to the most frequent class
                    classes = set(self.label_encoders[feature].classes_)
                    values = df_prepared[feature].fillna('Unknown')
                    # Replace unknown values with the first known class
                    default_value = self.label_encoders[feature].classes_[0]
                    values = values.apply(lambda x: x if x in classes else default_value)
                    df_prepared[f'{feature}_encoded'] = self.label_encoders[feature].transform(values)
        
        # Create derived features
        if 'start_date' in df_prepared.columns and 'event_date' in df_prepared.columns:
            try:
                df_prepared['days_to_event'] = (
                    pd.to_datetime(df_prepared['event_date']) - 
                    pd.to_datetime(df_prepared['start_date'])
                ).dt.days
                df_prepared['days_to_event'] = df_prepared['days_to_event'].fillna(-1)
            except:
                df_prepared['days_to_event'] = 0
        
        if 'event_date' in df_prepared.columns and 'report_date' in df_prepared.columns:
            try:
                df_prepared['report_lag'] = (
                    pd.to_datetime(df_prepared['report_date']) - 
                    pd.to_datetime(df_prepared['event_date'])
                ).dt.days
                df_prepared['report_lag'] = df_prepared['report_lag'].fillna(0)
            except:
                df_prepared['report_lag'] = 0
        
        # Boolean features
        bool_features = ['hospitalization_flag', 'pregnancy_flag']
        for feature in bool_features:
            if feature in df_prepared.columns:
                df_prepared[feature] = df_prepared[feature].fillna(0).astype(int)
        
        # Select all features for modeling
        feature_cols = (numeric_features + 
                       [f'{f}_encoded' for f in categorical_features] +
                       ['days_to_event', 'report_lag'] +
                       bool_features)
        
        # Keep only columns that exist
        feature_cols = [c for c in feature_cols if c in df_prepared.columns]
        self.feature_names = feature_cols
        
        X = df_prepared[feature_cols].fillna(0)
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, feature_cols
    
    def train(self, df):
        """
        Train Isolation Forest on dataset.
        
        Args:
            df: DataFrame with reports (should contain mostly normal cases)
        """
        print("Training anomaly detection model...")
        
        X, feature_names = self.prepare_features(df, fit=True)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        self.model.fit(X)
        print(f"✓ Model trained on {len(X)} samples")
        print(f"  Features: {', '.join(feature_names[:5])} ... ({len(feature_names)} total)")
        
    def predict(self, df):
        """
        Predict anomalies in dataset.
        
        Args:
            df: DataFrame with reports
            
        Returns:
            DataFrame with anomaly predictions and scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X, _ = self.prepare_features(df, fit=False)
        
        # Get predictions (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(X)
        
        # Get anomaly scores (negative scores = more anomalous)
        anomaly_scores = -self.model.score_samples(X)  # Negate to make positive = anomalous
        
        results = pd.DataFrame({
            'case_id': df['case_id'].values if 'case_id' in df.columns else range(len(df)),
            'is_anomaly': (predictions == -1).astype(int),  # 1 if anomaly, 0 if normal
            'anomaly_score': anomaly_scores,
            'anomaly_risk': pd.cut(anomaly_scores, 
                                   bins=[0, 0.5, 0.7, 1.0],
                                   labels=['Low', 'Medium', 'High'])
        })
        
        return results
    
    def get_top_anomalies(self, results, top_n=10):
        """Get most anomalous reports."""
        return results.nlargest(top_n, 'anomaly_score')
    
    def save_model(self, model_path, scaler_path, encoders_path):
        """Save trained model and preprocessing objects."""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoders, encoders_path)
        print(f"✓ Model saved to {model_path}")
        print(f"✓ Scaler saved to {scaler_path}")
        print(f"✓ Encoders saved to {encoders_path}")
    
    def load_model(self, model_path, scaler_path, encoders_path):
        """Load trained model and preprocessing objects."""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoders = joblib.load(encoders_path)
        print(f"✓ Model loaded from {model_path}")


class CompositeAnomalyDetector:
    """Combine rule-based and statistical anomaly detection."""
    
    def __init__(self):
        self.statistical_detector = AnomalyDetector(contamination=0.1)
    
    def detect_anomalies(self, df, validation_results=None):
        """
        Combine multiple anomaly detection methods.
        
        Args:
            df: Report DataFrame
            validation_results: Results from rule-based validator
            
        Returns:
            DataFrame with combined anomaly scores
        """
        # Statistical anomalies (Isolation Forest)
        stat_results = self.statistical_detector.predict(df)
        
        # Rule-based anomalies (from validation quality score)
        composite_results = stat_results.copy()
        
        if validation_results is not None:
            # Quality score < 40 suggests data anomalies
            composite_results['quality_score'] = validation_results['quality_score'].values
            
            # Combine scores: statistical + quality-based
            quality_anomaly = (1 - validation_results['quality_score'].values / 100).clip(0, 1)
            composite_results['combined_anomaly_score'] = (
                0.6 * composite_results['anomaly_score'] + 
                0.4 * quality_anomaly
            )
        else:
            composite_results['combined_anomaly_score'] = composite_results['anomaly_score']
        
        # Classification based on combined score
        composite_results['anomaly_risk'] = pd.cut(
            composite_results['combined_anomaly_score'],
            bins=[0, 0.4, 0.65, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        return composite_results


if __name__ == "__main__":
    # Example usage
    print("Anomaly Detection Module")
    print("Use in conjunction with rule_validator.py for comprehensive validation")
