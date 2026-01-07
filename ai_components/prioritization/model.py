"""Follow-up Prioritization Model using XGBoost."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import sys
sys.path.append('../..')
from utils.common import (save_metrics, plot_feature_importance, 
                         plot_confusion_matrix, print_classification_report)
from visualize_results import PrioritizationVisualizer

class PrioritizationModel:
    """XGBoost model for case prioritization."""
    
    def __init__(self):
        self.regression_model = None
        self.classification_model = None
        self.label_encoders = {}
        self.feature_names = None
        
    def prepare_features(self, df, fit_encoders=False):
        """Prepare features for model training."""
        df = df.copy()
        
        # Categorical features to encode
        categorical_cols = ['reporter_type', 'region', 'event_type', 'seriousness_type']
        
        for col in categorical_cols:
            if fit_encoders:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        # Select features for modeling
        feature_cols = [
            'is_serious', 'seriousness_score', 'completeness_pct',
            'days_since_report', 'days_to_deadline',
            'reporter_reliability', 'regulatory_strictness',
            'num_followup_attempts', 'historical_response_rate',
            'reporter_type_encoded', 'region_encoded', 
            'event_type_encoded', 'seriousness_type_encoded'
        ]
        
        X = df[feature_cols]
        self.feature_names = feature_cols
        
        return X
    
    def train_regression(self, X_train, y_train, X_val, y_val):
        """Train XGBoost regression model for priority scores."""
        print("\n" + "="*60)
        print("TRAINING REGRESSION MODEL (Priority Score Prediction)")
        print("="*60 + "\n")
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'eval_metric': 'rmse'
        }
        
        # Train with early stopping
        evals = [(dtrain, 'train'), (dval, 'val')]
        evals_result = {}
        
        self.regression_model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=20,
            verbose_eval=20
        )
        
        return evals_result
    
    def train_classification(self, X_train, y_train, X_val, y_val):
        """Train XGBoost classification model for priority categories."""
        print("\n" + "="*60)
        print("TRAINING CLASSIFICATION MODEL (Priority Category)")
        print("="*60 + "\n")
        
        # Encode labels
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_val_encoded = le.transform(y_val)
        self.category_encoder = le
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
        dval = xgb.DMatrix(X_val, label=y_val_encoded)
        
        # Parameters
        params = {
            'objective': 'multi:softmax',
            'num_class': len(le.classes_),
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss'
        }
        
        # Train
        evals = [(dtrain, 'train'), (dval, 'val')]
        evals_result = {}
        
        self.classification_model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=20,
            verbose_eval=20
        )
        
        return evals_result
    
    def predict_score(self, X):
        """Predict priority scores."""
        dtest = xgb.DMatrix(X)
        return self.regression_model.predict(dtest)
    
    def predict_category(self, X):
        """Predict priority categories."""
        dtest = xgb.DMatrix(X)
        predictions_encoded = self.classification_model.predict(dtest)
        return self.category_encoder.inverse_transform(predictions_encoded.astype(int))
    
    def save_models(self, reg_path, clf_path, encoders_path):
        """Save models and encoders."""
        self.regression_model.save_model(reg_path)
        self.classification_model.save_model(clf_path)
        joblib.dump({
            'label_encoders': self.label_encoders,
            'category_encoder': self.category_encoder,
            'feature_names': self.feature_names
        }, encoders_path)
        print(f"\nModels saved successfully!")

def evaluate_regression(y_true, y_pred, save_path=None):
    """Evaluate regression model."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("\nRegression Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', lw=2)
    axes[0].set_xlabel('Actual Priority Score')
    axes[0].set_ylabel('Predicted Priority Score')
    axes[0].set_title(f'Actual vs Predicted (R²={r2:.3f})')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Priority Score')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Regression plots saved to {save_path}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def evaluate_classification(y_true, y_pred, labels, save_path=None):
    """Evaluate classification model."""
    print_classification_report(y_true, y_pred, labels)
    
    # Confusion matrix
    cm_path = save_path.replace('.png', '_confusion_matrix.png') if save_path else None
    plot_confusion_matrix(y_true, y_pred, labels, 
                         title="Priority Category Confusion Matrix",
                         save_path=cm_path)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred)
    }

if __name__ == "__main__":
    print("Loading datasets...")
    train_df = pd.read_csv('../../data/processed/prioritization_train.csv')
    test_df = pd.read_csv('../../data/processed/prioritization_test.csv')
    
    # Initialize model
    model = PrioritizationModel()
    
    # Prepare features
    X_train_full = model.prepare_features(train_df, fit_encoders=True)
    X_test = model.prepare_features(test_df, fit_encoders=False)
    
    y_train_score = train_df['priority_score']
    y_train_category = train_df['priority_category']
    y_test_score = test_df['priority_score']
    y_test_category = test_df['priority_category']
    
    # Split training data for validation
    X_train, X_val, y_train_s, y_val_s, y_train_c, y_val_c = train_test_split(
        X_train_full, y_train_score, y_train_category,
        test_size=0.2, random_state=42
    )
    
    # Train regression model
    reg_history = model.train_regression(X_train, y_train_s, X_val, y_val_s)
    
    # Train classification model
    clf_history = model.train_classification(X_train, y_train_c, X_val, y_val_c)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("TEST SET EVALUATION - REGRESSION")
    print("="*60)
    y_pred_score = model.predict_score(X_test)
    reg_metrics = evaluate_regression(
        y_test_score.values, 
        y_pred_score,
        save_path='../../evaluation/prioritization_regression.png'
    )
    
    print("\n" + "="*60)
    print("TEST SET EVALUATION - CLASSIFICATION")
    print("="*60)
    y_pred_category = model.predict_category(X_test)
    clf_metrics = evaluate_classification(
        y_test_category,
        y_pred_category,
        labels=sorted(y_test_category.unique()),
        save_path='../../evaluation/prioritization_classification.png'
    )
    
    # Feature importance - use permutation importance as fallback
    from sklearn.inspection import permutation_importance as sklearn_permutation_importance
    
    feature_names = model.feature_names
    
    # First try XGBoost's built-in importance
    importance_dict = model.regression_model.get_score(importance_type='gain')
    importance_values = np.array([importance_dict.get(f'f{i}', 0) for i in range(len(feature_names))])
    
    # If XGBoost importance is empty, use permutation importance
    if importance_values.sum() == 0:
        print("Using permutation importance (XGBoost built-in was empty)...")
        # Use sklearn's permutation importance
        dtest = xgb.DMatrix(X_test)
        y_pred_baseline = model.regression_model.predict(dtest)
        
        importances = []
        for i in range(len(feature_names)):
            X_test_shuffled = X_test.copy()
            # Shuffle feature i
            X_test_shuffled.iloc[:, i] = np.random.permutation(X_test.iloc[:, i].values)
            dtest_shuffled = xgb.DMatrix(X_test_shuffled)
            y_pred_shuffled = model.regression_model.predict(dtest_shuffled)
            
            # Calculate importance as decrease in performance
            mse_baseline = mean_squared_error(y_test_score.values, y_pred_baseline)
            mse_shuffled = mean_squared_error(y_test_score.values, y_pred_shuffled)
            importance = mse_shuffled - mse_baseline
            importances.append(max(0, importance))  # Importance can't be negative
        
        importance_values = np.array(importances)
    
    # Normalize to sum to 100
    if importance_values.sum() > 0:
        importance_values = importance_values / importance_values.sum() * 100
    else:
        importance_values = np.ones(len(feature_names)) / len(feature_names) * 100
    
    print(f"Feature Importance (normalized %): {importance_values}")
    print(f"Top 3 features: {sorted(zip(feature_names, importance_values), key=lambda x: x[1], reverse=True)[:3]}")
    
    plot_feature_importance(
        feature_names,
        importance_values,
        title="Feature Importance - Prioritization Model",
        save_path='../../evaluation/prioritization_feature_importance.png'
    )
    
    # Save models
    model.save_models(
        '../../data/models/prioritization_regression.json',
        '../../data/models/prioritization_classification.json',
        '../../data/models/prioritization_encoders.pkl'
    )
    
    # Save metrics
    all_metrics = {
        'regression': reg_metrics,
        'classification': clf_metrics,
        'feature_importance': {
            name: float(val) for name, val in zip(feature_names, importance_values)
        }
    }
    save_metrics(all_metrics, '../../evaluation/prioritization_metrics.json')
    
    # Generate comprehensive visualizations
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60 + "\n")
    
    visualizer = PrioritizationVisualizer()
    
    # Generate detailed metrics tables
    visualizer.create_metrics_table(
        y_test_score.values, y_pred_score, y_test_category, y_pred_category,
        categories=sorted(y_test_category.unique()),
        save_path='../../evaluation/prioritization_metrics_table.png'
    )
    
    visualizer.create_classification_metrics_table(
        y_test_category, y_pred_category,
        categories=sorted(y_test_category.unique()),
        save_path='../../evaluation/prioritization_per_category_metrics.png'
    )
    
    # Generate distribution plots
    visualizer.create_prediction_distribution_plot(
        y_test_score.values, y_pred_score,
        save_path='../../evaluation/prioritization_prediction_distribution.png'
    )
    
    # Generate error analysis
    visualizer.create_error_analysis_plot(
        y_test_score.values, y_pred_score,
        save_path='../../evaluation/prioritization_error_analysis.png'
    )
    
    # Generate category distribution
    visualizer.create_category_distribution_plot(
        y_test_category, y_pred_category,
        categories=sorted(y_test_category.unique()),
        save_path='../../evaluation/prioritization_category_distribution.png'
    )
    
    # Generate calibration plot
    visualizer.create_prediction_calibration_plot(
        y_test_score.values, y_pred_score,
        save_path='../../evaluation/prioritization_calibration.png'
    )
    
    # Generate feature importance heatmap
    feature_importance_dict = {
        name: float(val) for name, val in zip(feature_names, importance_values)
    }
    visualizer.create_feature_correlation_heatmap(
        feature_importance_dict,
        save_path='../../evaluation/prioritization_top_features.png'
    )
    
    # Generate summary report
    visualizer.create_summary_report(
        save_path='../../evaluation/PRIORITIZATION_PERFORMANCE_REPORT.txt'
    )
    
    print("\n" + "="*60)
    print("PRIORITIZATION MODEL TRAINING COMPLETE!")
    print("="*60)
