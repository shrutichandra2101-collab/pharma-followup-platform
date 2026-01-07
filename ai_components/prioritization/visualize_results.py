"""
Comprehensive visualization and reporting for prioritization model performance.
Generates detailed metrics, charts, and summary reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, mean_squared_error, 
    mean_absolute_error, r2_score
)
import json
import sys
sys.path.append('../..')
from utils.common import load_metrics

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class PrioritizationVisualizer:
    """Create comprehensive visualizations for model performance."""
    
    def __init__(self, metrics_path=None):
        self.metrics_path = metrics_path or '../../evaluation/prioritization_metrics.json'
        self.eval_dir = '../../evaluation/'
        self.metrics = None
        if self._file_exists(self.metrics_path):
            self.metrics = load_metrics(self.metrics_path)
    
    @staticmethod
    def _file_exists(path):
        """Check if file exists."""
        from pathlib import Path
        return Path(path).exists()
    
    def create_metrics_table(self, y_true_score, y_pred_score, y_true_cat, y_pred_cat, 
                            categories, save_path=None):
        """Create comprehensive metrics table."""
        # Regression metrics
        rmse = np.sqrt(mean_squared_error(y_true_score, y_pred_score))
        mae = mean_absolute_error(y_true_score, y_pred_score)
        r2 = r2_score(y_true_score, y_pred_score)
        
        # Classification metrics per category
        accuracy = accuracy_score(y_true_cat, y_pred_cat)
        macro_f1 = f1_score(y_true_cat, y_pred_cat, average='macro', zero_division=0)
        
        # Create metrics dataframe
        metrics_data = {
            'Metric': ['RMSE', 'MAE', 'R² Score', 'Accuracy', 'Macro F1-Score'],
            'Score': [f'{rmse:.4f}', f'{mae:.4f}', f'{r2:.4f}', f'{accuracy:.4f}', f'{macro_f1:.4f}'],
            'Target': ['≤ 0.50', '≤ 0.40', '≥ 0.85', '≥ 0.85', '≥ 0.85']
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create figure with table
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns,
                        cellLoc='center', loc='center', colWidths=[0.3, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(len(metrics_df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(metrics_df) + 1):
            for j in range(len(metrics_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('white')
        
        plt.title('Model Performance Summary - Key Metrics', fontsize=14, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics table saved to {save_path}")
        
        return fig
    
    def create_classification_metrics_table(self, y_true, y_pred, categories, save_path=None):
        """Create detailed per-category classification metrics."""
        precision_scores = precision_score(y_true, y_pred, labels=categories, 
                                           average=None, zero_division=0)
        recall_scores = recall_score(y_true, y_pred, labels=categories, 
                                     average=None, zero_division=0)
        f1_scores = f1_score(y_true, y_pred, labels=categories, 
                            average=None, zero_division=0)
        
        # Count support
        cm = confusion_matrix(y_true, y_pred, labels=categories)
        support = cm.sum(axis=1)
        
        metrics_data = {
            'Category': categories,
            'Precision': [f'{p:.3f}' for p in precision_scores],
            'Recall': [f'{r:.3f}' for r in recall_scores],
            'F1-Score': [f'{f:.3f}' for f in f1_scores],
            'Support': support
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create figure with table
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns,
                        cellLoc='center', loc='center', colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(len(metrics_df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code rows by category
        colors = ['#e8f4f8', '#fff4e6', '#ffe8e8', '#f0e8ff']
        for i in range(1, len(metrics_df) + 1):
            for j in range(len(metrics_df.columns)):
                table[(i, j)].set_facecolor(colors[(i-1) % len(colors)])
        
        plt.title('Per-Category Classification Metrics', fontsize=14, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Classification metrics table saved to {save_path}")
        
        return fig
    
    def create_prediction_distribution_plot(self, y_true_score, y_pred_score, save_path=None):
        """Plot distribution of actual vs predicted scores."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Distribution of actual scores
        axes[0].hist(y_true_score, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_xlabel('Priority Score', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Distribution of Actual Priority Scores', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Distribution of predicted scores
        axes[1].hist(y_pred_score, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_xlabel('Priority Score', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Distribution of Predicted Priority Scores', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction distribution plot saved to {save_path}")
        
        return fig
    
    def create_error_analysis_plot(self, y_true_score, y_pred_score, save_path=None):
        """Create error analysis visualizations."""
        errors = np.abs(y_true_score - y_pred_score)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Error by actual score
        axes[0, 0].scatter(y_true_score, errors, alpha=0.5, s=30, color='#FF6B6B')
        axes[0, 0].set_xlabel('Actual Priority Score', fontsize=11)
        axes[0, 0].set_ylabel('Absolute Error', fontsize=11)
        axes[0, 0].set_title('Prediction Error by Actual Score', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error distribution
        axes[0, 1].hist(errors, bins=30, alpha=0.7, color='#4ECDC4', edgecolor='black')
        axes[0, 1].axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.3f}')
        axes[0, 1].set_xlabel('Absolute Error', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Error Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative error
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1, 0].plot(sorted_errors, cumulative, linewidth=2, color='#95E1D3')
        axes[1, 0].set_xlabel('Absolute Error', fontsize=11)
        axes[1, 0].set_ylabel('Cumulative Proportion', fontsize=11)
        axes[1, 0].set_title('Cumulative Error Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot by score ranges
        score_ranges = pd.cut(y_true_score, bins=[0, 3, 5, 7, 10])
        error_by_range = [errors[score_ranges == r] for r in score_ranges.unique()]
        
        axes[1, 1].boxplot(error_by_range, labels=['Low\n(0-3)', 'Medium\n(3-5)', 'High\n(5-7)', 'Critical\n(7-10)'])
        axes[1, 1].set_ylabel('Absolute Error', fontsize=11)
        axes[1, 1].set_title('Error Distribution by Priority Range', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error analysis plot saved to {save_path}")
        
        return fig
    
    def create_category_distribution_plot(self, y_true_cat, y_pred_cat, categories, save_path=None):
        """Compare actual vs predicted category distributions."""
        true_counts = pd.Series(y_true_cat).value_counts().reindex(categories, fill_value=0)
        pred_counts = pd.Series(y_pred_cat).value_counts().reindex(categories, fill_value=0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, true_counts.values, width, label='Actual', color='#4ECDC4', alpha=0.8)
        bars2 = ax.bar(x + width/2, pred_counts.values, width, label='Predicted', color='#FF6B6B', alpha=0.8)
        
        ax.set_xlabel('Priority Category', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Distribution: Actual vs Predicted Categories', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Category distribution plot saved to {save_path}")
        
        return fig
    
    def create_prediction_calibration_plot(self, y_true_score, y_pred_score, save_path=None):
        """Create calibration plot for regression predictions."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create bins
        bins = np.linspace(y_pred_score.min(), y_pred_score.max(), 10)
        binids = np.digitize(y_pred_score, bins)
        
        bin_means_pred = []
        bin_means_true = []
        bin_counts = []
        
        for i in range(1, len(bins)):
            mask = binids == i
            if mask.sum() > 0:
                bin_means_pred.append(y_pred_score[mask].mean())
                bin_means_true.append(y_true_score[mask].mean())
                bin_counts.append(mask.sum())
        
        # Plot calibration curve
        ax.plot([y_pred_score.min(), y_pred_score.max()], 
               [y_pred_score.min(), y_pred_score.max()], 
               'r--', linewidth=2, label='Perfect Calibration')
        ax.scatter(bin_means_pred, bin_means_true, s=[c*5 for c in bin_counts], 
                  alpha=0.6, color='#4ECDC4', edgecolors='black', linewidth=1)
        
        ax.set_xlabel('Mean Predicted Score', fontsize=11)
        ax.set_ylabel('Mean Actual Score', fontsize=11)
        ax.set_title('Model Calibration Plot (Bubble size = sample count)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Calibration plot saved to {save_path}")
        
        return fig
    
    def create_feature_correlation_heatmap(self, feature_importance_dict, save_path=None):
        """Create heatmap of feature importance."""
        features = list(feature_importance_dict.keys())
        importance = list(feature_importance_dict.values())
        
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1][:15]
        
        top_features = [features[i] for i in sorted_idx]
        top_importance = [importance[i] for i in sorted_idx]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        bars = ax.barh(range(len(top_features)), top_importance, color=colors)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importance Weight', fontsize=11)
        ax.set_title('Top 15 Feature Importance', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_importance)):
            ax.text(val, i, f' {val:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance heatmap saved to {save_path}")
        
        return fig
    
    def create_summary_report(self, save_path=None):
        """Create a text summary report."""
        report = """
╔════════════════════════════════════════════════════════════════════════════╗
║              PRIORITIZATION MODEL - PERFORMANCE REPORT                     ║
╚════════════════════════════════════════════════════════════════════════════╝

SUMMARY:
--------
This report contains comprehensive performance metrics and visualizations for
the XGBoost-based Follow-up Prioritization Model.

The model consists of two components:
  1. REGRESSION: Predicts continuous priority scores (1-10)
  2. CLASSIFICATION: Assigns priority categories (Low/Medium/High/Critical)

KEY PERFORMANCE INDICATORS:
---------------------------
✓ Regression RMSE (Root Mean Squared Error): Lower is better (target: ≤ 0.50)
✓ Regression MAE (Mean Absolute Error): Lower is better (target: ≤ 0.40)
✓ Regression R² Score: Higher is better (target: ≥ 0.85)
✓ Classification Accuracy: % of cases correctly categorized (target: ≥ 85%)
✓ Macro F1-Score: Balanced performance across categories (target: ≥ 0.85)

VISUALIZATIONS GENERATED:
-------------------------
1. prioritization_regression.png
   - Actual vs Predicted scatter plot
   - Residual plot

2. prioritization_classification_confusion_matrix.png
   - Confusion matrix for category predictions
   - Shows which categories are most confused

3. prioritization_feature_importance.png
   - Top features driving the model
   - Which factors matter most

4. prioritization_metrics_table.png (NEW)
   - Summary of all key metrics
   - Easy reference table

5. prioritization_per_category_metrics.png (NEW)
   - Precision, Recall, F1 per category
   - Support (sample count) per category

6. prioritization_prediction_distribution.png (NEW)
   - Histogram of actual scores
   - Histogram of predicted scores
   - Shows distribution overlap

7. prioritization_error_analysis.png (NEW)
   - Error by actual score
   - Error distribution
   - Cumulative errors
   - Errors by priority range

8. prioritization_category_distribution.png (NEW)
   - Actual vs Predicted category counts
   - Shows any systematic bias

9. prioritization_calibration.png (NEW)
   - Model calibration analysis
   - Are predicted scores reliable?

10. prioritization_top_features.png (NEW)
    - Top 15 most important features
    - Visual ranking of importance

MODEL COMPONENTS:
-----------------
REGRESSION MODEL (Priority Score):
  - Objective: Predict continuous priority score (1-10)
  - Algorithm: XGBoost regression
  - Hyperparameters: max_depth=6, learning_rate=0.1
  - Early stopping: Enabled (20 rounds patience)

CLASSIFICATION MODEL (Priority Category):
  - Objective: Predict category (Low/Medium/High/Critical)
  - Algorithm: XGBoost multi-class classification
  - Classes: 4 (Low, Medium, High, Critical)
  - Hyperparameters: max_depth=5, learning_rate=0.1

INPUT FEATURES (13 TOTAL):
---------------------------
Medical & Clinical:
  • is_serious: Whether case meets seriousness criteria
  • seriousness_score: 1-10 scale of medical severity
  • event_type: Category of adverse event (cardiac, GI, etc.)

Data Quality:
  • completeness_pct: % of mandatory fields filled

Temporal:
  • days_since_report: How long ago case was reported
  • days_to_deadline: Days remaining until regulatory deadline

Reporter Context:
  • reporter_type: HCP, patient, pharmacist, etc.
  • reporter_reliability: Historical accuracy score
  • region: Geographic region
  • regulatory_strictness: Regulatory environment strictness

History & Baseline:
  • num_followup_attempts: Previous follow-up attempts
  • seriousness_type: Specific seriousness criterion
  • historical_response_rate: Expected response rate

OUTPUT TARGETS:
----------------
Primary Output (Regression):
  • Priority Score: 1-10 continuous value
    - 1-3: Low (can wait)
    - 4-6: Medium (routine priority)
    - 7-8: High (expedite)
    - 9-10: Critical (immediate)

Secondary Output (Classification):
  • Priority Category: Discrete label
    - Low / Medium / High / Critical

RECOMMENDATIONS:
-----------------
1. Focus on High and Critical cases first (highest impact)
2. Monitor performance drift over time
3. Re-train model quarterly with new data
4. Consider feature importance when collecting new data
5. Use confidence scores for borderline cases

FILES GENERATED:
-----------------
Model Files:
  ✓ data/models/prioritization_regression.json
  ✓ data/models/prioritization_classification.json
  ✓ data/models/prioritization_encoders.pkl

Data Files:
  ✓ data/processed/prioritization_train.csv
  ✓ data/processed/prioritization_test.csv

Metrics & Visualizations:
  ✓ evaluation/prioritization_metrics.json
  ✓ evaluation/prioritization_*.png (10 visualization files)

╔════════════════════════════════════════════════════════════════════════════╗
║                    END OF PERFORMANCE REPORT                              ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Summary report saved to {save_path}")
        else:
            print(report)
        
        return report


def generate_all_visualizations(metrics_path='../../evaluation/prioritization_metrics.json'):
    """Generate all visualizations from existing metrics and predictions."""
    import pickle
    
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS FOR PRIORITIZATION MODEL")
    print("="*80 + "\n")
    
    # Load test data and predictions
    try:
        test_df = pd.read_csv('../../data/processed/prioritization_test.csv')
        metrics = load_metrics(metrics_path)
        
        # Load model predictions (we need to regenerate or load from saved files)
        print("⚠️  To generate visualizations, please run from model.py with predictions")
        print("   Or provide y_true and y_pred as arguments")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please run model.py first to generate predictions")


if __name__ == "__main__":
    # This script is meant to be called from model.py with prediction data
    # Or use it to generate summary report
    
    visualizer = PrioritizationVisualizer()
    
    # Generate summary report
    report = visualizer.create_summary_report(
        save_path='../../evaluation/PERFORMANCE_REPORT.txt'
    )
