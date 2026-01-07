"""
Visualization module for validation results and performance metrics.
Generates comprehensive plots for error analysis and anomaly detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


class ValidationVisualizer:
    """Create visualizations for validation results."""
    
    def __init__(self, output_dir=None):
        if output_dir is None:
            # Use absolute path
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(base_dir, 'evaluation', 'validation_visualizations')
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_error_distribution(self, validation_results, original_df):
        """
        Plot distribution of error counts.
        
        Args:
            validation_results: DataFrame with validation results
            original_df: Original data with ground truth
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # All reports error count
        ax = axes[0]
        error_counts = validation_results['error_count']
        ax.hist(error_counts, bins=20, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Number of Errors Found', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Error Count Distribution (All Reports)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Split by ground truth
        ax = axes[1]
        with_errors = original_df['has_errors'] == 1
        without_errors = original_df['has_errors'] == 0
        
        ax.hist(
            validation_results.loc[with_errors, 'error_count'],
            bins=20, alpha=0.6, label='Reports with Actual Errors', color='red'
        )
        ax.hist(
            validation_results.loc[without_errors, 'error_count'],
            bins=20, alpha=0.6, label='Clean Reports', color='green'
        )
        ax.set_xlabel('Number of Errors Found', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Error Detection: Actual vs Detected', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_error_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 01_error_distribution.png")
        plt.close()
    
    def plot_quality_score_distribution(self, validation_results):
        """
        Plot quality score distribution.
        
        Args:
            validation_results: DataFrame with validation results
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        quality_scores = validation_results['quality_score']
        
        ax.hist(quality_scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(quality_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {quality_scores.mean():.2f}')
        ax.axvline(quality_scores.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {quality_scores.median():.2f}')
        
        # Add interpretation zones
        ax.axvspan(0, 20, alpha=0.1, color='red', label='Critical')
        ax.axvspan(20, 40, alpha=0.1, color='orange', label='Poor')
        ax.axvspan(40, 60, alpha=0.1, color='yellow', label='Fair')
        ax.axvspan(60, 80, alpha=0.1, color='lightgreen', label='Good')
        ax.axvspan(80, 100, alpha=0.1, color='green', label='Excellent')
        
        ax.set_xlabel('Quality Score', fontsize=11)
        ax.set_ylabel('Number of Reports', fontsize=11)
        ax.set_title('Quality Score Distribution', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_quality_score_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 02_quality_score_distribution.png")
        plt.close()
    
    def plot_anomaly_score_distribution(self, combined_results):
        """
        Plot anomaly score distribution.
        
        Args:
            combined_results: DataFrame with combined results
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overall distribution
        ax = axes[0]
        anomaly_scores = combined_results['anomaly_score']
        ax.hist(anomaly_scores, bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
        ax.axvline(anomaly_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {anomaly_scores.mean():.3f}')
        ax.set_xlabel('Anomaly Score', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Anomaly Score Distribution (All Reports)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # By risk level
        ax = axes[1]
        risk_distribution = combined_results['anomaly_risk'].value_counts()
        colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        bar_colors = [colors.get(x, 'gray') for x in risk_distribution.index]
        
        ax.bar(risk_distribution.index, risk_distribution.values, color=bar_colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Anomaly Risk Level Distribution', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        # Add counts on bars
        for i, v in enumerate(risk_distribution.values):
            ax.text(i, v + 10, str(v), ha='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_anomaly_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 03_anomaly_distribution.png")
        plt.close()
    
    def plot_overall_status_distribution(self, combined_results):
        """
        Plot distribution of overall validation status.
        
        Args:
            combined_results: DataFrame with combined results
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        status_counts = combined_results['overall_status'].value_counts()
        colors = {
            'ACCEPT': 'green',
            'CONDITIONAL_ACCEPT': 'yellow',
            'REVIEW': 'orange',
            'REJECT': 'red'
        }
        bar_colors = [colors.get(x, 'gray') for x in status_counts.index]
        
        bars = ax.bar(status_counts.index, status_counts.values, color=bar_colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Number of Reports', fontsize=11)
        ax.set_title('Overall Validation Status Distribution', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        # Add counts and percentages
        total = len(combined_results)
        for i, v in enumerate(status_counts.values):
            pct = v / total * 100
            ax.text(i, v + 5, f'{v}\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_overall_status_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 04_overall_status_distribution.png")
        plt.close()
    
    def plot_quality_vs_anomaly_score(self, combined_results, original_df):
        """
        Scatter plot of quality score vs anomaly score.
        
        Args:
            combined_results: DataFrame with combined results
            original_df: Original data with ground truth
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Color by ground truth (has errors)
        colors = original_df['has_errors'].map({0: 'green', 1: 'red'})
        
        ax.scatter(
            combined_results['quality_score'],
            combined_results['anomaly_score'],
            c=colors, alpha=0.5, s=50
        )
        
        ax.set_xlabel('Quality Score (0-100)', fontsize=11)
        ax.set_ylabel('Anomaly Score (0-1)', fontsize=11)
        ax.set_title('Quality Score vs Anomaly Score', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.5, label='Clean Reports'),
            Patch(facecolor='red', alpha=0.5, label='Reports with Errors')
        ]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_quality_vs_anomaly.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 05_quality_vs_anomaly.png")
        plt.close()
    
    def plot_error_types_heatmap(self, validation_results, original_df):
        """
        Heatmap of error types vs report validity.
        
        Args:
            validation_results: DataFrame with validation results
            original_df: Original data with error_types
        """
        if 'error_types' not in original_df.columns:
            print("⚠ Skipping error types heatmap (error_types column not found)")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Count error types by ground truth
        error_type_counts = pd.DataFrame({
            'is_valid': validation_results['is_valid'],
            'error_types': original_df['error_types']
        })
        
        # This would need special processing of error_types
        # For now, just show detection accuracy
        ax.text(0.5, 0.5, 'Error types analysis requires structured error type data', 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/06_error_types.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 06_error_types.png (placeholder)")
        plt.close()
    
    def plot_performance_metrics_summary(self, metrics_dict):
        """
        Plot key performance metrics.
        
        Args:
            metrics_dict: Dictionary with validation and anomaly metrics
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Validation metrics
        ax = axes[0]
        val_metrics = metrics_dict.get('validation_metrics', {})
        metric_names = ['precision', 'recall', 'f1', 'accuracy']
        metric_values = [val_metrics.get(m, 0) for m in metric_names]
        
        bars = ax.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7, edgecolor='black')
        ax.set_ylim([0, 1])
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Validation Metrics', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Anomaly metrics
        ax = axes[1]
        anom_metrics = metrics_dict.get('anomaly_metrics', {})
        metric_names = ['precision', 'recall', 'f1', 'auc_roc']
        metric_values = [anom_metrics.get(m, 0) for m in metric_names]
        
        bars = ax.bar(metric_names, metric_values, color=['#e377c2', '#7f7f7f', '#bcbd22', '#17becf'], alpha=0.7, edgecolor='black')
        ax.set_ylim([0, 1])
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Anomaly Detection Metrics', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/07_metrics_summary.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 07_metrics_summary.png")
        plt.close()
    
    def generate_all_visualizations(self, validation_results, anomaly_results, combined_results, 
                                   original_df, metrics_dict):
        """
        Generate all visualizations.
        
        Args:
            validation_results: Rule-based validation results
            anomaly_results: Anomaly detection results
            combined_results: Combined results
            original_df: Original data with ground truth
            metrics_dict: Performance metrics dictionary
        """
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70 + "\n")
        
        self.plot_error_distribution(validation_results, original_df)
        self.plot_quality_score_distribution(validation_results)
        self.plot_anomaly_score_distribution(combined_results)
        self.plot_overall_status_distribution(combined_results)
        self.plot_quality_vs_anomaly_score(combined_results, original_df)
        self.plot_error_types_heatmap(validation_results, original_df)
        self.plot_performance_metrics_summary(metrics_dict)
        
        print(f"\n✓ All visualizations saved to {self.output_dir}/")


if __name__ == "__main__":
    print("Visualization Module")
    print("Creates comprehensive plots for validation and anomaly detection results")
