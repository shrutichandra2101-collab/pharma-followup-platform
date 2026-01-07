"""
Smart Follow-Up Questionnaire Generator - Visualizer Module
Generate professional visualizations for questionnaire analysis

Step 7: Create 8 publication-quality visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from typing import Optional


class QuestionnaireVisualizer:
    """Generate questionnaire performance visualizations."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize visualizer."""
        if output_dir is None:
            # Calculate absolute path to evaluation/questionnaire_visualizations
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(base_dir, 'evaluation', 'questionnaire_visualizations')
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 7)
        plt.rcParams['font.size'] = 10
    
    def generate_all_visualizations(self, test_df: pd.DataFrame, report: dict):
        """Generate all 8 visualizations."""
        print("\nGenerating questionnaire visualizations...")
        
        self.plot_1_effectiveness_distribution(test_df)
        self.plot_2_coverage_by_profile(test_df)
        self.plot_3_response_quality_distribution(test_df)
        self.plot_4_time_vs_effectiveness(test_df)
        self.plot_5_roi_analysis(test_df)
        self.plot_6_completion_rate_by_status(test_df)
        self.plot_7_field_coverage_heatmap(test_df)
        self.plot_8_satisfaction_metrics(test_df)
        
        print(f"✓ All visualizations saved to {self.output_dir}")
    
    def plot_1_effectiveness_distribution(self, df: pd.DataFrame):
        """Questionnaire Effectiveness Distribution."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Histogram with kde
        ax.hist(df['questionnaire_effectiveness'], bins=30, alpha=0.7, color='#1f77b4', edgecolor='black')
        
        # Add mean and median lines
        mean_val = df['questionnaire_effectiveness'].mean()
        median_val = df['questionnaire_effectiveness'].median()
        
        ax.axvline(mean_val, color='#d62728', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='#2ca02c', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
        
        ax.set_xlabel('Questionnaire Effectiveness Score (0-100)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Cases', fontsize=11, fontweight='bold')
        ax.set_title('Distribution of Questionnaire Effectiveness', fontsize=13, fontweight='bold', pad=20)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '01_effectiveness_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_2_coverage_by_profile(self, df: pd.DataFrame):
        """Field Coverage by Case Profile."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate coverage % for each profile
        coverage_by_profile = df.groupby('profile').apply(
            lambda x: (x['critical_fields_obtained'] / x['num_missing_fields']).mean() * 100
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        bars = ax.bar(range(len(coverage_by_profile)), coverage_by_profile.values, color=colors[:len(coverage_by_profile)])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Case Profile', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Field Coverage (%)', fontsize=11, fontweight='bold')
        ax.set_title('Questionnaire Coverage by Case Profile', fontsize=13, fontweight='bold', pad=20)
        ax.set_xticks(range(len(coverage_by_profile)))
        ax.set_xticklabels(coverage_by_profile.index, rotation=45, ha='right')
        ax.set_ylim([0, 100])
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '02_coverage_by_profile.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_3_response_quality_distribution(self, df: pd.DataFrame):
        """Response Quality Distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Quality score histogram
        ax1.hist(df['avg_response_quality'], bins=25, alpha=0.7, color='#1f77b4', edgecolor='black')
        ax1.axvline(df['avg_response_quality'].mean(), color='#d62728', linestyle='--', linewidth=2,
                   label=f"Mean: {df['avg_response_quality'].mean():.2f}")
        ax1.set_xlabel('Average Response Quality (1-5)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Cases', fontsize=11, fontweight='bold')
        ax1.set_title('Response Quality Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Completion rate
        ax2.hist(df['response_completion_rate'], bins=25, alpha=0.7, color='#2ca02c', edgecolor='black')
        ax2.axvline(df['response_completion_rate'].mean(), color='#d62728', linestyle='--', linewidth=2,
                   label=f"Mean: {df['response_completion_rate'].mean():.1%}")
        ax2.set_xlabel('Question Completion Rate', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Number of Cases', fontsize=11, fontweight='bold')
        ax2.set_title('Response Completion Rate', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '03_response_quality_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_4_time_vs_effectiveness(self, df: pd.DataFrame):
        """Time Estimate vs Actual vs Effectiveness."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Scatter plot
        scatter = ax.scatter(df['actual_completion_time']/60, df['questionnaire_effectiveness'],
                            c=df['response_completion_rate'], cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
        
        ax.set_xlabel('Actual Completion Time (minutes)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Questionnaire Effectiveness (0-100)', fontsize=11, fontweight='bold')
        ax.set_title('Effectiveness vs Time to Complete', fontsize=13, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Completion Rate', fontsize=10, fontweight='bold')
        
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '04_time_vs_effectiveness.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_5_roi_analysis(self, df: pd.DataFrame):
        """ROI (Information Value vs Time Cost) Analysis."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate ROI
        df_copy = df.copy()
        df_copy['info_value'] = df_copy['questionnaire_effectiveness'] * (
            df_copy['critical_fields_obtained'] / (df_copy['num_missing_fields'] + 0.1)
        )
        df_copy['time_cost'] = df_copy['actual_completion_time'] / 900
        df_copy['roi'] = df_copy['info_value'] / (df_copy['time_cost'] + 0.1)
        
        # ROI by validation status
        roi_by_status = df_copy.groupby('validation_status')['roi'].mean().sort_values(ascending=False)
        
        colors = ['#2ca02c' if x > 70 else '#ff7f0e' if x > 50 else '#d62728' for x in roi_by_status.values]
        bars = ax.bar(range(len(roi_by_status)), roi_by_status.values, color=colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Validation Status', fontsize=11, fontweight='bold')
        ax.set_ylabel('ROI Score', fontsize=11, fontweight='bold')
        ax.set_title('Questionnaire ROI by Validation Status', fontsize=13, fontweight='bold', pad=20)
        ax.set_xticks(range(len(roi_by_status)))
        ax.set_xticklabels(roi_by_status.index, rotation=45, ha='right')
        ax.grid(alpha=0.3, axis='y')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ca02c', label='High ROI (>70)'),
            Patch(facecolor='#ff7f0e', label='Medium ROI (50-70)'),
            Patch(facecolor='#d62728', label='Low ROI (<50)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '05_roi_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_6_completion_rate_by_status(self, df: pd.DataFrame):
        """Completion Rate and Satisfaction by Validation Status."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        status_stats = df.groupby('validation_status').agg({
            'response_completion_rate': 'mean',
            'user_satisfaction': 'mean'
        }).reset_index()
        
        x = np.arange(len(status_stats))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, status_stats['response_completion_rate'], width,
                      label='Completion Rate', color='#1f77b4', edgecolor='black')
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, status_stats['user_satisfaction'], width,
                       label='Satisfaction', color='#2ca02c', edgecolor='black')
        
        ax.set_xlabel('Validation Status', fontsize=11, fontweight='bold')
        ax.set_ylabel('Completion Rate', fontsize=11, fontweight='bold', color='#1f77b4')
        ax2.set_ylabel('Satisfaction (1-5)', fontsize=11, fontweight='bold', color='#2ca02c')
        ax.set_title('Completion Rate & Satisfaction by Validation Status', fontsize=13, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(status_stats['validation_status'])
        ax.tick_params(axis='y', labelcolor='#1f77b4')
        ax2.tick_params(axis='y', labelcolor='#2ca02c')
        ax.grid(alpha=0.3, axis='y')
        
        # Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '06_completion_by_status.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_7_field_coverage_heatmap(self, df: pd.DataFrame):
        """Field Coverage Heatmap by Profile and Status."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Create pivot table
        pivot_data = df.groupby(['profile', 'validation_status']).apply(
            lambda x: (x['critical_fields_obtained'] / x['num_missing_fields']).mean() * 100
        ).unstack(fill_value=0)
        
        # Heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
                   cbar_kws={'label': 'Coverage %'}, ax=ax, linewidths=1, linecolor='gray')
        
        ax.set_xlabel('Validation Status', fontsize=11, fontweight='bold')
        ax.set_ylabel('Case Profile', fontsize=11, fontweight='bold')
        ax.set_title('Field Coverage Heatmap', fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '07_field_coverage_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_8_satisfaction_metrics(self, df: pd.DataFrame):
        """User Satisfaction and Response Quality."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Box plot of satisfaction by profile
        profiles = sorted(df['profile'].unique())
        satisfaction_data = [df[df['profile'] == p]['user_satisfaction'].values for p in profiles]
        
        bp = ax.boxplot(satisfaction_data, labels=profiles, patch_artist=True,
                       medianprops=dict(color='#d62728', linewidth=2),
                       boxprops=dict(facecolor='#1f77b4', alpha=0.7),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        ax.set_xlabel('Case Profile', fontsize=11, fontweight='bold')
        ax.set_ylabel('User Satisfaction (1-5)', fontsize=11, fontweight='bold')
        ax.set_title('User Satisfaction Distribution by Case Profile', fontsize=13, fontweight='bold', pad=20)
        ax.set_ylim([0, 5.5])
        ax.grid(alpha=0.3, axis='y')
        
        # Add mean line
        means = [df[df['profile'] == p]['user_satisfaction'].mean() for p in profiles]
        ax.plot(range(1, len(profiles)+1), means, 'o-', color='#ff7f0e', linewidth=2, markersize=8, label='Mean')
        ax.legend()
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '08_satisfaction_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    from data_generator import QuestionnaireDataGenerator
    from evaluation_metrics import generate_evaluation_report
    
    print("Generating test data...")
    generator = QuestionnaireDataGenerator(num_samples=1000)
    test_df = generator.generate_dataset()
    
    print("\nGenerating report...")
    report = generate_evaluation_report(test_df)
    
    print("Generating visualizations...")
    visualizer = QuestionnaireVisualizer()
    visualizer.generate_all_visualizations(test_df, report)
    
    print("\n✓ Visualization generation complete!")
