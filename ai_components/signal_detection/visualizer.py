"""
Geospatial Signal Detection - Visualization Engine
Create 8 professional visualizations at 300 DPI

Step 5: Implement visualizations
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rcParams
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for 300 DPI output
rcParams['figure.dpi'] = 100
rcParams['savefig.dpi'] = 300
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 10
rcParams['legend.fontsize'] = 9

# Set color palette
sns.set_palette("husl")


class SignalDetectionVisualizer:
    """Create professional visualizations for signal detection results."""
    
    def __init__(self, output_dir: str = 'signal_detection_visualizations'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def viz_1_geographic_clusters(self, df: pd.DataFrame) -> str:
        """
        Visualization 1: Geographic Cluster Distribution
        Scatter plot of all cases colored by cluster assignment
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot non-noise points
        clustered = df[df['cluster_id'] != -1]
        noise = df[df['cluster_id'] == -1]
        
        scatter = ax.scatter(
            clustered['longitude'],
            clustered['latitude'],
            c=clustered['cluster_id'],
            s=50,
            alpha=0.6,
            cmap='tab20',
            edgecolors='black',
            linewidth=0.5,
            label='Clustered cases'
        )
        
        # Plot noise points
        ax.scatter(
            noise['longitude'],
            noise['latitude'],
            c='gray',
            s=30,
            alpha=0.3,
            marker='x',
            linewidth=1.5,
            label='Noise/Outliers'
        )
        
        ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')
        ax.set_title('Geospatial Cluster Distribution - Adverse Events by Location',
                     fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax, label='Cluster ID')
        
        plt.tight_layout()
        path = f'{self.output_dir}/01_geographic_clusters.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return path
    
    def viz_2_batch_risk_distribution(self, batch_scores_df: pd.DataFrame) -> str:
        """
        Visualization 2: Batch Risk Score Distribution
        Histogram of risk scores with alert level colors
        """
        fig, ax = plt.subplots(figsize=(11, 7))
        
        colors = {
            'CRITICAL': '#d62728',  # Red
            'HIGH': '#ff7f0e',      # Orange
            'MEDIUM': '#ffbb78',    # Light orange
            'LOW': '#2ca02c'        # Green
        }
        
        for alert_level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
            data = batch_scores_df[batch_scores_df['alert_level'] == alert_level]['risk_score']
            ax.hist(
                data,
                bins=20,
                alpha=0.6,
                label=f'{alert_level} (n={len(data)})',
                color=colors[alert_level]
            )
        
        ax.axvline(0.3, color='gray', linestyle='--', alpha=0.5, label='MEDIUM threshold')
        ax.axvline(0.5, color='gray', linestyle='-', alpha=0.7, label='HIGH threshold')
        ax.axvline(0.7, color='red', linestyle='-', alpha=0.7, linewidth=2, label='CRITICAL threshold')
        
        ax.set_xlabel('Risk Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Batches', fontsize=11, fontweight='bold')
        ax.set_title('Batch Risk Score Distribution',
                     fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = f'{self.output_dir}/02_batch_risk_distribution.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return path
    
    def viz_3_alert_level_breakdown(self, batch_scores_df: pd.DataFrame) -> str:
        """
        Visualization 3: Alert Level Breakdown
        Pie chart showing distribution across alert levels
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        alert_counts = batch_scores_df['alert_level'].value_counts()
        colors = ['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c']
        alert_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        
        # Reorder
        alert_counts = alert_counts.reindex([a for a in alert_order if a in alert_counts.index])
        colors = colors[:len(alert_counts)]
        
        wedges, texts, autotexts = ax.pie(
            alert_counts.values,
            labels=alert_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 11, 'weight': 'bold'}
        )
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
            autotext.set_fontsize(10)
        
        ax.set_title('Alert Level Distribution Across Batches',
                     fontsize=13, fontweight='bold', pad=15)
        
        # Add legend with counts
        legend_labels = [f'{level}: {count} batches' for level, count in zip(alert_counts.index, alert_counts.values)]
        ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(0.85, 1))
        
        plt.tight_layout()
        path = f'{self.output_dir}/03_alert_level_breakdown.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return path
    
    def viz_4_risk_component_heatmap(self, batch_scores_df: pd.DataFrame) -> str:
        """
        Visualization 4: Risk Component Contribution Heatmap
        Shows contribution of each risk factor for top batches
        """
        fig, ax = plt.subplots(figsize=(11, 8))
        
        # Get top 20 high-risk batches
        top_batches = batch_scores_df.nlargest(20, 'risk_score')
        
        components = [
            'temporal_concentration',
            'geographic_concentration',
            'event_similarity',
            'severity_concentration',
            'size_anomaly',
            'manufacturing_concentration'
        ]
        
        heatmap_data = top_batches[components].values
        
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(range(len(components)))
        ax.set_yticks(range(len(top_batches)))
        ax.set_xticklabels(
            ['Temporal', 'Geographic', 'Event\nSimilarity', 'Severity', 'Size', 'Manufacturing'],
            rotation=45,
            ha='right',
            fontsize=10
        )
        ax.set_yticklabels([f"Batch {i+1}" for i in range(len(top_batches))], fontsize=9)
        
        # Add text annotations
        for i in range(len(top_batches)):
            for j in range(len(components)):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Risk Component Contribution - Top 20 High-Risk Batches',
                     fontsize=13, fontweight='bold', pad=15)
        
        cbar = plt.colorbar(im, ax=ax, label='Component Score')
        
        plt.tight_layout()
        path = f'{self.output_dir}/04_risk_component_heatmap.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return path
    
    def viz_5_event_type_heatmap(self, df: pd.DataFrame) -> str:
        """
        Visualization 5: Event Type Distribution by Cluster
        Heatmap showing which event types dominate each cluster
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create event type by cluster crosstab
        cluster_event_crosstab = pd.crosstab(
            df[df['cluster_id'] != -1]['cluster_id'],
            df[df['cluster_id'] != -1]['event_type']
        )
        
        # Normalize by cluster
        cluster_event_norm = cluster_event_crosstab.div(cluster_event_crosstab.sum(axis=1), axis=0)
        
        sns.heatmap(
            cluster_event_norm,
            cmap='YlOrRd',
            cbar_kws={'label': 'Proportion'},
            ax=ax,
            annot=True,
            fmt='.2f',
            cbar=True
        )
        
        ax.set_xlabel('Event Type', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cluster ID', fontsize=11, fontweight='bold')
        ax.set_title('Event Type Distribution by Geographic Cluster',
                     fontsize=13, fontweight='bold', pad=15)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        path = f'{self.output_dir}/05_event_type_heatmap.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return path
    
    def viz_6_severity_distribution_by_alert(self, batch_scores_df: pd.DataFrame, df: pd.DataFrame) -> str:
        """
        Visualization 6: Case Severity Distribution by Alert Level
        Violin plot showing severity across alert levels
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Merge severity info with batch scores
        batch_severity = df.groupby('batch_id')['severity'].apply(
            lambda x: pd.Series({
                'Mild': (x == 'Mild').sum(),
                'Moderate': (x == 'Moderate').sum(),
                'Severe': (x == 'Severe').sum(),
                'Life-threatening': (x == 'Life-threatening').sum()
            })
        ).reset_index()
        
        merged = batch_scores_df.merge(batch_severity, on='batch_id')
        
        severity_map = {'Mild': 0, 'Moderate': 1, 'Severe': 2, 'Life-threatening': 3}
        
        # Create data for plot
        plot_data = []
        for _, row in merged.iterrows():
            for severity, value in severity_map.items():
                count = row[severity] if severity in row else 0
                plot_data.append({
                    'Alert Level': row['alert_level'],
                    'Severity Numeric': value,
                    'Count': count
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        sns.boxplot(
            data=merged,
            x='alert_level',
            y='size_anomaly',  # Using size_anomaly as proxy for severity trend
            palette='Set2',
            ax=ax
        )
        
        ax.set_xlabel('Alert Level', fontsize=11, fontweight='bold')
        ax.set_ylabel('Size Anomaly Score', fontsize=11, fontweight='bold')
        ax.set_title('Batch Characteristics by Alert Level',
                     fontsize=13, fontweight='bold', pad=15)
        ax.set_xticklabels(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = f'{self.output_dir}/06_severity_by_alert.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return path
    
    def viz_7_cluster_size_distribution(self, df: pd.DataFrame) -> str:
        """
        Visualization 7: Cluster Size Distribution
        Bar chart showing number of cases per cluster
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        cluster_sizes = df[df['cluster_id'] != -1].groupby('cluster_id').size().sort_values(ascending=False)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_sizes)))
        
        bars = ax.bar(
            range(len(cluster_sizes)),
            cluster_sizes.values,
            color=colors,
            edgecolor='black',
            linewidth=1
        )
        
        ax.set_xlabel('Cluster ID', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Cases', fontsize=11, fontweight='bold')
        ax.set_title('Cluster Size Distribution - Top 20 Clusters',
                     fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(range(min(20, len(cluster_sizes))))
        ax.set_xticklabels([f"C{cluster_sizes.index[i]}" for i in range(min(20, len(cluster_sizes)))],
                           rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars[:min(20, len(bars))]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=8)
        
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = f'{self.output_dir}/07_cluster_size_distribution.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return path
    
    def viz_8_temporal_concentration(self, df: pd.DataFrame) -> str:
        """
        Visualization 8: Temporal Concentration Timeline
        Shows how cases are distributed over time within top clusters
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        df = df.copy()
        df['date_reported'] = pd.to_datetime(df['date_reported'])
        
        # Get top 5 clusters by size
        top_clusters = df[df['cluster_id'] != -1]['cluster_id'].value_counts().head(5).index
        
        for cluster_id in top_clusters:
            cluster_df = df[df['cluster_id'] == cluster_id]
            
            # Group by date
            daily_counts = cluster_df.groupby(cluster_df['date_reported'].dt.date).size()
            
            ax.plot(
                daily_counts.index,
                daily_counts.values,
                marker='o',
                label=f'Cluster {cluster_id} (n={len(cluster_df)})',
                linewidth=2,
                markersize=4
            )
        
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cases per Day', fontsize=11, fontweight='bold')
        ax.set_title('Temporal Distribution of Cases - Top 5 Clusters',
                     fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        path = f'{self.output_dir}/08_temporal_concentration.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return path
    
    def generate_all_visualizations(self, df: pd.DataFrame, batch_scores_df: pd.DataFrame = None) -> Dict[str, str]:
        """
        Generate all 8 visualizations.
        
        Args:
            df: DataFrame with clustering results
            batch_scores_df: Optional DataFrame with batch scores
            
        Returns:
            Dictionary mapping viz names to file paths
        """
        print(f"\n{'='*70}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*70}\n")
        
        visualizations = {}
        
        print("Generating visualization 1: Geographic clusters...")
        visualizations['01_geographic_clusters'] = self.viz_1_geographic_clusters(df)
        
        if batch_scores_df is not None:
            print("Generating visualization 2: Batch risk distribution...")
            visualizations['02_batch_risk_distribution'] = self.viz_2_batch_risk_distribution(batch_scores_df)
            
            print("Generating visualization 3: Alert level breakdown...")
            visualizations['03_alert_level_breakdown'] = self.viz_3_alert_level_breakdown(batch_scores_df)
            
            print("Generating visualization 4: Risk component heatmap...")
            visualizations['04_risk_component_heatmap'] = self.viz_4_risk_component_heatmap(batch_scores_df)
        
        print("Generating visualization 5: Event type heatmap...")
        visualizations['05_event_type_heatmap'] = self.viz_5_event_type_heatmap(df)
        
        if batch_scores_df is not None:
            print("Generating visualization 6: Severity by alert...")
            visualizations['06_severity_by_alert'] = self.viz_6_severity_distribution_by_alert(batch_scores_df, df)
        
        print("Generating visualization 7: Cluster size distribution...")
        visualizations['07_cluster_size_distribution'] = self.viz_7_cluster_size_distribution(df)
        
        print("Generating visualization 8: Temporal concentration...")
        visualizations['08_temporal_concentration'] = self.viz_8_temporal_concentration(df)
        
        print(f"\nGenerated {len(visualizations)} visualizations in {self.output_dir}/\n")
        
        return visualizations


if __name__ == "__main__":
    from .data_generator import PopulationDataGenerator
    from .clustering_engine import DBSCANClusteringEngine
    from .batch_risk_scorer import BatchRiskScorer
    
    # Generate data
    print("Generating adverse event data...")
    gen = PopulationDataGenerator()
    df = gen.generate_train_test(num_cases=5000, anomalous_batches=5)
    
    # Cluster
    print("Running DBSCAN clustering...")
    clustering = DBSCANClusteringEngine(eps_km=50, min_samples=5)
    results = clustering.fit(df)
    df = results['df']
    
    # Score batches
    print("Scoring batches...")
    scorer = BatchRiskScorer()
    batch_scores = scorer.score_batches(df)
    
    # Visualize
    print("Creating visualizations...")
    visualizer = SignalDetectionVisualizer()
    viz_paths = visualizer.generate_all_visualizations(df, batch_scores)
    
    for name, path in viz_paths.items():
        print(f"  ✓ {name}: {path}")
    
    print("\nVisualization generation complete!")
