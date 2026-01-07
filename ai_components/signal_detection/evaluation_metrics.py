"""
Geospatial Signal Detection - Evaluation Metrics Engine
Calculate cluster quality and signal detection performance

Step 4: Implement evaluation metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, List, Tuple
import json
import warnings
warnings.filterwarnings('ignore')


class SignalDetectionMetrics:
    """Calculate metrics for signal detection performance."""
    
    @staticmethod
    def silhouette_coefficient(features: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate silhouette coefficient (-1 to 1, higher is better).
        Measures how similar points are to their own cluster vs other clusters.
        """
        # Ignore noise points (-1 label)
        mask = labels != -1
        if mask.sum() < 2:
            return 0.0
        
        try:
            score = silhouette_score(features[mask], labels[mask])
            return float(score)
        except:
            return 0.0
    
    @staticmethod
    def davies_bouldin_index(features: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate Davies-Bouldin Index (lower is better).
        Ratio of within-cluster to between-cluster distances.
        """
        mask = labels != -1
        if mask.sum() < 2 or len(set(labels[mask])) < 2:
            return float('inf')
        
        try:
            score = davies_bouldin_score(features[mask], labels[mask])
            return float(score)
        except:
            return float('inf')
    
    @staticmethod
    def calinski_harabasz_index(features: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate Calinski-Harabasz Index (higher is better).
        Ratio of between-cluster to within-cluster dispersion.
        """
        mask = labels != -1
        if mask.sum() < 2 or len(set(labels[mask])) < 2:
            return 0.0
        
        try:
            score = calinski_harabasz_score(features[mask], labels[mask])
            return float(score)
        except:
            return 0.0
    
    @staticmethod
    def cluster_density_metrics(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate density metrics for each cluster."""
        metrics = {}
        
        for cluster_id in set(df['cluster_id']):
            if cluster_id == -1:
                continue
            
            cluster_df = df[df['cluster_id'] == cluster_id]
            
            # Geographic density (cases per square degree)
            lat_range = cluster_df['latitude'].max() - cluster_df['latitude'].min()
            lng_range = cluster_df['longitude'].max() - cluster_df['longitude'].min()
            
            area = lat_range * lng_range
            if area == 0:
                area = 1e-6
            
            density = len(cluster_df) / area
            
            metrics[f'cluster_{cluster_id}_density'] = float(density)
            metrics[f'cluster_{cluster_id}_size'] = len(cluster_df)
        
        return metrics
    
    @staticmethod
    def temporal_lead_time(df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate how early signal detection identifies issues vs baseline.
        Baseline = typical case discovery lag (14 days).
        """
        df = df.copy()
        df['date_reported'] = pd.to_datetime(df['date_reported'])
        
        # For anomalous clusters, find how concentrated cases are
        metrics = {}
        
        for cluster_id in set(df['cluster_id']):
            if cluster_id == -1:
                continue
            
            cluster_df = df[df['cluster_id'] == cluster_id]
            
            if len(cluster_df) < 3:
                continue
            
            dates = cluster_df['date_reported'].sort_values()
            
            # Calculate how many cases occur in first week vs overall
            first_date = dates.iloc[0]
            week_later = first_date + pd.Timedelta(days=7)
            
            cases_in_first_week = len(cluster_df[cluster_df['date_reported'] <= week_later])
            pct_in_first_week = cases_in_first_week / len(cluster_df)
            
            # Lead time in days (how early we could detect)
            # Higher concentration in first week = earlier detection possible
            lead_time_days = max(0, 14 * (1 - pct_in_first_week))
            
            metrics[f'cluster_{cluster_id}_lead_time_days'] = float(lead_time_days)
            metrics[f'cluster_{cluster_id}_concentration_week1'] = float(pct_in_first_week)
        
        return metrics
    
    @staticmethod
    def batch_anomaly_detection_accuracy(df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate accuracy of batch anomaly detection.
        Based on whether truly anomalous batches are detected as high-risk.
        """
        # Get batches that came from anomalous clusters in data_generator
        df = df.copy()
        
        # Calculate metrics
        total_batches = df['batch_id'].nunique()
        batches_with_high_risk_score = df[df['risk_score'] >= 0.5]['batch_id'].nunique()
        batches_with_critical_alert = df[df['alert_level'] == 'CRITICAL']['batch_id'].nunique()
        
        return {
            'total_unique_batches': total_batches,
            'batches_flagged_high_risk': batches_with_high_risk_score,
            'batches_flagged_critical': batches_with_critical_alert,
            'pct_high_risk': batches_with_high_risk_score / total_batches * 100,
            'pct_critical': batches_with_critical_alert / total_batches * 100
        }
    
    @staticmethod
    def noise_point_analysis(df: pd.DataFrame) -> Dict[str, any]:
        """Analyze noise points (potential outliers)."""
        noise_df = df[df['cluster_id'] == -1]
        
        if len(noise_df) == 0:
            return {
                'total_noise_points': 0,
                'pct_noise': 0.0,
                'noise_severity_distribution': {}
            }
        
        severity_dist = noise_df['severity'].value_counts().to_dict()
        
        return {
            'total_noise_points': len(noise_df),
            'pct_noise': len(noise_df) / len(df) * 100,
            'noise_severity_distribution': {k: int(v) for k, v in severity_dist.items()},
            'avg_noise_severity': float(
                pd.Series({
                    'Mild': 0, 'Moderate': 1, 'Severe': 2, 'Life-threatening': 3
                }).reindex(noise_df['severity']).mean()
            )
        }
    
    @staticmethod
    def clustering_performance_report(
        df: pd.DataFrame,
        features: np.ndarray,
        labels: np.ndarray,
        batch_scores_df: pd.DataFrame = None
    ) -> Dict[str, any]:
        """
        Generate comprehensive clustering performance report.
        
        Args:
            df: DataFrame with clustering results
            features: Feature matrix used for clustering
            labels: Cluster labels
            batch_scores_df: Optional batch-level scores
            
        Returns:
            Dictionary with all metrics
        """
        print(f"\n{'='*70}")
        print("SIGNAL DETECTION EVALUATION METRICS")
        print(f"{'='*70}\n")
        
        report = {
            'clustering_quality': {
                'silhouette_coefficient': SignalDetectionMetrics.silhouette_coefficient(
                    features, labels
                ),
                'davies_bouldin_index': SignalDetectionMetrics.davies_bouldin_index(
                    features, labels
                ),
                'calinski_harabasz_index': SignalDetectionMetrics.calinski_harabasz_index(
                    features, labels
                )
            },
            'cluster_statistics': {
                'total_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'total_points': len(df),
                'clustered_points': (labels != -1).sum(),
                'noise_points': (labels == -1).sum()
            },
            'density_metrics': SignalDetectionMetrics.cluster_density_metrics(df),
            'temporal_metrics': SignalDetectionMetrics.temporal_lead_time(df),
            'noise_analysis': SignalDetectionMetrics.noise_point_analysis(df),
        }
        
        if batch_scores_df is not None:
            report['batch_detection'] = SignalDetectionMetrics.batch_anomaly_detection_accuracy(df)
        
        # Print summary
        print("Clustering Quality Metrics:")
        print(f"  Silhouette Coefficient: {report['clustering_quality']['silhouette_coefficient']:.3f}")
        print(f"    (Range: -1 to 1, Higher is better)")
        print(f"  Davies-Bouldin Index: {report['clustering_quality']['davies_bouldin_index']:.3f}")
        print(f"    (Lower is better, <1.5 is excellent)")
        print(f"  Calinski-Harabasz Index: {report['clustering_quality']['calinski_harabasz_index']:.1f}")
        print(f"    (Higher is better)")
        
        print(f"\nCluster Statistics:")
        print(f"  Total clusters: {report['cluster_statistics']['total_clusters']}")
        print(f"  Total points: {report['cluster_statistics']['total_points']}")
        print(f"  Clustered: {report['cluster_statistics']['clustered_points']} ({report['cluster_statistics']['clustered_points']/report['cluster_statistics']['total_points']*100:.1f}%)")
        print(f"  Noise: {report['cluster_statistics']['noise_points']} ({report['cluster_statistics']['noise_points']/report['cluster_statistics']['total_points']*100:.1f}%)")
        
        print(f"\nNoise Point Analysis:")
        noise = report['noise_analysis']
        print(f"  Total noise points: {noise['total_noise_points']} ({noise['pct_noise']:.1f}%)")
        if noise['total_noise_points'] > 0:
            print(f"  Avg severity: {noise['avg_noise_severity']:.2f}")
        
        if 'batch_detection' in report:
            print(f"\nBatch Anomaly Detection:")
            bd = report['batch_detection']
            print(f"  Total unique batches: {bd['total_unique_batches']}")
            print(f"  Flagged as high-risk: {bd['batches_flagged_high_risk']} ({bd['pct_high_risk']:.1f}%)")
            print(f"  Flagged as critical: {bd['batches_flagged_critical']} ({bd['pct_critical']:.1f}%)")
        
        return report


if __name__ == "__main__":
    from .data_generator import PopulationDataGenerator
    from .clustering_engine import DBSCANClusteringEngine, GeospatialFeatureExtractor
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
    features = clustering.scaler.fit_transform(
        GeospatialFeatureExtractor().extract_features(df)
    )
    
    # Score batches
    print("Scoring batches...")
    scorer = BatchRiskScorer()
    batch_scores = scorer.score_batches(df)
    
    # Evaluate
    print("Evaluating performance...")
    metrics = SignalDetectionMetrics.clustering_performance_report(
        df, features, clustering.cluster_labels, batch_scores
    )
    
    # Save metrics
    with open('signal_detection_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print("\nMetrics saved to signal_detection_metrics.json")
