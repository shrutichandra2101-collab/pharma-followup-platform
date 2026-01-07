"""
Geospatial Signal Detection - Batch Risk Scoring Engine
Calculate anomaly severity and batch-level risk scores

Step 3: Implement batch risk scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import cdist
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class BatchRiskScorer:
    """Calculate risk scores for batches based on clustering results."""
    
    def __init__(self):
        """Initialize risk scorer."""
        self.batch_scores = {}
        self.case_scores = {}
    
    def calculate_temporal_concentration(self, df: pd.DataFrame, batch_id: str) -> float:
        """
        Calculate temporal concentration score (0-1).
        Higher score = cases concentrated in shorter time period = more anomalous.
        """
        batch_df = df[df['batch_id'] == batch_id]
        
        if len(batch_df) < 2:
            return 0.0
        
        # Convert date_reported to datetime if needed
        dates = pd.to_datetime(batch_df['date_reported'])
        date_range_days = (dates.max() - dates.min()).days
        
        if date_range_days == 0:
            return 1.0  # All cases reported on same day = high concentration
        
        # Expected range is 90 days (3 months) for normal batches
        # Concentration score increases as range decreases
        concentration = max(0, 1 - (date_range_days / 90.0))
        
        return min(1.0, concentration)
    
    def calculate_geographic_concentration(self, df: pd.DataFrame, cluster_id: int) -> float:
        """
        Calculate geographic concentration score (0-1).
        Higher score = cases clustered in smaller geographic area = more anomalous.
        """
        cluster_df = df[df['cluster_id'] == cluster_id]
        
        if len(cluster_df) < 2:
            return 0.0
        
        # Calculate geographic spread using standard deviation of coordinates
        lat_std = cluster_df['latitude'].std()
        lng_std = cluster_df['longitude'].std()
        
        # Combined spread (normalized to 0-1)
        # Spread of 0.1 degrees = ~11 km, is considered concentrated
        spread = np.sqrt(lat_std**2 + lng_std**2)
        concentration = max(0, 1 - (spread / 0.1))
        
        return min(1.0, concentration)
    
    def calculate_event_similarity(self, df: pd.DataFrame, cluster_id: int) -> float:
        """
        Calculate event type similarity score (0-1).
        Higher score = similar event types across cluster = more anomalous.
        """
        cluster_df = df[df['cluster_id'] == cluster_id]
        
        if len(cluster_df) < 2:
            return 0.0
        
        # Get event type distribution
        event_counts = cluster_df['event_type'].value_counts()
        
        # Calculate entropy (normalized)
        from scipy.stats import entropy
        event_probs = event_counts.values / event_counts.sum()
        max_entropy = np.log(len(event_counts))
        
        if max_entropy == 0:
            return 1.0
        
        event_entropy = entropy(event_probs)
        # Low entropy (similar events) = high similarity = high anomaly risk
        similarity = 1 - (event_entropy / max_entropy)
        
        return min(1.0, similarity)
    
    def calculate_severity_concentration(self, df: pd.DataFrame, cluster_id: int) -> float:
        """
        Calculate severity concentration score (0-1).
        Higher score = cluster has high-severity cases = more anomalous.
        """
        cluster_df = df[df['cluster_id'] == cluster_id]
        
        if len(cluster_df) == 0:
            return 0.0
        
        # Map severity to numeric
        severity_map = {'Mild': 0, 'Moderate': 1, 'Severe': 2, 'Life-threatening': 3}
        severities = cluster_df['severity'].map(severity_map)
        
        avg_severity = severities.mean()
        
        # Score based on average severity (0-3 scale normalized to 0-1)
        return min(1.0, avg_severity / 3.0)
    
    def calculate_size_anomaly(self, cluster_size: int, baseline_size: float = 5) -> float:
        """
        Calculate size anomaly score (0-1).
        Cluster much larger than expected = more anomalous.
        """
        if cluster_size <= baseline_size:
            return 0.0
        
        # Exponential growth: size=10 -> score=0.2, size=30 -> score=0.6, size=50 -> score=0.85
        size_anomaly = 1 - np.exp(-0.05 * (cluster_size - baseline_size))
        
        return min(1.0, size_anomaly)
    
    def calculate_manufacturing_concentration(self, df: pd.DataFrame, batch_id: str) -> float:
        """
        Check if batch came from single manufacturing site (potential source).
        """
        batch_df = df[df['batch_id'] == batch_id]
        
        if len(batch_df) < 2:
            return 0.0
        
        mfg_sites = batch_df['manufacturing_site'].nunique()
        
        # All cases from single site = high concentration
        concentration = 1 - (1 / mfg_sites)
        
        return min(1.0, concentration)
    
    def score_batches(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive risk scores for all batches.
        
        Args:
            df: DataFrame with clustering results (must have cluster_id column)
            
        Returns:
            DataFrame with batch-level risk scores
        """
        print(f"\n{'='*70}")
        print("BATCH RISK SCORING")
        print(f"{'='*70}\n")
        
        batch_scores = []
        
        for batch_id in df['batch_id'].unique():
            batch_df = df[df['batch_id'] == batch_id]
            
            # Get cluster ID for this batch
            cluster_id = batch_df['cluster_id'].mode()[0] if len(batch_df['cluster_id'].mode()) > 0 else -1
            
            # Calculate component scores
            temporal = self.calculate_temporal_concentration(df, batch_id)
            geographic = self.calculate_geographic_concentration(df, cluster_id)
            event_sim = self.calculate_event_similarity(df, cluster_id)
            severity = self.calculate_severity_concentration(df, cluster_id)
            size = self.calculate_size_anomaly(len(batch_df))
            mfg = self.calculate_manufacturing_concentration(df, batch_id)
            
            # Weighted combination
            components = {
                'temporal_concentration': temporal,
                'geographic_concentration': geographic,
                'event_similarity': event_sim,
                'severity_concentration': severity,
                'size_anomaly': size,
                'manufacturing_concentration': mfg
            }
            
            # Weights emphasize geographic, temporal, and size factors
            weights = {
                'temporal_concentration': 0.20,
                'geographic_concentration': 0.25,
                'event_similarity': 0.15,
                'severity_concentration': 0.15,
                'size_anomaly': 0.20,
                'manufacturing_concentration': 0.05
            }
            
            total_score = sum(
                components[key] * weights[key] for key in components.keys()
            )
            
            # Determine alert level
            if total_score >= 0.7:
                alert_level = 'CRITICAL'
            elif total_score >= 0.5:
                alert_level = 'HIGH'
            elif total_score >= 0.3:
                alert_level = 'MEDIUM'
            else:
                alert_level = 'LOW'
            
            batch_scores.append({
                'batch_id': batch_id,
                'cluster_id': cluster_id,
                'num_cases': len(batch_df),
                'risk_score': total_score,
                'alert_level': alert_level,
                'temporal_concentration': temporal,
                'geographic_concentration': geographic,
                'event_similarity': event_sim,
                'severity_concentration': severity,
                'size_anomaly': size,
                'manufacturing_concentration': mfg,
                'primary_region': batch_df['region'].mode()[0] if len(batch_df['region'].mode()) > 0 else 'Unknown',
                'primary_drug': batch_df['drug_name'].mode()[0] if len(batch_df['drug_name'].mode()) > 0 else 'Unknown',
                'primary_event': batch_df['event_type'].mode()[0] if len(batch_df['event_type'].mode()) > 0 else 'Unknown',
            })
        
        scores_df = pd.DataFrame(batch_scores)
        scores_df = scores_df.sort_values('risk_score', ascending=False)
        
        print(f"Scored {len(scores_df)} batches:")
        print(f"\n  Alert Level Distribution:")
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            count = len(scores_df[scores_df['alert_level'] == level])
            pct = count / len(scores_df) * 100
            print(f"    {level}: {count} ({pct:.1f}%)")
        
        print(f"\n  Top 10 High-Risk Batches:")
        for i, row in scores_df.head(10).iterrows():
            print(f"\n    Batch {i+1}: {row['batch_id']}")
            print(f"      Risk Score: {row['risk_score']:.3f} ({row['alert_level']})")
            print(f"      Cases: {row['num_cases']}")
            print(f"      Region: {row['primary_region']}")
            print(f"      Event: {row['primary_event']}")
            print(f"      Geographic Concentration: {row['geographic_concentration']:.2f}")
            print(f"      Temporal Concentration: {row['temporal_concentration']:.2f}")
        
        return scores_df
    
    def score_individual_cases(self, df: pd.DataFrame, batch_scores_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk scores for individual cases.
        
        Args:
            df: DataFrame with all cases and clustering results
            batch_scores_df: DataFrame with batch-level scores
            
        Returns:
            DataFrame with case-level risk scores
        """
        # Merge batch scores into case data
        df = df.merge(
            batch_scores_df[['batch_id', 'risk_score', 'alert_level']],
            on='batch_id',
            how='left'
        )
        
        # Case-level adjustments
        # Cases in high-risk batches with high severity get boosted scores
        df['case_risk_score'] = df['risk_score'].copy()
        
        severity_map = {'Mild': 0.1, 'Moderate': 0.3, 'Severe': 0.6, 'Life-threatening': 0.9}
        severity_boost = df['severity'].map(severity_map).fillna(0.3)
        
        # Boost score based on severity within high-risk batch
        df['case_risk_score'] = (
            df['risk_score'] * 0.7 + severity_boost * 0.3
        )
        
        df['case_alert_level'] = df['case_risk_score'].apply(
            lambda x: 'CRITICAL' if x >= 0.7 else (
                'HIGH' if x >= 0.5 else (
                    'MEDIUM' if x >= 0.3 else 'LOW'
                )
            )
        )
        
        return df


if __name__ == "__main__":
    from .data_generator import PopulationDataGenerator
    from .clustering_engine import DBSCANClusteringEngine
    
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
    
    # Score cases
    df = scorer.score_individual_cases(df, batch_scores)
    
    print("\nRisk scoring complete!")
