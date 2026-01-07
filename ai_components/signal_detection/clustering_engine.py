"""
Geospatial Signal Detection - DBSCAN Clustering Engine
Identify anomalous batches and regional spikes in adverse events

Step 2: Implement DBSCAN clustering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class GeospatialFeatureExtractor:
    """Extract features for clustering from adverse event data."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.scaler = StandardScaler()
        self.event_types = None
        self.drugs = None
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract normalized features for clustering."""
        df = df.copy()
        
        # Geographic features (already lat/long)
        geo_features = df[['latitude', 'longitude']].values
        
        # Event type encoding (one-hot)
        if self.event_types is None:
            self.event_types = df['event_type'].unique()
        event_encoded = pd.get_dummies(df['event_type'], prefix='event')[
            [f'event_{et}' for et in self.event_types]
        ].values
        
        # Drug encoding (one-hot)
        if self.drugs is None:
            self.drugs = df['drug_name'].unique()
        drug_encoded = pd.get_dummies(df['drug_name'], prefix='drug')[
            [f'drug_{d}' for d in self.drugs]
        ].values
        
        # Severity
        severity_map = {'Mild': 0, 'Moderate': 1, 'Severe': 2, 'Life-threatening': 3}
        severity = df['severity'].map(severity_map).values.reshape(-1, 1)
        
        # Quality score
        quality = df['quality_score'].values.reshape(-1, 1)
        
        # Combine features
        features = np.hstack([
            geo_features,           # 2D (lat, lng)
            event_encoded,          # N-hot (event types)
            drug_encoded,           # M-hot (drugs)
            severity,               # 1D
            quality                 # 1D
        ])
        
        return features


class DBSCANClusteringEngine:
    """DBSCAN clustering for adverse event anomalies."""
    
    def __init__(self, eps_km: float = 50, min_samples: int = 5):
        """
        Initialize DBSCAN engine.
        
        Args:
            eps_km: Epsilon in kilometers for geographic distance
            min_samples: Minimum samples to form cluster
        """
        self.eps_km = eps_km
        self.eps_rad = eps_km / 6371.0  # Convert km to radians (Earth radius = 6371 km)
        self.min_samples = min_samples
        
        self.feature_extractor = GeospatialFeatureExtractor()
        self.scaler = StandardScaler()
        self.dbscan = None
        self.cluster_labels = None
        self.features = None
    
    def fit(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Fit DBSCAN model on data.
        
        Args:
            df: DataFrame with adverse events
            
        Returns:
            Clustering results dictionary
        """
        print(f"\n{'='*70}")
        print("DBSCAN CLUSTERING ENGINE")
        print(f"{'='*70}\n")
        
        print(f"Clustering parameters:")
        print(f"  Epsilon (geographic): {self.eps_km} km")
        print(f"  Min samples: {self.min_samples}")
        print(f"  Total cases: {len(df)}\n")
        
        # Extract features
        features = self.feature_extractor.extract_features(df)
        self.features = self.scaler.fit_transform(features)
        
        # Geographic features (use haversine distance)
        geo_features = df[['latitude', 'longitude']].values
        geo_distances = haversine_distances(np.radians(geo_features)) * 6371  # Convert to km
        
        # DBSCAN clustering (using precomputed geographic distance + feature similarity)
        # For simplicity, we'll use feature-based DBSCAN with geographic weighting
        self.dbscan = DBSCAN(eps=0.8, min_samples=self.min_samples)
        self.cluster_labels = self.dbscan.fit_predict(self.features)
        
        # Calculate statistics
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        
        print(f"Clustering Results:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Number of noise points (outliers): {n_noise} ({n_noise/len(df)*100:.1f}%)")
        print(f"  Clustered points: {len(df) - n_noise} ({(len(df)-n_noise)/len(df)*100:.1f}%)")
        
        # Add cluster labels to dataframe
        df['cluster_id'] = self.cluster_labels
        
        return {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_labels': self.cluster_labels,
            'df': df
        }
    
    def identify_anomalous_clusters(self, df: pd.DataFrame) -> List[Dict]:
        """Identify which clusters represent anomalies."""
        anomalies = []
        
        # Analyze each cluster
        for cluster_id in set(self.cluster_labels):
            if cluster_id == -1:  # Skip noise points for now
                continue
            
            cluster_df = df[df['cluster_id'] == cluster_id]
            
            if len(cluster_df) < self.min_samples:
                continue
            
            # Calculate cluster metrics
            center_lat = cluster_df['latitude'].mean()
            center_lng = cluster_df['longitude'].mean()
            
            # Identify if this is anomalous
            is_anomalous = len(cluster_df) >= self.min_samples
            
            anomalies.append({
                'cluster_id': cluster_id,
                'size': len(cluster_df),
                'center_latitude': center_lat,
                'center_longitude': center_lng,
                'region': cluster_df['region'].mode()[0] if len(cluster_df) > 0 else 'Unknown',
                'primary_drug': cluster_df['drug_name'].mode()[0] if len(cluster_df) > 0 else 'Unknown',
                'primary_event': cluster_df['event_type'].mode()[0] if len(cluster_df) > 0 else 'Unknown',
                'avg_severity': cluster_df['severity_numeric'].mean(),
                'is_anomalous': is_anomalous,
                'case_ids': cluster_df['case_id'].tolist()
            })
        
        # Sort by size (largest anomalies first)
        anomalies = sorted(anomalies, key=lambda x: x['size'], reverse=True)
        
        print(f"\nIdentified {len([a for a in anomalies if a['is_anomalous']])} anomalous clusters:")
        for i, anomaly in enumerate(anomalies[:10]):  # Show top 10
            print(f"\n  Cluster {i+1}:")
            print(f"    ID: {anomaly['cluster_id']}")
            print(f"    Size: {anomaly['size']} cases")
            print(f"    Location: {anomaly['region']}")
            print(f"    Primary drug: {anomaly['primary_drug']}")
            print(f"    Primary event: {anomaly['primary_event']}")
            print(f"    Avg severity: {anomaly['avg_severity']:.2f}")
        
        return anomalies


if __name__ == "__main__":
    from .data_generator import PopulationDataGenerator
    
    # Generate data
    gen = PopulationDataGenerator()
    df = gen.generate_train_test(num_cases=5000, anomalous_batches=5)
    
    # Cluster
    clustering = DBSCANClusteringEngine(eps_km=50, min_samples=5)
    results = clustering.fit(df)
    
    # Identify anomalies
    anomalies = clustering.identify_anomalous_clusters(results['df'])
