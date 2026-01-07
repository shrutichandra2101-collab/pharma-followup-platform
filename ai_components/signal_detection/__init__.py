"""
Geospatial Signal Detection Component
Batch anomaly detection and regional adverse event monitoring using DBSCAN clustering
"""

from .data_generator import PopulationDataGenerator
from .clustering_engine import DBSCANClusteringEngine, GeospatialFeatureExtractor
from .batch_risk_scorer import BatchRiskScorer
from .evaluation_metrics import SignalDetectionMetrics
from .visualizer import SignalDetectionVisualizer
from .signal_detector import SignalDetectionOrchestrator

__all__ = [
    'PopulationDataGenerator',
    'DBSCANClusteringEngine',
    'GeospatialFeatureExtractor',
    'BatchRiskScorer',
    'SignalDetectionMetrics',
    'SignalDetectionVisualizer',
    'SignalDetectionOrchestrator'
]

__version__ = '1.0.0'
