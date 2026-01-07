"""
Geospatial Signal Detection - Orchestrator
Main pipeline coordinator for the signal detection system

Step 6: Implement orchestrator
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
import sys
from datetime import datetime

try:
    from .data_generator import PopulationDataGenerator
    from .clustering_engine import DBSCANClusteringEngine, GeospatialFeatureExtractor
    from .batch_risk_scorer import BatchRiskScorer
    from .evaluation_metrics import SignalDetectionMetrics
    from .visualizer import SignalDetectionVisualizer
except ImportError:
    from data_generator import PopulationDataGenerator
    from clustering_engine import DBSCANClusteringEngine, GeospatialFeatureExtractor
    from batch_risk_scorer import BatchRiskScorer
    from evaluation_metrics import SignalDetectionMetrics
    from visualizer import SignalDetectionVisualizer


class SignalDetectionOrchestrator:
    """Main orchestrator for the signal detection pipeline."""
    
    def __init__(self, output_dir: str = 'signal_detection_results'):
        """
        Initialize orchestrator.
        
        Args:
            output_dir: Directory to save all outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.data_generator = PopulationDataGenerator()
        self.clustering_engine = DBSCANClusteringEngine(eps_km=50, min_samples=5)
        self.batch_scorer = BatchRiskScorer()
        self.visualizer = SignalDetectionVisualizer(str(self.output_dir / 'visualizations'))
        
        self.df = None
        self.batch_scores = None
        self.features = None
        self.metrics = None
        self.visualizations = None
    
    def run_pipeline(self, num_cases: int = 5000, anomalous_batches: int = 5) -> Dict[str, any]:
        """
        Run complete signal detection pipeline.
        
        Args:
            num_cases: Number of adverse events to generate
            anomalous_batches: Number of anomalous batch clusters
            
        Returns:
            Dictionary with all pipeline outputs
        """
        print(f"\n{'='*70}")
        print("GEOSPATIAL SIGNAL DETECTION PIPELINE")
        print(f"{'='*70}\n")
        
        print(f"Pipeline Configuration:")
        print(f"  Total cases to generate: {num_cases}")
        print(f"  Anomalous batch clusters: {anomalous_batches}")
        print(f"  Output directory: {self.output_dir}\n")
        
        # Step 1: Generate data
        print(f"{'='*70}")
        print("STEP 1: GENERATING SYNTHETIC DATA")
        print(f"{'='*70}\n")
        
        self.df = self.data_generator.generate_train_test(
            num_cases=num_cases,
            anomalous_batches=anomalous_batches
        )
        
        # Add severity_numeric for metrics calculation
        severity_map = {'Mild': 0, 'Moderate': 1, 'Severe': 2, 'Life-threatening': 3}
        self.df['severity_numeric'] = self.df['severity'].map(severity_map)
        
        # Step 2: Clustering
        print(f"\n{'='*70}")
        print("STEP 2: GEOGRAPHIC CLUSTERING")
        print(f"{'='*70}\n")
        
        clustering_results = self.clustering_engine.fit(self.df)
        self.df = clustering_results['df']
        
        # Step 3: Batch risk scoring
        print(f"\n{'='*70}")
        print("STEP 3: BATCH RISK SCORING")
        print(f"{'='*70}\n")
        
        self.batch_scores = self.batch_scorer.score_batches(self.df)
        self.df = self.batch_scorer.score_individual_cases(self.df, self.batch_scores)
        
        # Step 4: Evaluation metrics
        print(f"\n{'='*70}")
        print("STEP 4: EVALUATION METRICS")
        print(f"{'='*70}\n")
        
        # Prepare features for metrics
        feature_extractor = GeospatialFeatureExtractor()
        features = feature_extractor.extract_features(self.df)
        self.features = self.clustering_engine.scaler.transform(features)
        
        self.metrics = SignalDetectionMetrics.clustering_performance_report(
            self.df,
            self.features,
            self.clustering_engine.cluster_labels,
            self.batch_scores
        )
        
        # Step 5: Visualizations
        print(f"\n{'='*70}")
        print("STEP 5: GENERATING VISUALIZATIONS")
        print(f"{'='*70}\n")
        
        self.visualizations = self.visualizer.generate_all_visualizations(self.df, self.batch_scores)
        
        # Step 6: Save outputs
        print(f"\n{'='*70}")
        print("STEP 6: SAVING OUTPUTS")
        print(f"{'='*70}\n")
        
        self._save_outputs()
        
        print("Pipeline execution complete!")
        
        return {
            'df': self.df,
            'batch_scores': self.batch_scores,
            'metrics': self.metrics,
            'visualizations': self.visualizations
        }
    
    def _save_outputs(self):
        """Save all pipeline outputs to files."""
        
        # Save processed data
        print("Saving processed data...")
        data_file = self.output_dir / 'signal_detection_data.csv'
        self.df.to_csv(data_file, index=False)
        print(f"  ✓ Data: {data_file}")
        
        # Save batch scores
        print("Saving batch scores...")
        scores_file = self.output_dir / 'batch_risk_scores.csv'
        self.batch_scores.to_csv(scores_file, index=False)
        print(f"  ✓ Batch scores: {scores_file}")
        
        # Save metrics
        print("Saving evaluation metrics...")
        metrics_file = self.output_dir / 'signal_detection_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        print(f"  ✓ Metrics: {metrics_file}")
        
        # Save comprehensive report
        print("Saving comprehensive report...")
        report = self._generate_report()
        report_file = self.output_dir / 'SIGNAL_DETECTION_REPORT.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"  ✓ Report: {report_file}")
    
    def _generate_report(self) -> str:
        """Generate comprehensive text report."""
        report = []
        
        report.append("="*70)
        report.append("GEOSPATIAL SIGNAL DETECTION SYSTEM - FINAL REPORT")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Pipeline statistics
        report.append("PIPELINE STATISTICS")
        report.append("-" * 70)
        report.append(f"Total cases processed: {len(self.df)}")
        report.append(f"Total unique batches: {self.df['batch_id'].nunique()}")
        report.append(f"Total clusters identified: {self.metrics['cluster_statistics']['total_clusters']}")
        report.append(f"Clustered cases: {self.metrics['cluster_statistics']['clustered_points']}")
        report.append(f"Noise points (potential outliers): {self.metrics['cluster_statistics']['noise_points']}")
        report.append("")
        
        # Clustering quality
        report.append("CLUSTERING QUALITY METRICS")
        report.append("-" * 70)
        clustering = self.metrics['clustering_quality']
        report.append(f"Silhouette Coefficient: {clustering['silhouette_coefficient']:.4f}")
        report.append(f"  (Range: -1 to 1, closer to 1 indicates better defined clusters)")
        report.append(f"Davies-Bouldin Index: {clustering['davies_bouldin_index']:.4f}")
        report.append(f"  (Lower is better, <1.5 indicates excellent separation)")
        report.append(f"Calinski-Harabasz Index: {clustering['calinski_harabasz_index']:.2f}")
        report.append(f"  (Higher indicates stronger, more separated clusters)")
        report.append("")
        
        # Batch anomaly detection
        if 'batch_detection' in self.metrics:
            report.append("BATCH ANOMALY DETECTION")
            report.append("-" * 70)
            bd = self.metrics['batch_detection']
            report.append(f"Total unique batches: {bd['total_unique_batches']}")
            report.append(f"Flagged as high-risk (≥0.5): {bd['batches_flagged_high_risk']} ({bd['pct_high_risk']:.1f}%)")
            report.append(f"Flagged as critical (≥0.7): {bd['batches_flagged_critical']} ({bd['pct_critical']:.1f}%)")
            report.append("")
        
        # Noise analysis
        report.append("OUTLIER/NOISE ANALYSIS")
        report.append("-" * 70)
        noise = self.metrics['noise_analysis']
        report.append(f"Total noise points: {noise['total_noise_points']} ({noise['pct_noise']:.1f}%)")
        if noise['total_noise_points'] > 0:
            report.append(f"Average noise severity: {noise['avg_noise_severity']:.2f}")
            report.append(f"Noise severity distribution:")
            for severity, count in noise['noise_severity_distribution'].items():
                report.append(f"  {severity}: {count}")
        report.append("")
        
        # Top high-risk batches
        report.append("TOP 10 HIGH-RISK BATCHES")
        report.append("-" * 70)
        for i, (_, batch) in enumerate(self.batch_scores.nlargest(10, 'risk_score').iterrows(), 1):
            report.append(f"\n{i}. Batch: {batch['batch_id']}")
            report.append(f"   Risk Score: {batch['risk_score']:.3f} ({batch['alert_level']})")
            report.append(f"   Cases: {batch['num_cases']}")
            report.append(f"   Location: {batch['primary_region']}")
            report.append(f"   Drug: {batch['primary_drug']}")
            report.append(f"   Primary Event: {batch['primary_event']}")
            report.append(f"   Geographic Concentration: {batch['geographic_concentration']:.2f}")
            report.append(f"   Temporal Concentration: {batch['temporal_concentration']:.2f}")
            report.append(f"   Event Similarity: {batch['event_similarity']:.2f}")
            report.append(f"   Size Anomaly: {batch['size_anomaly']:.2f}")
        report.append("")
        
        # Temporal insights
        report.append("TEMPORAL ANALYSIS")
        report.append("-" * 70)
        for key, value in self.metrics['temporal_metrics'].items():
            if 'lead_time' in key:
                report.append(f"{key}: {value:.2f} days (early detection potential)")
        report.append("")
        
        # Recommendations
        report.append("KEY FINDINGS & RECOMMENDATIONS")
        report.append("-" * 70)
        
        critical_count = len(self.batch_scores[self.batch_scores['alert_level'] == 'CRITICAL'])
        high_count = len(self.batch_scores[self.batch_scores['alert_level'] == 'HIGH'])
        
        report.append(f"\n1. ALERT SUMMARY")
        report.append(f"   - {critical_count} critical batches detected")
        report.append(f"   - {high_count} high-risk batches detected")
        report.append(f"   - Recommend immediate investigation of critical batches")
        
        report.append(f"\n2. GEOGRAPHIC PATTERNS")
        top_region = self.batch_scores['primary_region'].value_counts().index[0]
        report.append(f"   - Highest risk concentration in: {top_region}")
        report.append(f"   - Recommend regional regulatory outreach")
        
        report.append(f"\n3. EVENT PATTERNS")
        top_event = self.batch_scores['primary_event'].value_counts().index[0]
        report.append(f"   - Most frequent event type: {top_event}")
        report.append(f"   - Recommend clinical review for pattern analysis")
        
        report.append(f"\n4. EARLY DETECTION")
        avg_lead_time = np.mean([v for k, v in self.metrics['temporal_metrics'].items() if 'lead_time' in k])
        if avg_lead_time > 0:
            report.append(f"   - System can detect signals {avg_lead_time:.1f} days earlier than baseline")
            report.append(f"   - Enables proactive intervention and monitoring")
        
        report.append(f"\n5. MONITORING RECOMMENDATIONS")
        report.append(f"   - Implement continuous surveillance for high-risk batches")
        report.append(f"   - Monitor geographic clusters in {top_region} closely")
        report.append(f"   - Consider batch-level restrictions or additional oversight")
        
        report.append("\n" + "="*70)
        report.append(f"End of Report")
        report.append("="*70 + "\n")
        
        return "\n".join(report)


def main():
    """Run the complete signal detection pipeline."""
    
    # Create orchestrator
    orchestrator = SignalDetectionOrchestrator()
    
    # Run pipeline
    results = orchestrator.run_pipeline(
        num_cases=5000,
        anomalous_batches=5
    )
    
    print("\n" + "="*70)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*70)
    print(f"\nOutputs saved to: {orchestrator.output_dir}")
    print("\nGenerated files:")
    print(f"  ✓ signal_detection_data.csv")
    print(f"  ✓ batch_risk_scores.csv")
    print(f"  ✓ signal_detection_metrics.json")
    print(f"  ✓ SIGNAL_DETECTION_REPORT.txt")
    print(f"  ✓ 8 professional visualizations (PNG)")
    print("\nPipeline complete! Ready for dashboard deployment.")


if __name__ == "__main__":
    main()
