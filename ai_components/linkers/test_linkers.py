"""
Pipeline Linker Test Script
Demonstrates data flow through Components 1 → 2 → 3
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

sys.path.append('../..')

from linkers.end_to_end_linker import EndToEndPipelineLinker
from linkers.prioritization_to_validation import PrioritizationToValidationLinker
from linkers.validation_to_questionnaire import ValidationToQuestionnaireLinker


def generate_sample_prioritization_output(n_records: int = 1000) -> pd.DataFrame:
    """Generate sample prioritization model output."""
    print("Generating sample Prioritization Model (Component 1) output...")
    
    return pd.DataFrame({
        'case_id': [f'CASE_{i:06d}' for i in range(n_records)],
        'priority_score': np.random.beta(8, 4, n_records),
        'follow_up_urgency': np.random.beta(6, 3, n_records),
        'estimated_response_time_hours': np.random.gamma(2, 24, n_records),
        'reporter_reliability': np.random.beta(7, 3, n_records),
        'regional_significance': np.random.choice([0.3, 0.5, 0.7, 0.9], n_records),
        'regulatory_deadline': [(datetime.now() + timedelta(days=np.random.randint(1, 90))).date() 
                                for _ in range(n_records)],
    })


def generate_sample_validation_output(n_records: int = 1000) -> pd.DataFrame:
    """Generate sample validation model output."""
    print("Generating sample Validation Model (Component 2) output...")
    
    statuses = ['ACCEPT', 'CONDITIONAL_ACCEPT', 'REVIEW', 'REJECT']
    anomalies = ['Low', 'Medium', 'High']
    fields = ['patient_demographics', 'medication_details', 'adverse_event_details', 
             'event_outcome', 'medical_history', 'causality_assessment']
    
    return pd.DataFrame({
        'case_id': [f'CASE_{i:06d}' for i in range(n_records)],
        'validation_status': np.random.choice(statuses, n_records, p=[0.35, 0.25, 0.25, 0.15]),
        'quality_score': np.random.beta(8, 2, n_records) * 100,
        'completeness_score': np.random.beta(7, 2, n_records) * 100,
        'missing_fields': [[np.random.choice(fields) for _ in range(np.random.randint(0, 4))] 
                          for _ in range(n_records)],
        'anomaly_risk': np.random.choice(anomalies, n_records, p=[0.6, 0.3, 0.1]),
    })


def test_individual_linkers():
    """Test individual linkers."""
    print("\n" + "="*70)
    print("TEST 1: INDIVIDUAL LINKER TESTS")
    print("="*70 + "\n")
    
    # Test Prioritization → Validation Linker
    print("\n" + "-"*70)
    print("Testing Prioritization → Validation Linker")
    print("-"*70)
    prio_df = generate_sample_prioritization_output(100)
    prio_val_linker = PrioritizationToValidationLinker()
    val_input = prio_val_linker.transform(prio_df)
    print(f"\nInput records: {len(prio_df)}")
    print(f"Output records: {len(val_input)}")
    print(f"Output columns: {list(val_input.columns)}")
    
    # Test Validation → Questionnaire Linker
    print("\n" + "-"*70)
    print("Testing Validation → Questionnaire Linker")
    print("-"*70)
    val_df = generate_sample_validation_output(100)
    val_quest_linker = ValidationToQuestionnaireLinker()
    quest_input = val_quest_linker.transform(val_df)
    print(f"\nInput records: {len(val_df)}")
    print(f"Output records: {len(quest_input)}")
    print(f"Output columns: {list(quest_input.columns)}")


def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline."""
    print("\n" + "="*70)
    print("TEST 2: END-TO-END PIPELINE TEST")
    print("="*70 + "\n")
    
    # Generate inputs
    prio_df = generate_sample_prioritization_output(1000)
    val_df = generate_sample_validation_output(1000)
    
    # Run pipeline
    linker = EndToEndPipelineLinker()
    result = linker.run_pipeline(prio_df, val_df)
    
    # Get results
    questionnaire_input = result['questionnaire_input']
    metrics = result['pipeline_metrics']
    
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*70)
    print(f"\nGenerated questionnaire input for {len(questionnaire_input)} cases")
    print(f"Pipeline processing time: {metrics['pipeline_duration_seconds']:.2f} seconds")
    print(f"Throughput: {metrics['records_per_second']:.2f} records/second")
    
    # Validate schemas
    print("\n" + "-"*70)
    print("Schema Validation")
    print("-"*70)
    validation_results = linker.validate_schemas()
    for check, valid in validation_results.items():
        status = "✓" if valid else "✗"
        print(f"{status} {check}: {valid}")
    
    # Save outputs
    print("\n" + "-"*70)
    print("Saving Outputs")
    print("-"*70)
    linker.save_pipeline_report()
    linker.save_questionnaire_input()
    
    return questionnaire_input, metrics


def print_sample_records(df: pd.DataFrame, n_samples: int = 5):
    """Print sample records from dataframe."""
    print(f"\nSample records (showing {min(n_samples, len(df))} of {len(df)}):")
    print("-" * 70)
    for idx, row in df.head(n_samples).iterrows():
        print(f"\nRecord {idx}:")
        for col, val in row.items():
            if isinstance(val, (list, dict)):
                print(f"  {col}: {str(val)[:60]}...")
            else:
                print(f"  {col}: {val}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PIPELINE LINKER TEST SUITE")
    print("="*70)
    
    # Test individual linkers
    test_individual_linkers()
    
    # Test end-to-end pipeline
    questionnaire_df, metrics = test_end_to_end_pipeline()
    
    # Print sample questionnaire records
    print_sample_records(questionnaire_df, n_samples=3)
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")
