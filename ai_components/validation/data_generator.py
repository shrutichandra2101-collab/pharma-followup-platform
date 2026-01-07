"""
Data generator for synthetic adverse event reports with validation issues.
Creates 10,000 realistic cases with various data quality problems.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import sys
sys.path.append('../..')
from validation_constants import (
    VALID_VALUES, VALUE_RANGES, FIELD_WEIGHTS, VALID_REGIONS
)

class ValidationDataGenerator:
    """Generate synthetic adverse event reports with quality issues."""
    
    def __init__(self, num_samples=10000, error_rate=0.35, seed=42):
        """
        Initialize generator.
        
        Args:
            num_samples: Number of reports to generate
            error_rate: Fraction of reports with at least one error (0-1)
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.error_rate = error_rate
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_clean_report(self):
        """Generate a clean, valid adverse event report."""
        start_date = datetime.now() - timedelta(days=random.randint(1, 365))
        event_date = start_date + timedelta(days=random.randint(1, 30))
        report_date = event_date + timedelta(days=random.randint(0, 7))
        
        patient_age = np.random.randint(18, 85)
        patient_gender = random.choice(VALID_VALUES['patient_gender'][:-1])  # Exclude 'Not Specified'
        
        report = {
            'case_id': f'CASE-{np.random.randint(100000, 999999)}',
            'patient_id': f'PAT-{np.random.randint(10000, 99999)}',
            'patient_age': patient_age,
            'patient_gender': patient_gender,
            'event_date': event_date.strftime('%Y-%m-%d'),
            'drug_name': random.choice(['Aspirin', 'Ibuprofen', 'Paracetamol', 'Metformin', 'Lisinopril', 'Atorvastatin']),
            'dose': round(np.random.uniform(10, 1000), 2),
            'dose_unit': random.choice(['mg', 'mcg', 'g', 'IU']),
            'route': random.choice(VALID_VALUES['route']),
            'start_date': start_date.strftime('%Y-%m-%d'),
            'event_type': random.choice(VALID_VALUES['event_type']),
            'event_description': f'Patient experienced {random.choice(["rash", "headache", "nausea", "dizziness", "fatigue"])}',
            'outcome': random.choice(VALID_VALUES['outcome'][:-1]),  # Exclude 'Unknown'
            'reporter_type': random.choice(VALID_VALUES['reporter_type'][:-1]),  # Exclude 'Unknown'
            'report_date': report_date.strftime('%Y-%m-%d'),
            'reporter_name': f'Dr. {random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones"])}',
            'reporter_contact': f'{np.random.randint(10000000, 99999999)}',
            'causality_assessment': random.choice(VALID_VALUES['causality_assessment']),
            'medical_history': random.choice(['No history', 'Hypertension', 'Diabetes', 'None reported']),
            'concomitant_medications': random.choice(['None', 'Metoprolol', 'Lisinopril', 'Not specified']),
            'hospitalization_flag': random.choice([0, 1]),
            'pregnancy_flag': 1 if patient_gender == 'Female' and patient_age < 50 and random.random() < 0.1 else 0,
            'region': random.choice(VALID_REGIONS),
        }
        
        return report
    
    def introduce_errors(self, report):
        """Introduce realistic validation errors into a report."""
        error_types = []
        
        # Type 1: Missing mandatory field (~15% of errors)
        if random.random() < 0.15:
            field = random.choice(['patient_id', 'drug_name', 'event_date', 'start_date'])
            report[field] = None
            error_types.append('missing_mandatory')
        
        # Type 2: Invalid categorical value (~15% of errors)
        if random.random() < 0.15:
            field = random.choice(['patient_gender', 'route', 'event_type', 'outcome'])
            report[field] = 'INVALID_VALUE_' + field
            error_types.append('invalid_category')
        
        # Type 3: Out of range numeric (~15% of errors)
        if random.random() < 0.15:
            if random.random() < 0.5:
                report['patient_age'] = random.choice([-5, 150, 999])  # Invalid ages
            else:
                report['dose'] = random.choice([-100, 500000])  # Invalid doses
            error_types.append('range_error')
        
        # Type 4: Date logic errors (~15% of errors)
        if random.random() < 0.15:
            if report.get('event_date') and report.get('start_date'):
                try:
                    # Event date before start date
                    event_date = datetime.strptime(report['event_date'], '%Y-%m-%d')
                    start_date = datetime.strptime(report['start_date'], '%Y-%m-%d')
                    if random.random() < 0.5:
                        report['event_date'] = (start_date - timedelta(days=5)).strftime('%Y-%m-%d')
                    else:
                        report['report_date'] = (event_date - timedelta(days=2)).strftime('%Y-%m-%d')
                    error_types.append('date_logic')
                except:
                    pass  # Skip if dates are invalid
        
        # Type 5: Cross-field conflicts (~10% of errors)
        if random.random() < 0.10:
            if report.get('patient_gender') == 'Male' and report.get('pregnancy_flag') == 1:
                error_types.append('gender_pregnancy_conflict')
            else:
                # Outcome = Fatal but hospitalization flag = 0
                if report.get('outcome') == 'Fatal' and report.get('hospitalization_flag') == 0:
                    error_types.append('inconsistent_outcome')
        
        # Type 6: Incomplete data (missing optional fields) (~20% of errors)
        if random.random() < 0.20:
            optional_fields = ['causality_assessment', 'medical_history', 'reporter_contact']
            for field in random.sample(optional_fields, k=random.randint(1, 2)):
                report[field] = None
            error_types.append('missing_optional')
        
        # Type 7: Invalid formats (~10% of errors)
        if random.random() < 0.10:
            if random.random() < 0.5:
                report['patient_age'] = 'Twenty-five'  # Non-numeric
            else:
                report['event_date'] = '2025/13/45'  # Invalid date format
            error_types.append('invalid_format')
        
        return report, error_types
    
    def generate_dataset(self):
        """Generate full dataset with mixed clean and error-containing reports."""
        reports = []
        error_summary = {
            'missing_mandatory': 0,
            'invalid_category': 0,
            'range_error': 0,
            'date_logic': 0,
            'gender_pregnancy_conflict': 0,
            'inconsistent_outcome': 0,
            'missing_optional': 0,
            'invalid_format': 0,
        }
        
        print(f"Generating {self.num_samples} synthetic adverse event reports...")
        print(f"Target error rate: {self.error_rate*100}%\n")
        
        for i in range(self.num_samples):
            # Create base clean report
            report = self.generate_clean_report()
            error_types = []
            
            # Decide if this report should have errors
            if random.random() < self.error_rate:
                report, error_types = self.introduce_errors(report)
            
            # Add metadata
            report['has_errors'] = 1 if error_types else 0
            report['error_count'] = len(error_types)
            report['error_types'] = '|'.join(error_types) if error_types else 'None'
            
            # Track error types
            for error in error_types:
                if error in error_summary:
                    error_summary[error] += 1
            
            reports.append(report)
            
            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1} reports...")
        
        df = pd.DataFrame(reports)
        
        # Print summary
        print("\n" + "="*70)
        print("DATASET SUMMARY")
        print("="*70)
        print(f"Total reports: {len(df)}")
        print(f"Reports with errors: {df['has_errors'].sum()} ({df['has_errors'].sum()/len(df)*100:.1f}%)")
        print(f"Reports without errors: {(1-df['has_errors']).sum()} ({(1-df['has_errors']).sum()/len(df)*100:.1f}%)")
        print(f"Average errors per report: {df['error_count'].mean():.2f}")
        print(f"Max errors in single report: {df['error_count'].max()}")
        
        print("\nERROR TYPE DISTRIBUTION:")
        print("-" * 70)
        for error_type, count in sorted(error_summary.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                pct = count / df[df['has_errors']==1].shape[0] * 100 if df['has_errors'].sum() > 0 else 0
                print(f"  {error_type:30s}: {count:5d} ({pct:5.1f}%)")
        
        print("\nSAMPLE REPORTS:")
        print("-" * 70)
        print("\nClean report (no errors):")
        print(df[df['has_errors']==0].iloc[0] if (df['has_errors']==0).any() else "No clean reports")
        print("\nReport with errors:")
        print(df[df['has_errors']==1].iloc[0] if (df['has_errors']==1).any() else "No error reports")
        
        return df
    
    def save_dataset(self, df, output_path='../../data/processed/validation_reports.csv'):
        """Save generated dataset to CSV."""
        df.to_csv(output_path, index=False)
        print(f"\n✓ Dataset saved to {output_path}")
        return output_path


if __name__ == "__main__":
    generator = ValidationDataGenerator(num_samples=10000, error_rate=0.35)
    df = generator.generate_dataset()
    generator.save_dataset(df)
