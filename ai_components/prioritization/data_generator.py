"""Generate synthetic training data for follow-up prioritization."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import sys
sys.path.append('../..')
from utils.common import EVENT_TYPES, SERIOUSNESS_CRITERIA, REGIONS, REPORTER_TYPES

np.random.seed(42)
random.seed(42)

def generate_prioritization_dataset(n_samples=5000):
    """
    Generate synthetic adverse event cases for prioritization model.
    
    Features:
    - Medical severity indicators
    - Data completeness metrics
    - Time-based features
    - Reporter characteristics
    - Regional/regulatory context
    
    Target: Priority score (1-10)
    """
    data = []
    
    for i in range(n_samples):
        case_id = f"CASE-{i+1:06d}"
        
        # Temporal features
        days_since_report = np.random.randint(0, 90)
        days_to_deadline = np.random.randint(1, 45)
        
        # Seriousness (higher score = more serious)
        is_serious = np.random.choice([0, 1], p=[0.6, 0.4])
        if is_serious:
            seriousness_type = np.random.choice(SERIOUSNESS_CRITERIA)
            seriousness_score = {
                "Death": 10,
                "Life-threatening": 9,
                "Hospitalization": 7,
                "Disability": 8,
                "Congenital Anomaly": 8,
                "Other Medically Important": 6
            }[seriousness_type]
        else:
            seriousness_type = "Non-serious"
            seriousness_score = np.random.randint(1, 5)
        
        # Data completeness (0-100%)
        num_mandatory_fields = 10
        fields_complete = np.random.binomial(num_mandatory_fields, 
                                            p=np.random.uniform(0.3, 0.9))
        completeness_pct = (fields_complete / num_mandatory_fields) * 100
        
        # Reporter characteristics
        reporter_type = np.random.choice(REPORTER_TYPES)
        reporter_reliability = {
            "Healthcare Professional": np.random.uniform(0.7, 0.95),
            "Pharmacist": np.random.uniform(0.75, 0.95),
            "Other HCP": np.random.uniform(0.6, 0.85),
            "Patient": np.random.uniform(0.4, 0.7),
            "Consumer": np.random.uniform(0.3, 0.6)
        }[reporter_type]
        
        # Regional features
        region = np.random.choice(REGIONS)
        regulatory_strictness = {
            "North America": 0.9,
            "Europe": 0.95,
            "Asia-Pacific": 0.7,
            "Latin America": 0.6,
            "Middle East": 0.5,
            "Africa": 0.5
        }[region]
        
        # Event characteristics
        event_type = np.random.choice(EVENT_TYPES)
        
        # Previous follow-up attempts
        num_followup_attempts = np.random.poisson(1.5)
        
        # Historical response rate for similar cases
        historical_response_rate = np.random.uniform(0.2, 0.8)
        
        # Calculate priority score (1-10)
        # Formula weights various factors
        priority_score = (
            seriousness_score * 0.35 +                    # Medical severity
            (100 - completeness_pct) / 10 * 0.25 +        # Data gaps
            min(days_since_report / 10, 5) * 0.15 +       # Age of case
            (45 - days_to_deadline) / 5 * 0.15 +          # Deadline urgency
            (1 - reporter_reliability) * 5 * 0.05 +       # Reporter quality
            regulatory_strictness * 2 * 0.05              # Regulatory pressure
        )
        
        # Add some noise
        priority_score += np.random.normal(0, 0.5)
        priority_score = np.clip(priority_score, 1, 10)
        
        # Discretize into priority categories
        if priority_score >= 8:
            priority_category = "Critical"
        elif priority_score >= 6:
            priority_category = "High"
        elif priority_score >= 4:
            priority_category = "Medium"
        else:
            priority_category = "Low"
        
        data.append({
            'case_id': case_id,
            'is_serious': is_serious,
            'seriousness_type': seriousness_type,
            'seriousness_score': seriousness_score,
            'completeness_pct': round(completeness_pct, 2),
            'days_since_report': days_since_report,
            'days_to_deadline': days_to_deadline,
            'reporter_type': reporter_type,
            'reporter_reliability': round(reporter_reliability, 3),
            'region': region,
            'regulatory_strictness': regulatory_strictness,
            'event_type': event_type,
            'num_followup_attempts': num_followup_attempts,
            'historical_response_rate': round(historical_response_rate, 3),
            'priority_score': round(priority_score, 2),
            'priority_category': priority_category
        })
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Generating prioritization training dataset...")
    
    # Generate datasets
    train_df = generate_prioritization_dataset(n_samples=4000)
    test_df = generate_prioritization_dataset(n_samples=1000)
    
    # Save datasets
    train_df.to_csv('../../data/processed/prioritization_train.csv', index=False)
    test_df.to_csv('../../data/processed/prioritization_test.csv', index=False)
    
    print(f"\nTraining set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"\nPriority distribution (training):")
    print(train_df['priority_category'].value_counts().sort_index())
    print(f"\nSample of generated data:")
    print(train_df.head())
    print(f"\nData saved to data/processed/")
