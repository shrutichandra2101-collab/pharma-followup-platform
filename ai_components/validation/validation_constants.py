"""
Validation constants and field definitions for adverse event reports.
Follows ICH E2B(R3) standard and pharmacovigilance best practices.
"""

# MANDATORY FIELDS (must be present in all reports)
MANDATORY_FIELDS = {
    'patient_id': 'string',
    'patient_age': 'numeric',
    'patient_gender': 'string',
    'event_date': 'date',
    'drug_name': 'string',
    'dose': 'numeric',
    'dose_unit': 'string',
    'route': 'string',
    'start_date': 'date',
    'event_type': 'string',
    'event_description': 'string',
    'outcome': 'string',
    'reporter_type': 'string',
    'report_date': 'date'
}

# OPTIONAL BUT IMPORTANT FIELDS
OPTIONAL_FIELDS = {
    'concomitant_medications': 'string',
    'medical_history': 'string',
    'causality_assessment': 'string',
    'action_taken': 'string',
    'reporter_name': 'string',
    'reporter_contact': 'string',
    'pregnancy_flag': 'boolean',
    'hospitalization_flag': 'boolean',
}

# VALID VALUES FOR CATEGORICAL FIELDS
VALID_VALUES = {
    'patient_gender': ['Male', 'Female', 'Unknown', 'Not Specified'],
    'route': ['Oral', 'IV', 'IM', 'SC', 'Topical', 'Rectal', 'Inhalation', 'Other'],
    'event_type': [
        'Cardiac Disorders', 'Gastrointestinal Disorders', 'Nervous System Disorders',
        'Skin Disorders', 'Respiratory Disorders', 'Blood Disorders', 
        'Psychiatric Disorders', 'Renal Disorders', 'Hepatic Disorders', 
        'Musculoskeletal Disorders', 'Immune System Disorders', 'Other'
    ],
    'outcome': [
        'Recovered', 'Recovering', 'Not Recovered', 'Fatal', 'Unknown',
        'Recovered with Sequelae'
    ],
    'reporter_type': [
        'Healthcare Professional', 'Patient', 'Pharmacist', 'Other HCP', 'Consumer', 'Unknown'
    ],
    'causality_assessment': [
        'Unrelated', 'Unlikely', 'Possible', 'Probable', 'Definite', 'Unknown'
    ],
    'seriousness_criteria': [
        'Death', 'Life-threatening', 'Hospitalization', 'Disability', 
        'Congenital Anomaly', 'Other Medically Important'
    ]
}

# ACCEPTABLE VALUE RANGES
VALUE_RANGES = {
    'patient_age': (0, 120),  # Age in years
    'dose': (0, 100000),  # Dose in mg/units
    'days_to_deadline': (0, 365),
    'completeness_pct': (0, 100),
}

# COMMON VALIDATION ERRORS
VALIDATION_ERRORS = {
    'missing_mandatory': 'Missing mandatory field: {}',
    'invalid_value': 'Invalid value for {}: {}',
    'date_logic': 'Date logic error: {} must be after {}',
    'range_error': '{} out of acceptable range: {} (expected: {}-{})',
    'invalid_category': 'Invalid category for {}: {} (must be one of: {})',
    'gender_pregnancy_conflict': 'Gender is male but pregnancy flag is set',
    'inconsistent_dates': 'Event date {} is before drug start date {}',
    'invalid_format': 'Invalid format for {}: {}',
    'duplicate_report': 'Potential duplicate report (similar case exists)',
}

# FIELD IMPORTANCE WEIGHTS for completeness scoring
FIELD_WEIGHTS = {
    'patient_id': 1.0,
    'event_date': 1.0,
    'drug_name': 1.0,
    'start_date': 1.0,
    'event_type': 1.0,
    'event_description': 1.0,
    'outcome': 0.9,
    'reporter_type': 0.8,
    'patient_age': 0.8,
    'patient_gender': 0.7,
    'dose': 0.7,
    'route': 0.7,
    'causality_assessment': 0.6,
    'medical_history': 0.5,
    'concomitant_medications': 0.5,
}

# ANOMALY DETECTION THRESHOLDS
ANOMALY_THRESHOLDS = {
    'isolation_forest_contamination': 0.1,  # Expect ~10% anomalies
    'anomaly_score_critical': 0.8,  # Score > 0.8 is highly anomalous
    'anomaly_score_warning': 0.6,  # Score > 0.6 warrants review
    'min_samples_for_training': 100,
}

# QUALITY SCORE INTERPRETATION
QUALITY_SCORE_INTERPRETATION = {
    (0, 20): 'Critical - Severe data quality issues',
    (20, 40): 'Poor - Significant data gaps',
    (40, 60): 'Fair - Moderate data gaps',
    (60, 80): 'Good - Minor data gaps',
    (80, 100): 'Excellent - Complete report'
}

# REGIONS FOR GEOGRAPHICAL VALIDATION
VALID_REGIONS = [
    'North America', 'Europe', 'Asia-Pacific', 'Latin America', 
    'Middle East & Africa', 'Sub-Saharan Africa', 'South Asia', 
    'East Asia', 'Central America', 'Caribbean'
]

# REGULATORY REQUIREMENTS BY REGION
REGULATORY_DEADLINES = {
    'North America': 15,  # Days to report
    'Europe': 15,
    'Japan': 15,
    'Other': 60,
}

# DATA TYPE CHECKS
DATA_TYPES = {
    'string': ['patient_id', 'drug_name', 'event_description', 'reporter_name', 'medical_history'],
    'numeric': ['patient_age', 'dose'],
    'date': ['event_date', 'start_date', 'report_date'],
    'boolean': ['pregnancy_flag', 'hospitalization_flag'],
    'categorical': ['patient_gender', 'route', 'event_type', 'outcome', 'reporter_type']
}
