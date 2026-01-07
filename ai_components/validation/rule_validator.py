"""
Rule-based validator for adverse event reports.
Implements ICH E2B(R3) compliance checks and domain-specific validation rules.
"""

import pandas as pd
from datetime import datetime
import sys
sys.path.append('../..')
from validation_constants import (
    MANDATORY_FIELDS, OPTIONAL_FIELDS, VALID_VALUES, VALUE_RANGES,
    VALIDATION_ERRORS, FIELD_WEIGHTS
)


class RuleBasedValidator:
    """Validate adverse event reports against defined business rules."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def validate_report(self, report):
        """
        Validate a single report against all rules.
        
        Args:
            report: Dictionary or Series representing one adverse event
            
        Returns:
            dict with keys: is_valid, error_count, errors, warnings, quality_score
        """
        self.errors = []
        self.warnings = []
        
        # Run all validation checks
        self._check_mandatory_fields(report)
        self._check_data_types(report)
        self._check_categorical_values(report)
        self._check_numeric_ranges(report)
        self._check_date_logic(report)
        self._check_cross_field_logic(report)
        
        is_valid = len(self.errors) == 0
        quality_score = self._calculate_quality_score(report)
        
        return {
            'is_valid': is_valid,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings,
            'quality_score': quality_score,
        }
    
    def _check_mandatory_fields(self, report):
        """Check that all mandatory fields are present and not null."""
        for field in MANDATORY_FIELDS.keys():
            if field not in report or pd.isna(report.get(field)) or report.get(field) == '':
                self.errors.append({
                    'type': 'missing_mandatory',
                    'field': field,
                    'message': VALIDATION_ERRORS['missing_mandatory'].format(field)
                })
    
    def _check_data_types(self, report):
        """Check that field values match expected data types."""
        type_checks = {
            'string': ['patient_id', 'drug_name', 'event_description'],
            'numeric': ['patient_age', 'dose'],
            'date': ['event_date', 'start_date', 'report_date'],
            'boolean': ['hospitalization_flag', 'pregnancy_flag'],
        }
        
        for dtype, fields in type_checks.items():
            for field in fields:
                if field not in report or pd.isna(report.get(field)):
                    continue
                
                value = report[field]
                is_valid = False
                
                try:
                    if dtype == 'string':
                        is_valid = isinstance(value, str)
                    elif dtype == 'numeric':
                        float(value)
                        is_valid = True
                    elif dtype == 'date':
                        datetime.strptime(str(value), '%Y-%m-%d')
                        is_valid = True
                    elif dtype == 'boolean':
                        is_valid = value in [0, 1, True, False]
                except:
                    is_valid = False
                
                if not is_valid:
                    self.errors.append({
                        'type': 'invalid_format',
                        'field': field,
                        'value': value,
                        'expected_type': dtype,
                        'message': VALIDATION_ERRORS['invalid_format'].format(field, value)
                    })
    
    def _check_categorical_values(self, report):
        """Check that categorical fields have valid values."""
        for field, valid_vals in VALID_VALUES.items():
            if field not in report or pd.isna(report.get(field)):
                continue
            
            value = report[field]
            if value not in valid_vals:
                valid_str = ', '.join(valid_vals[:3]) + f'... (+{len(valid_vals)-3} more)'
                self.errors.append({
                    'type': 'invalid_category',
                    'field': field,
                    'value': value,
                    'valid_values': valid_vals,
                    'message': VALIDATION_ERRORS['invalid_category'].format(field, value, valid_str)
                })
    
    def _check_numeric_ranges(self, report):
        """Check that numeric fields fall within acceptable ranges."""
        for field, (min_val, max_val) in VALUE_RANGES.items():
            if field not in report or pd.isna(report.get(field)):
                continue
            
            try:
                value = float(report[field])
                if not (min_val <= value <= max_val):
                    self.errors.append({
                        'type': 'range_error',
                        'field': field,
                        'value': value,
                        'min': min_val,
                        'max': max_val,
                        'message': VALIDATION_ERRORS['range_error'].format(
                            field, value, min_val, max_val
                        )
                    })
            except (ValueError, TypeError):
                pass  # Already caught by data type check
    
    def _check_date_logic(self, report):
        """Check temporal consistency of dates."""
        date_fields = ['start_date', 'event_date', 'report_date']
        dates = {}
        
        # Parse dates
        for field in date_fields:
            if field in report and not pd.isna(report[field]):
                try:
                    dates[field] = datetime.strptime(str(report[field]), '%Y-%m-%d')
                except:
                    pass
        
        # Check: event_date must be >= start_date
        if 'start_date' in dates and 'event_date' in dates:
            if dates['event_date'] < dates['start_date']:
                self.errors.append({
                    'type': 'date_logic',
                    'fields': ['start_date', 'event_date'],
                    'message': VALIDATION_ERRORS['inconsistent_dates'].format(
                        dates['event_date'].strftime('%Y-%m-%d'),
                        dates['start_date'].strftime('%Y-%m-%d')
                    )
                })
        
        # Check: report_date must be >= event_date
        if 'event_date' in dates and 'report_date' in dates:
            if dates['report_date'] < dates['event_date']:
                self.errors.append({
                    'type': 'date_logic',
                    'fields': ['event_date', 'report_date'],
                    'message': 'Report date cannot be before event date'
                })
        
        # Check: all dates should be in past or recent
        if 'report_date' in dates:
            if dates['report_date'] > datetime.now():
                self.warnings.append({
                    'type': 'future_date',
                    'field': 'report_date',
                    'message': 'Report date is in the future'
                })
    
    def _check_cross_field_logic(self, report):
        """Check logical consistency across related fields."""
        # Gender-Pregnancy logic
        if ('patient_gender' in report and 'pregnancy_flag' in report and
            report.get('patient_gender') == 'Male' and report.get('pregnancy_flag') == 1):
            self.errors.append({
                'type': 'gender_pregnancy_conflict',
                'fields': ['patient_gender', 'pregnancy_flag'],
                'message': VALIDATION_ERRORS['gender_pregnancy_conflict']
            })
        
        # Age-Pregnancy logic
        if ('patient_age' in report and 'pregnancy_flag' in report and
            report.get('pregnancy_flag') == 1):
            try:
                age = float(report['patient_age'])
                if age < 12 or age > 55:
                    self.warnings.append({
                        'type': 'age_pregnancy_conflict',
                        'fields': ['patient_age', 'pregnancy_flag'],
                        'message': f'Pregnancy flagged but age is {age} (unusual)'
                    })
            except:
                pass
        
        # Serious outcome flags
        if report.get('outcome') == 'Fatal':
            if report.get('hospitalization_flag') != 1:
                self.warnings.append({
                    'type': 'serious_outcome_flag',
                    'fields': ['outcome', 'hospitalization_flag'],
                    'message': 'Outcome is Fatal but hospitalization flag not set'
                })
        
        # Event without causality assessment
        if (report.get('event_type') and 
            ('causality_assessment' not in report or pd.isna(report.get('causality_assessment')))):
            self.warnings.append({
                'type': 'missing_causality',
                'field': 'causality_assessment',
                'message': 'Event reported but causality assessment not provided'
            })
    
    def _calculate_quality_score(self, report):
        """
        Calculate data quality score (0-100) based on:
        - Presence of mandatory fields
        - Presence of important optional fields
        - Absence of errors
        - Field value completeness
        """
        total_weight = 0
        filled_weight = 0
        
        # Score mandatory fields (weight = 1.0)
        for field in MANDATORY_FIELDS.keys():
            total_weight += FIELD_WEIGHTS.get(field, 0.5)
            if field in report and not pd.isna(report.get(field)) and report.get(field) != '':
                filled_weight += FIELD_WEIGHTS.get(field, 0.5)
        
        # Score optional fields (weight = 0.5)
        for field in OPTIONAL_FIELDS.keys():
            field_weight = FIELD_WEIGHTS.get(field, 0.3)
            total_weight += field_weight
            if field in report and not pd.isna(report.get(field)) and report.get(field) != '':
                filled_weight += field_weight
        
        # Base quality score
        if total_weight > 0:
            base_score = (filled_weight / total_weight) * 100
        else:
            base_score = 0
        
        # Deduct points for errors
        error_deduction = len(self.errors) * 5  # -5 per error
        quality_score = max(0, base_score - error_deduction)
        
        return round(quality_score, 2)


class BatchValidator:
    """Validate entire datasets of adverse event reports."""
    
    def __init__(self):
        self.validator = RuleBasedValidator()
    
    def validate_dataset(self, df):
        """
        Validate all reports in a dataset.
        
        Args:
            df: DataFrame with reports
            
        Returns:
            DataFrame with validation results
        """
        print(f"Validating {len(df)} reports...")
        
        results = []
        for idx, row in df.iterrows():
            result = self.validator.validate_report(row)
            result['case_id'] = row.get('case_id', f'CASE-{idx}')
            results.append(result)
            
            if (idx + 1) % 1000 == 0:
                valid_count = sum(1 for r in results if r['is_valid'])
                print(f"  Processed {idx + 1} reports | Valid: {valid_count} ({valid_count/(idx+1)*100:.1f}%)")
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def get_summary_report(self, results_df, original_df=None):
        """Generate summary statistics from validation results."""
        valid_count = results_df['is_valid'].sum()
        invalid_count = len(results_df) - valid_count
        
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"Total reports: {len(results_df)}")
        print(f"Valid reports: {valid_count} ({valid_count/len(results_df)*100:.1f}%)")
        print(f"Invalid reports: {invalid_count} ({invalid_count/len(results_df)*100:.1f}%)")
        print(f"Average quality score: {results_df['quality_score'].mean():.2f}/100")
        print(f"Average errors per report: {results_df['error_count'].mean():.2f}")
        print(f"Average warnings per report: {results_df['warning_count'].mean():.2f}")
        
        print("\nQUALITY SCORE DISTRIBUTION:")
        print("-" * 70)
        for score_range, label in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
            count = ((results_df['quality_score'] >= score_range) & 
                    (results_df['quality_score'] < score_range + 20)).sum()
            print(f"  {score_range:2d}-{score_range+20:2d}: {count:5d} ({count/len(results_df)*100:5.1f}%)")
        
        return {
            'valid_count': valid_count,
            'invalid_count': invalid_count,
            'avg_quality_score': results_df['quality_score'].mean(),
            'avg_errors': results_df['error_count'].mean(),
            'avg_warnings': results_df['warning_count'].mean(),
        }


if __name__ == "__main__":
    # Test single report validation
    test_report = {
        'patient_id': 'PAT-12345',
        'patient_age': 45,
        'patient_gender': 'Female',
        'event_date': '2025-12-15',
        'drug_name': 'Aspirin',
        'dose': 500,
        'dose_unit': 'mg',
        'route': 'Oral',
        'start_date': '2025-12-01',
        'event_type': 'Gastrointestinal Disorders',
        'event_description': 'Stomach pain',
        'outcome': 'Recovered',
        'reporter_type': 'Healthcare Professional',
        'report_date': '2025-12-16',
    }
    
    validator = RuleBasedValidator()
    result = validator.validate_report(test_report)
    print("Single report validation result:")
    print(result)
