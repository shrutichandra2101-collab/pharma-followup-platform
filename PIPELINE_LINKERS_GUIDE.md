# Pipeline Linkers: Component Integration Guide

## Overview

Pipeline Linkers are data transformation modules that connect the outputs of upstream AI components to the inputs of downstream components. This document describes the linkers that integrate Components 1, 2, and 3.

**Architecture:**
```
Component 1: Prioritization Model
           ↓
    LINKER 1 (P→V)
           ↓
Component 2: Validation Model
           ↓
    LINKER 2 (V→Q)
           ↓
Component 3: Questionnaire Generator
```

## Linker 1: Prioritization → Validation

**File:** `ai_components/linkers/prioritization_to_validation.py`

**Purpose:** Transform prioritization model outputs into validation model inputs

**Input Schema (from Component 1):**
```
- case_id: str
- priority_score: float (0-1)
- follow_up_urgency: float (0-1)
- estimated_response_time_hours: float
- reporter_reliability: float (0-1)
- regional_significance: float (0-1)
- regulatory_deadline: date
```

**Output Schema (to Component 2):**
```
- case_id: str
- priority_score: float
- follow_up_urgency: float
- estimated_response_time_hours: float
- reporter_reliability: float
- regional_significance: float
- regulatory_deadline: date
- prioritization_score: float (mapped from priority_score)
- requires_urgent_review: int (0/1)
- validation_category: str (CRITICAL/HIGH/MEDIUM/LOW)
- validation_urgency: float (0-1)
- source_component: str ('prioritization_model')
- linker_timestamp: datetime
- linker_version: str
```

**Transformations:**
1. **Categorization:** Priority score mapped to validation urgency categories
   - ≥0.8: CRITICAL
   - 0.6-0.8: HIGH
   - 0.4-0.6: MEDIUM
   - <0.4: LOW

2. **Urgency Calculation:** Combined score from priority (60%) + urgency (40%)

3. **Metadata Addition:** Tracking fields for audit trail

**Usage:**
```python
from linkers.prioritization_to_validation import PrioritizationToValidationLinker

linker = PrioritizationToValidationLinker()
validation_input = linker.transform(prioritization_output_df)
```

---

## Linker 2: Validation → Questionnaire

**File:** `ai_components/linkers/validation_to_questionnaire.py`

**Purpose:** Transform validation model outputs into questionnaire generator inputs

**Input Schema (from Component 2):**
```
- case_id: str
- validation_status: str (ACCEPT/CONDITIONAL_ACCEPT/REVIEW/REJECT)
- quality_score: float (0-100)
- completeness_score: float (0-100)
- missing_fields: list[str]
- anomaly_risk: str (Low/Medium/High)
```

**Output Schema (to Component 3):**
```
- case_id: str
- validation_status: str
- quality_score: float (0-100)
- completeness_score: float (0-100)
- missing_fields: list[str]
- anomaly_risk: str (Low/Medium/High)
- anomaly_risk_numeric: int (0/1/2)
- questionnaire_type: str (quick_verification/targeted_followup/comprehensive_followup/detailed_investigation)
- fields_by_category: dict (categorized missing fields)
- expected_difficulty: str (Easy/Medium/Hard)
- expected_completion_minutes: float
- questionnaire_priority: float (0-1)
- source_component: str ('validation_model')
- linker_timestamp: datetime
- linker_version: str
```

**Transformations:**

1. **Questionnaire Type Mapping:**
   - ACCEPT → quick_verification (minimal follow-up needed)
   - CONDITIONAL_ACCEPT → targeted_followup (focused questions)
   - REVIEW → comprehensive_followup (detailed investigation)
   - REJECT → detailed_investigation (extensive follow-up)

2. **Field Categorization:** Missing fields grouped by question category
   - Safety (adverse event details, severity)
   - Efficacy (event outcome, recovery status)
   - Patient (demographics, hospitalization)
   - Medication (dosage, route, concomitant medications)
   - Medical History (relevant past conditions)
   - Causality (assessment information)

3. **Difficulty Estimation:**
   - Easy: 0-2 missing fields, high quality
   - Medium: 2-4 missing fields, moderate quality
   - Hard: 4+ missing fields or anomalies

4. **Completion Time Estimate:**
   - Base: 3 minutes
   - Per field: +1.5 minutes
   - Anomaly multiplier: 1.0 + (anomaly_risk × 0.3)
   - Max cap: 20 minutes

5. **Priority Calculation:**
   - Critical gaps (completeness < 50): +3
   - Quality issues (quality < 60): +2
   - High anomaly risk: +2

**Usage:**
```python
from linkers.validation_to_questionnaire import ValidationToQuestionnaireLinker

linker = ValidationToQuestionnaireLinker()
questionnaire_input = linker.transform(validation_output_df)
```

---

## End-to-End Pipeline Linker

**File:** `ai_components/linkers/end_to_end_linker.py`

**Purpose:** Orchestrate complete data flow through all three components

**Methods:**

### `run_pipeline(prioritization_df, validation_df)`
Executes complete pipeline with both linkers in sequence.

**Parameters:**
- `prioritization_df`: Output from Component 1 (Prioritization)
- `validation_df`: Output from Component 2 (Validation)

**Returns:**
```python
{
    'questionnaire_input': pd.DataFrame,  # Final input for Component 3
    'pipeline_metrics': dict,              # Processing statistics
    'validation_input': pd.DataFrame,      # Intermediate output from Linker 1
    'stages_completed': int                # Number of stages (2)
}
```

**Pipeline Metrics:**
- `total_records_processed`: Total cases through pipeline
- `pipeline_duration_seconds`: Total execution time
- `records_per_second`: Throughput
- `avg_quality_score`: Average quality (0-100)
- `avg_completeness`: Average completeness (0-100)
- `validation_status_distribution`: Counts by status
- `questionnaire_type_distribution`: Counts by type
- `difficulty_distribution`: Counts by difficulty
- `avg_expected_completion_minutes`: Mean completion time

### `save_pipeline_report(output_dir)`
Saves pipeline execution report to JSON.

**Returns:** Path to saved report

**Output file:** `evaluation/pipeline_linker_report.json`

### `save_questionnaire_input(output_dir)`
Saves questionnaire input to CSV.

**Returns:** Path to saved file

**Output file:** `data/processed/pipeline_questionnaire_input.csv`

### `validate_schemas()`
Validates that outputs match expected schemas.

**Returns:**
```python
{
    'stage_1_schema_valid': bool,
    'stage_2_schema_valid': bool,
    'pipeline_valid': bool
}
```

**Usage:**
```python
from linkers.end_to_end_linker import EndToEndPipelineLinker

linker = EndToEndPipelineLinker()
result = linker.run_pipeline(prio_df, val_df)
questionnaire_input = result['questionnaire_input']
metrics = result['pipeline_metrics']

linker.save_pipeline_report()
linker.save_questionnaire_input()
```

---

## Test Script

**File:** `ai_components/linkers/test_linkers.py`

Comprehensive test suite with:
- Individual linker tests
- End-to-end pipeline test
- Schema validation
- Sample data generation

**Run tests:**
```bash
cd ai_components
python linkers/test_linkers.py
```

**Test output includes:**
- Record transformation verification
- Schema compliance
- Pipeline performance metrics
- Sample records from each stage

---

## Data Flow Example

**Stage 1: Prioritization → Validation**

Input (1,000 cases):
```
case_id: CASE_000001
priority_score: 0.75
follow_up_urgency: 0.68
reporter_reliability: 0.82
...
```

Output after Linker 1:
```
case_id: CASE_000001
priority_score: 0.75
follow_up_urgency: 0.68
reporter_reliability: 0.82
prioritization_score: 0.75
requires_urgent_review: 1
validation_category: HIGH
validation_urgency: 0.72
...
```

**Stage 2: Validation → Questionnaire**

Input:
```
case_id: CASE_000001
validation_status: CONDITIONAL_ACCEPT
quality_score: 72.5
completeness_score: 68.3
missing_fields: ['medication_details', 'dosage_info']
anomaly_risk: Medium
```

Output after Linker 2:
```
case_id: CASE_000001
validation_status: CONDITIONAL_ACCEPT
quality_score: 72.5
completeness_score: 68.3
missing_fields: ['medication_details', 'dosage_info']
anomaly_risk: Medium
anomaly_risk_numeric: 1
questionnaire_type: targeted_followup
fields_by_category: {'Medication': ['medication_details', 'dosage_info']}
expected_difficulty: Medium
expected_completion_minutes: 8.2
questionnaire_priority: 0.18
```

---

## Performance Characteristics

**Throughput:** ~30,000+ records/second per linker

**Processing Time:** <50ms for 1,000 records

**Memory Usage:** Minimal (in-memory transformations)

**Scalability:** Tested with 1,000+ records; linear time complexity

---

## Integration Points

### Upstream: Component 1 Output
Prioritization model provides:
- Priority scores for each case
- Follow-up urgency estimates
- Reporter reliability metrics
- Regional significance assessments
- Regulatory deadlines

### Downstream: Component 3 Input
Questionnaire generator receives:
- Validation status and quality metrics
- Missing field information
- Case difficulty and priority
- Expected completion times
- Anomaly risk assessment

### Feedback Loops (Future)
- Component 3 can report questionnaire effectiveness back to Component 2
- Component 2 validation improvements can be fed back to Component 1
- Closed-loop learning and model improvement

---

## Error Handling

**Missing Fields:** Linkers automatically generate synthetic data for missing columns with appropriate distributions

**Schema Validation:** `validate_schemas()` method checks output compliance

**Data Type Coercion:** Automatic conversion where possible (dates, numerics)

**Logging:** Detailed transformation logs at each stage

---

## Version History

**Linker Version 1.0** (Jan 7, 2026)
- Initial release
- Support for 1-stage and 2-stage pipelines
- Comprehensive schema validation
- Performance metrics collection

---

## Future Enhancements

1. **Adaptive Linkers:** Learn optimal transformations from data
2. **Feedback Integration:** Incorporate downstream model feedback
3. **Multi-path Routing:** Different paths based on case characteristics
4. **Real-time Streaming:** Support streaming data pipelines
5. **Model Versioning:** Track linker version with model versions
6. **Data Lineage:** Full audit trail of transformations

---

## Contact & Support

For linker questions or integration issues:
- Check test_linkers.py for examples
- Review schema definitions in each linker class
- Validate with `validate_schemas()` method
