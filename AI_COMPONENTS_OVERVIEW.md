# AI Components Overview

Complete list of AI/ML models for the Pharmacovigilance Follow-up Platform.

---

## 1. Follow-up Prioritization Engine ✅ 

**Status:** Implemented

### Purpose
Automatically rank adverse event cases by urgency to optimize follow-up resource allocation.

### Models
1. **Regression Model**: Predicts continuous priority score (1-10)
2. **Classification Model**: Categorizes into Low/Medium/High/Critical

### Training Data
- **Size:** 4,000 training samples, 1,000 test samples
- **Features (13 total):**
  - Medical: seriousness score, event type, serious/non-serious flag
  - Data quality: completeness percentage (% mandatory fields filled)
  - Temporal: days since report, days to regulatory deadline
  - Reporter: type (HCP/Patient/Pharmacist), reliability score
  - Context: region, regulatory strictness, historical response rate
  - History: number of previous follow-up attempts

### Target Variable
- **Priority Score:** 1-10 (weighted formula based on medical severity, data gaps, urgency)
- **Priority Category:** Low (<4), Medium (4-6), High (6-8), Critical (≥8)

### Performance Metrics
- **Regression:** RMSE, MAE, R² score
- **Classification:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Interpretability:** Feature importance (weight-based from XGBoost)

### Technology
- XGBoost (regression + multi-class classification)
- scikit-learn for preprocessing

### Output Files
- `data/processed/prioritization_train.csv` - Training dataset
- `data/processed/prioritization_test.csv` - Test dataset
- `data/models/prioritization_regression.json` - Trained regression model
- `data/models/prioritization_classification.json` - Trained classification model
- `data/models/prioritization_encoders.pkl` - Label encoders
- `evaluation/prioritization_metrics.json` - Performance metrics
- `evaluation/prioritization_regression.png` - Actual vs predicted plot
- `evaluation/prioritization_classification_confusion_matrix.png`
- `evaluation/prioritization_feature_importance.png`

---

## 2. Data Validation & Gap Detection Engine

**Status:** To be implemented

### Purpose
Automatically identify missing, inconsistent, or anomalous information in adverse event reports before human review.

### Components

#### 2.1 Rule-Based Validator
- **ICH E2B Compliance Check:** Validate mandatory fields per regulatory standards
- **Cross-field Consistency:** Check logical relationships (e.g., pregnancy + male gender = flag)
- **Date Validation:** Event date after drug start date, timeline consistency
- **Value Range Check:** Age 0-120, dosage within therapeutic range

#### 2.2 Anomaly Detection Model
- **Algorithm:** Isolation Forest / One-Class SVM
- **Purpose:** Detect unusual patterns (e.g., extreme dosages, rare event combinations)

#### 2.3 Completeness Scoring
- **Formula-based:** Calculate % of mandatory vs optional fields filled
- **Quality score:** Weight fields by importance (e.g., causality assessment > reporter phone)

### Training Data
- **Size:** 10,000 adverse event reports (mix of complete and incomplete)
- **Features:**
  - Boolean flags for each mandatory field (present/missing)
  - Field value distributions
  - Cross-field relationship features
  - Historical data quality by reporter type

### Target Variable
- **Validation Status:** Pass/Fail with specific error codes
- **Anomaly Score:** 0-1 (higher = more anomalous)
- **Missing Fields:** List of required fields needing follow-up

### Performance Metrics
- **Validation:** Precision/Recall for catching true data issues
- **Anomaly Detection:** AUC-ROC, precision at k%
- **False Positive Rate:** Critical for user trust

### Technology
- Pandas + Pydantic for rule-based validation
- scikit-learn Isolation Forest for anomaly detection
- Custom business logic for pharmacovigilance-specific rules

---

## 3. Medical NER (Named Entity Recognition)

**Status:** To be implemented

### Purpose
Extract structured information from free-text narrative fields to auto-populate missing data.

### Entities to Extract
1. **Medications:** Drug names, brand names, generic names
2. **Dosages:** Numeric dose + unit (e.g., "100 mg")
3. **Routes of Administration:** Oral, IV, topical, etc.
4. **Medical Conditions:** Diseases, symptoms, adverse events
5. **Temporal Expressions:** Dates, durations ("3 days after starting")
6. **Patient Demographics:** Age references, gender mentions
7. **Outcomes:** Recovered, recovering, death, hospitalization

### Model
- **Base Model:** BioBERT / ClinicalBERT (pre-trained on medical text)
- **Fine-tuning:** Custom labeled dataset with pharma adverse events
- **Architecture:** Transformer-based token classification

### Training Data
- **Size:** 5,000 adverse event narratives with entity annotations
- **Format:** BIO tagging (Beginning, Inside, Outside)
- **Example:**
  ```
  "Patient took aspirin 100mg orally and developed rash"
  Patient    O
  took       O
  aspirin    B-DRUG
  100mg      B-DOSE
  orally     B-ROUTE
  and        O
  developed  O
  rash       B-EVENT
  ```

### Target Variable
- **Token labels:** B-DRUG, I-DRUG, B-DOSE, B-ROUTE, B-EVENT, etc.

### Performance Metrics
- **Entity-level:** Precision, Recall, F1 per entity type
- **Token-level:** Overall accuracy
- **Strict vs Relaxed Matching:** Exact span vs partial overlap

### Technology
- Hugging Face Transformers (AutoModelForTokenClassification)
- Pre-trained: `emilyalsentzer/Bio_ClinicalBERT`
- PyTorch for training

---

## 4. Predictive Response Model

**Status:** To be implemented

### Purpose
Predict likelihood that a healthcare professional or patient will respond to a follow-up request.

### Use Case
Optimize outreach strategy (email vs phone, timing, persistence level).

### Model
- **Type:** Binary classification (Will Respond / Won't Respond)
- **Algorithm:** Logistic Regression / Random Forest / LightGBM

### Training Data
- **Size:** 15,000 historical follow-up attempts with outcomes
- **Features:**
  - **Reporter characteristics:** Type (HCP/patient), previous response rate, country
  - **Case characteristics:** Seriousness, completeness at time of request
  - **Communication features:** Channel used (email/phone/portal), time of day, day of week
  - **Temporal:** Days since initial report, number of prior attempts
  - **Contextual:** Language barrier, regulatory environment

### Target Variable
- **Response:** 1 (responded within 14 days) / 0 (no response)

### Performance Metrics
- **Primary:** AUC-ROC, Precision-Recall curve
- **Business Metrics:** 
  - Precision @ top 20% (focus on high-probability responders)
  - Recall for serious cases
- **Calibration:** Reliability diagram (predicted prob vs actual response rate)

### Technology
- scikit-learn / LightGBM
- SMOTE for handling class imbalance if needed

---

## 5. Multilingual Translation Pipeline

**Status:** To be implemented

### Purpose
Enable seamless communication across 30+ languages for global operations.

### Components

#### 5.1 Language Detection
- **Library:** `langdetect` or FastText language identification
- **Purpose:** Auto-detect input language from reports

#### 5.2 Translation Engine
- **API:** Google Cloud Translation API / Azure Translator
- **Specialty:** Medical terminology dictionaries for accuracy
- **Direction:** Bi-directional (local language ↔ English)

#### 5.3 Medical Terminology Preservation
- **Technique:** Named entity masking before translation
- **Process:**
  1. Extract drug names, medical terms with NER
  2. Replace with placeholders (e.g., `<DRUG_1>`)
  3. Translate surrounding text
  4. Re-insert original medical terms

#### 5.4 Quality Assurance
- **Back-translation:** Translate result back to source language, compare
- **Confidence scores:** Flag low-confidence translations for human review
- **Glossary enforcement:** Maintain approved translations for key terms

### Training/Configuration Data
- **Language pairs:** 30+ languages × English
- **Medical glossaries:** 10,000+ term pairs per language (drug names, anatomical terms)
- **Validation set:** 1,000 professionally translated adverse event texts

### Performance Metrics
- **BLEU Score:** Automated translation quality metric
- **Human Evaluation:** Fluency + adequacy ratings from native speakers
- **Medical Accuracy:** % of drug names / terms correctly preserved
- **Latency:** Translation time per 1000 characters

### Technology
- Google Cloud Translation API (Medical domain specialization)
- spaCy for entity masking
- Custom glossary management system

---

## Training Pipeline Summary

| Component | Data Size | Model Type | Training Time | Key Metric |
|-----------|-----------|------------|---------------|------------|
| 1. Prioritization | 5K cases | XGBoost | ~2 min | R² = 0.85+ |
| 2. Validation | 10K reports | Rule+Isolation Forest | ~5 min | FPR < 5% |
| 3. Medical NER | 5K narratives | BioBERT fine-tune | ~2 hours (GPU) | F1 = 0.90+ |
| 4. Response Prediction | 15K attempts | LightGBM | ~3 min | AUC = 0.75+ |
| 5. Translation | N/A (API) | Cloud API | Real-time | BLEU > 0.40 |

---

## Integration Architecture

```
┌─────────────────────────────────────────────────┐
│           User Input (New AE Report)            │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────┐
    │  1. Data Validation & Gap Detection│
    │     - Check mandatory fields       │
    │     - Flag inconsistencies         │
    │     - Calculate completeness       │
    └──────────────┬────────────────────┘
                   │
                   ▼
    ┌───────────────────────────────────┐
    │  2. Medical NER (if free text)    │
    │     - Extract entities             │
    │     - Auto-populate fields         │
    └──────────────┬────────────────────┘
                   │
                   ▼
    ┌───────────────────────────────────┐
    │  3. Follow-up Prioritization      │
    │     - Calculate priority score     │
    │     - Assign category              │
    └──────────────┬────────────────────┘
                   │
                   ▼
    ┌───────────────────────────────────┐
    │  4. Response Prediction           │
    │     - Estimate response likelihood │
    │     - Suggest contact strategy     │
    └──────────────┬────────────────────┘
                   │
                   ▼
    ┌───────────────────────────────────┐
    │  5. Translation (if non-English)  │
    │     - Translate follow-up message  │
    │     - Preserve medical terms       │
    └──────────────┬────────────────────┘
                   │
                   ▼
    ┌───────────────────────────────────┐
    │    Send Follow-up Request         │
    └───────────────────────────────────┘
```

---

## Deployment Considerations

### Model Serving
- **REST API:** FastAPI endpoints for each model
- **Batch Processing:** Celery + Redis for bulk prioritization
- **Real-time:** Low-latency inference (<100ms) for validation

### Monitoring
- **Model Drift:** Track prediction distributions over time
- **Data Quality:** Monitor input feature distributions
- **Business Metrics:** Response rates, time to closure
- **A/B Testing:** Compare model-driven vs manual prioritization

### Compliance
- **HIPAA/GDPR:** All patient data encrypted at rest and in transit
- **Audit Trails:** Log all model predictions with timestamps
- **Explainability:** Feature importance + SHAP values for high-stakes decisions
- **Human-in-the-loop:** Final approval for critical cases

---

## Next Steps

1. ✅ **Prioritization Engine** - Complete
2. **Validation Engine** - Implement rule-based + anomaly detection
3. **Medical NER** - Prepare annotated dataset, fine-tune BioBERT
4. **Response Prediction** - Generate training data, train classifier
5. **Unified Dashboard** - Streamlit app to visualize all model metrics
6. **API Development** - FastAPI endpoints for production use
7. **Testing & Validation** - End-to-end testing with synthetic cases
8. **Documentation** - API docs, deployment guide

Would you like to proceed with building the next component?
