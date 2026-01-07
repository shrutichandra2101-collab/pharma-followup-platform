# AI Components - Quick Reference

## Complete List of AI/ML Models

### 1. 🎯 Follow-up Prioritization Engine
**What it does:** Ranks adverse event cases by urgency (1-10 score + Low/Medium/High/Critical category)  
**Training data:** 5,000 synthetic adverse event cases  
**Model:** XGBoost (regression + classification)  
**Key metrics:** R², RMSE, Classification Accuracy  
**Training time:** ~2 minutes  
**Status:** ✅ **IMPLEMENTED**

---

### 2. ✅ Data Validation & Gap Detection
**What it does:** Identifies missing/inconsistent data + detects anomalies  
**Training data:** 10,000 reports with validation errors  
**Model:** Rule-based validator + Isolation Forest  
**Key metrics:** Precision, Recall, False Positive Rate  
**Training time:** ~5 minutes  
**Status:** 🔨 To be built

---

### 3. 🔍 Medical Named Entity Recognition (NER)
**What it does:** Extracts drugs, dosages, events from free text narratives  
**Training data:** 5,000 annotated medical narratives  
**Model:** Fine-tuned BioBERT/ClinicalBERT  
**Key metrics:** Entity-level F1 score per type (DRUG, DOSE, EVENT, etc.)  
**Training time:** ~2 hours (requires GPU)  
**Status:** 🔨 To be built

---

### 4. 📊 Response Prediction Model
**What it does:** Predicts if HCP/patient will respond to follow-up (yes/no probability)  
**Training data:** 15,000 historical follow-up attempts  
**Model:** LightGBM binary classifier  
**Key metrics:** AUC-ROC, Precision-Recall curve, Calibration  
**Training time:** ~3 minutes  
**Status:** 🔨 To be built

---

### 5. 🌍 Multilingual Translation Pipeline
**What it does:** Translates follow-up messages across 30+ languages while preserving medical terms  
**Training data:** Translation API + 10,000-term medical glossaries per language  
**Model:** Google Cloud Translation API + entity masking  
**Key metrics:** BLEU score, Medical term preservation accuracy  
**Training time:** N/A (API-based)  
**Status:** 🔨 To be built

---

## Quick Comparison Table

| # | Component | Input | Output | Purpose |
|---|-----------|-------|--------|---------|
| 1 | **Prioritization** | Case features (severity, completeness, timing) | Priority score 1-10 + category | Triage cases for follow-up |
| 2 | **Validation** | Adverse event report fields | List of errors/missing data | Ensure data quality |
| 3 | **Medical NER** | Free-text narrative | Extracted entities (drugs, events, doses) | Auto-populate structured fields |
| 4 | **Response Prediction** | Reporter profile + case context | Response probability 0-1 | Optimize outreach strategy |
| 5 | **Translation** | Text in any language | Translated text + preserved medical terms | Enable global communication |

---

## Training Data Requirements

| Component | Samples | Labels/Targets | Generation Method |
|-----------|---------|----------------|-------------------|
| Prioritization | 5,000 | Continuous score 1-10 + 4 categories | Synthetic formula-based |
| Validation | 10,000 | Pass/Fail + error types | Synthetic with deliberate errors |
| Medical NER | 5,000 | BIO tags per token | Synthetic narratives + rule-based tagging |
| Response Prediction | 15,000 | Binary (responded/not) | Synthetic with probabilistic rules |
| Translation | N/A | N/A | Pre-trained API |

---

## How to Build Each Component

### Component 1: Prioritization ✅ (Done)
```bash
cd ai_components/prioritization
python3 data_generator.py  # Creates train/test CSV files
python3 model.py           # Trains models, generates metrics & plots
```

**Outputs:**
- `data/processed/prioritization_train.csv`
- `data/models/prioritization_regression.json`
- `evaluation/prioritization_metrics.json`
- `evaluation/prioritization_*.png` (3 visualizations)

---

### Component 2: Validation (Next)
```bash
cd ai_components/validation
python3 data_generator.py  # Creates reports with various data quality issues
python3 validator.py       # Rule-based validation engine
python3 anomaly_detector.py # Train Isolation Forest
```

**Outputs:**
- List of validation errors per report
- Anomaly scores
- Precision/Recall curves

---

### Component 3: Medical NER (Next)
```bash
cd ai_components/ner
python3 data_generator.py  # Synthetic narratives with BIO tags
python3 train_ner.py       # Fine-tune BioBERT
python3 evaluate.py        # Entity-level metrics
```

**Outputs:**
- Trained PyTorch model
- Per-entity F1 scores
- Example predictions

---

### Component 4: Response Prediction (Next)
```bash
cd ai_components/response_prediction
python3 data_generator.py  # Historical follow-up attempts
python3 model.py           # Train LightGBM classifier
```

**Outputs:**
- Trained model
- ROC curve, PR curve
- Calibration plot

---

### Component 5: Translation (API-based)
```bash
cd ai_components/translation
python3 translator.py      # Wrapper around translation API
python3 test_glossary.py   # Validate medical term preservation
```

**Outputs:**
- Translation quality metrics
- Back-translation comparisons

---

## Performance Benchmarks (Expected)

| Component | Key Metric | Target | Rationale |
|-----------|------------|--------|-----------|
| Prioritization | R² score | ≥ 0.85 | Good predictive power for scoring |
| Prioritization | Classification Accuracy | ≥ 85% | Reliable category assignment |
| Validation | False Positive Rate | < 5% | Minimize false alarms |
| Medical NER | F1 (DRUG) | ≥ 0.90 | Critical for drug name extraction |
| Medical NER | F1 (EVENT) | ≥ 0.85 | Adverse events are diverse/harder |
| Response Prediction | AUC-ROC | ≥ 0.75 | Useful discrimination |
| Translation | BLEU score | ≥ 0.40 | Acceptable quality for medical domain |
| Translation | Term Preservation | ≥ 95% | Critical for accuracy |

---

## Visualization Outputs

Each model generates performance visualizations:

### 1. Prioritization
- Actual vs Predicted scatter plot (regression)
- Residual plot
- Confusion matrix (classification)
- Feature importance bar chart

### 2. Validation
- Precision-Recall curve for anomaly detection
- Distribution of completeness scores
- Error type frequency chart

### 3. Medical NER
- Per-entity F1 score bar chart
- Confusion matrix for entity types
- Sample predictions with highlighting

### 4. Response Prediction
- ROC curve
- Precision-Recall curve
- Calibration plot (predicted prob vs actual)
- Feature importance

### 5. Translation
- BLEU score by language pair
- Medical term preservation rate
- Back-translation error distribution

---

## Dependencies Summary

**Core ML:**
- XGBoost, LightGBM, scikit-learn
- PyTorch, Transformers (Hugging Face)

**Data Processing:**
- pandas, numpy

**Visualization:**
- matplotlib, seaborn, plotly

**NLP:**
- spaCy, nltk, transformers

**Validation:**
- pydantic, pandera

**API/Web:**
- FastAPI, Streamlit

---

## Next Steps

### Immediate (This Session)
1. ✅ Set up project structure
2. ✅ Build Prioritization Engine  
3. Build Validation Engine
4. Build Response Prediction Model
5. Build Medical NER System

### Future Sessions
6. Create unified Streamlit dashboard
7. Build FastAPI REST endpoints
8. Integration testing
9. Deployment documentation

---

## File Structure After All Components Built

```
pharma-followup-platform/
├── README.md
├── AI_COMPONENTS_OVERVIEW.md (detailed specs)
├── AI_COMPONENTS_SUMMARY.md (this file)
├── requirements.txt
├── setup.sh
├── train_prioritization.sh
│
├── ai_components/
│   ├── prioritization/ ✅
│   │   ├── data_generator.py
│   │   └── model.py
│   ├── validation/
│   │   ├── data_generator.py
│   │   ├── validator.py
│   │   └── anomaly_detector.py
│   ├── ner/
│   │   ├── data_generator.py
│   │   ├── train_ner.py
│   │   └── evaluate.py
│   ├── response_prediction/
│   │   ├── data_generator.py
│   │   └── model.py
│   └── translation/
│       ├── translator.py
│       └── test_glossary.py
│
├── data/
│   ├── processed/  (CSV training datasets)
│   └── models/     (Trained model files)
│
├── evaluation/     (Metrics JSON + visualization PNGs)
│
├── utils/
│   └── common.py   (Shared utilities)
│
└── dashboard/
    └── app.py      (Streamlit unified dashboard)
```

---

## Training Order Recommendation

1. **Prioritization** (✅ done) - Quickest, foundational
2. **Validation** - No ML training needed for rules, fast Isolation Forest
3. **Response Prediction** - Standard tabular ML, quick to train
4. **Medical NER** - Requires GPU, longest training time
5. **Translation** - API integration, no training

---

Let me know which component you'd like to build next!
