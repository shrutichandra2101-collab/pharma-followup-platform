# Getting Started

## Overview

This project contains 5 AI/ML components for improving pharmacovigilance follow-up processes:

1. ✅ **Follow-up Prioritization Engine** - Implemented
2. **Data Validation & Gap Detection** - To be built
3. **Medical NER (Named Entity Recognition)** - To be built
4. **Response Prediction Model** - To be built
5. **Multilingual Translation** - To be built

---

## Quick Start (5 minutes)

### Step 1: Install Dependencies

```bash
# From project root directory
./setup.sh
```

This will:
- Create a Python virtual environment
- Install all required packages (numpy, pandas, xgboost, scikit-learn, etc.)

### Step 2: Train the Prioritization Model

```bash
./train_prioritization.sh
```

This will:
- Generate 5,000 synthetic training cases
- Train XGBoost regression model (priority scores 1-10)
- Train XGBoost classification model (Low/Medium/High/Critical)
- Generate performance metrics and visualizations
- Save models to `data/models/`

### Step 3: View Results

**Metrics (JSON):**
```bash
cat evaluation/prioritization_metrics.json
```

**Visualizations (on Mac):**
```bash
open evaluation/prioritization_regression.png
open evaluation/prioritization_classification_confusion_matrix.png
open evaluation/prioritization_feature_importance.png
```

---

## Manual Setup (if scripts don't work)

### Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Train Prioritization Model
```bash
cd ai_components/prioritization
python3 data_generator.py
python3 model.py
```

---

## Project Structure

```
pharma-followup-platform/
│
├── README.md                          # Main documentation
├── GETTING_STARTED.md                 # This file
├── AI_COMPONENTS_OVERVIEW.md          # Detailed specs for all 5 components
├── AI_COMPONENTS_SUMMARY.md           # Quick reference guide
│
├── requirements.txt                   # Python dependencies
├── setup.sh                           # Environment setup script
├── train_prioritization.sh            # Quick training script
│
├── ai_components/                     # AI model implementations
│   ├── prioritization/                # ✅ Component 1 (done)
│   │   ├── data_generator.py          # Generate synthetic training data
│   │   └── model.py                   # Train XGBoost models
│   ├── validation/                    # Component 2 (to be built)
│   ├── ner/                           # Component 3 (to be built)
│   ├── response_prediction/           # Component 4 (to be built)
│   └── translation/                   # Component 5 (to be built)
│
├── data/
│   ├── processed/                     # Generated training datasets (CSV)
│   │   ├── prioritization_train.csv
│   │   └── prioritization_test.csv
│   └── models/                        # Trained models
│       ├── prioritization_regression.json
│       ├── prioritization_classification.json
│       └── prioritization_encoders.pkl
│
├── evaluation/                        # Metrics and visualizations
│   ├── prioritization_metrics.json
│   ├── prioritization_regression.png
│   ├── prioritization_classification_confusion_matrix.png
│   └── prioritization_feature_importance.png
│
└── utils/
    └── common.py                      # Shared utilities
```

---

## Understanding the Prioritization Model

### What It Does
Automatically ranks adverse event cases by urgency to help prioritize which follow-ups should happen first.

### Input Features (13 total)
1. **Medical Severity:**
   - Is the case serious? (death, hospitalization, etc.)
   - Seriousness score (1-10)
   - Event type (cardiac, GI, neurological, etc.)

2. **Data Quality:**
   - Completeness percentage (% of mandatory fields filled)

3. **Timing:**
   - Days since initial report
   - Days until regulatory deadline

4. **Reporter Context:**
   - Reporter type (HCP, patient, pharmacist)
   - Reporter reliability score
   - Region (North America, Europe, Asia-Pacific, etc.)
   - Regulatory strictness in that region

5. **History:**
   - Number of previous follow-up attempts
   - Historical response rate for similar cases

### Outputs

**1. Regression Model:**
- Continuous priority score from 1 to 10
- Higher score = more urgent

**2. Classification Model:**
- Low (<4): Can wait
- Medium (4-6): Normal priority
- High (6-8): Expedite
- Critical (≥8): Immediate attention

### Performance Metrics

After training, you'll see:
- **R² Score:** How well the model predicts scores (target: ≥0.85)
- **RMSE:** Root mean squared error (lower is better)
- **MAE:** Mean absolute error (lower is better)
- **Classification Accuracy:** % correctly categorized (target: ≥85%)
- **Feature Importance:** Which factors matter most

---

## Sample Training Output

```
============================================================
TRAINING REGRESSION MODEL (Priority Score Prediction)
============================================================

[Training progress with validation scores...]

============================================================
TEST SET EVALUATION - REGRESSION
============================================================

Regression Metrics:
  RMSE: 0.5432
  MAE:  0.4123
  R²:   0.8876

Regression plots saved to ../../evaluation/prioritization_regression.png

============================================================
TEST SET EVALUATION - CLASSIFICATION
============================================================

============================================================
CLASSIFICATION REPORT
============================================================
Accuracy: 0.8920
Macro F1-Score: 0.8743

Detailed Report:
              precision    recall  f1-score   support

    Critical       0.92      0.89      0.91       250
        High       0.87      0.90      0.88       235
         Low       0.91      0.88      0.89       260
      Medium       0.87      0.90      0.88       255

    accuracy                           0.89      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.89      0.89      1000

============================================================
```

---

## Next Steps

### Option A: Build Next Component (Validation Engine)
The validation component identifies missing or inconsistent data automatically.

**To start:**
1. Review `AI_COMPONENTS_OVERVIEW.md` (section 2)
2. Implement `ai_components/validation/` scripts
3. Train and evaluate

### Option B: Build Response Prediction Model
Predicts whether an HCP/patient will respond to follow-up requests.

**To start:**
1. Review `AI_COMPONENTS_OVERVIEW.md` (section 4)
2. Implement `ai_components/response_prediction/` scripts
3. Train and evaluate

### Option C: Customize Prioritization Model
- Adjust feature weights in `data_generator.py` (lines 88-97)
- Add new features (e.g., product type, patient age)
- Retrain and compare performance

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'numpy'`
**Solution:** Activate the virtual environment first:
```bash
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
```

### Issue: `python: command not found`
**Solution:** Use `python3` instead of `python`:
```bash
python3 -m venv venv
```

### Issue: Training script fails
**Solution:** Run steps manually:
```bash
cd ai_components/prioritization
python3 data_generator.py
python3 model.py
```

---

## System Requirements

- **Python:** 3.8 or higher
- **RAM:** 2GB+ (4GB recommended)
- **Storage:** 500MB for dependencies + models
- **OS:** MacOS, Linux, or Windows
- **GPU:** Not required for current components (optional for NER)

---

## Key Files to Explore

1. **`ai_components/prioritization/data_generator.py`**
   - See how synthetic training data is created
   - Formula for calculating priority scores (lines 88-97)

2. **`ai_components/prioritization/model.py`**
   - XGBoost model configuration
   - Feature engineering
   - Evaluation metrics

3. **`utils/common.py`**
   - Reusable plotting functions
   - Pharmacovigilance constants (event types, regions, etc.)

4. **`AI_COMPONENTS_OVERVIEW.md`**
   - Complete technical specifications
   - Training data requirements
   - Integration architecture

---

## Questions?

- Review `AI_COMPONENTS_SUMMARY.md` for quick reference
- Check `AI_COMPONENTS_OVERVIEW.md` for detailed specs
- Examine code comments in Python files

Ready to build the next component? Let me know!
