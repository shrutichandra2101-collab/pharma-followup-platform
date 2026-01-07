# Pharmacovigilance Follow-up Platform - AI Components

An AI-powered platform to improve follow-up processes in pharmacovigilance by prioritizing cases, validating data, and predicting response likelihood.

## Project Structure

```
pharma-followup-platform/
├── ai_components/
│   ├── prioritization/      # Follow-up prioritization engine
│   ├── validation/           # Data validation & gap detection
│   ├── ner/                  # Medical NER for text extraction
│   ├── response_prediction/  # Response likelihood prediction
│   └── translation/          # Multilingual support
├── data/
│   ├── raw/                  # Raw data files
│   ├── processed/            # Processed training data
│   └── models/               # Trained models
├── evaluation/               # Model metrics and visualizations
├── notebooks/                # Jupyter notebooks for analysis
└── utils/                    # Shared utilities
```

## Setup

### 1. Create virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download spaCy model (for NER component)

```bash
python3 -m spacy download en_core_web_sm
```

## AI Components

### 1. Follow-up Prioritization Engine
Ranks adverse event cases by urgency using XGBoost regression and classification.

**Features:**
- Medical severity (seriousness criteria, event type)
- Data completeness percentage
- Time metrics (days since report, deadline proximity)
- Reporter characteristics
- Regulatory context

**Training:**
```bash
cd ai_components/prioritization
python3 data_generator.py  # Generate synthetic training data
python3 model.py           # Train and evaluate models
```

**Outputs:**
- Priority scores (1-10 continuous scale)
- Priority categories (Low, Medium, High, Critical)
- Feature importance analysis
- Performance metrics (RMSE, MAE, R², Accuracy)

### 2. Data Validation & Gap Detection
Identifies missing, inconsistent, or anomalous data in adverse event reports.

### 3. Medical NER (Named Entity Recognition)
Extracts structured information from free-text narratives using fine-tuned BioBERT/ClinicalBERT.

### 4. Predictive Response Model
Forecasts likelihood of response to follow-up requests.

### 5. Multilingual Translation Pipeline
Handles cross-language communication for global operations.

## Usage

### Generate Data and Train All Models

```bash
# Activate virtual environment
source venv/bin/activate

# Train prioritization model
cd ai_components/prioritization
python3 data_generator.py
python3 model.py

# View results
ls ../../evaluation/prioritization_*.png
cat ../../evaluation/prioritization_metrics.json
```

### View Visualizations

All model performance metrics and visualizations are saved in the `evaluation/` directory:
- `prioritization_regression.png` - Actual vs predicted scores
- `prioritization_classification_confusion_matrix.png` - Category classification matrix
- `prioritization_feature_importance.png` - Top features driving priorities
- `prioritization_metrics.json` - Numerical metrics

## Model Performance Metrics

### Prioritization Model
- **Regression (Priority Score):** RMSE, MAE, R²
- **Classification (Priority Category):** Accuracy, Precision, Recall, F1-Score
- **Feature Importance:** Weight-based ranking

## Next Steps

1. ✅ Build Prioritization Engine
2. Build Data Validation Engine
3. Build Medical NER System
4. Build Response Prediction Model
5. Create unified dashboard for all metrics
6. Deploy as REST API service

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`
- 2GB+ RAM for model training
- MacOS/Linux/Windows

## License

MIT License
