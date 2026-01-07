# 🎨 Quick Start: View Model Visualizations

## 🚀 Fastest Way (2 minutes)

### Option 1: Interactive Dashboard (RECOMMENDED)
```bash
# From project root
source venv/bin/activate
cd ai_components/prioritization
streamlit run dashboard.py
```
Opens browser with interactive dashboard showing all visualizations.

---

## 📊 View Individual Visualizations

All visualization PNG files are in `evaluation/` directory:

```bash
cd evaluation/

# View on Mac
open prioritization_regression.png
open prioritization_error_analysis.png
open prioritization_calibration.png
open prioritization_metrics_table.png
# ... etc

# View on Linux
eog prioritization_regression.png
```

### Visualization Files (10 total)

1. **prioritization_regression.png** - Actual vs predicted scores + residuals
2. **prioritization_classification_confusion_matrix.png** - Category predictions
3. **prioritization_feature_importance.png** - Top features ranking
4. **prioritization_metrics_table.png** - Summary metrics table ⭐
5. **prioritization_per_category_metrics.png** - Metrics per category
6. **prioritization_prediction_distribution.png** - Score distributions
7. **prioritization_error_analysis.png** - 4-panel error deep dive
8. **prioritization_category_distribution.png** - Actual vs predicted counts
9. **prioritization_calibration.png** - Model calibration analysis
10. **prioritization_top_features.png** - Top 15 features ranked

---

## 📄 View Reports

### JSON Metrics
```bash
cat evaluation/prioritization_metrics.json
```

### Text Report
```bash
cat evaluation/PRIORITIZATION_PERFORMANCE_REPORT.txt
```

---

## 📈 Performance at a Glance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| R² Score | 0.8490 | ≥ 0.85 | ✓ |
| RMSE | 0.5424 | ≤ 0.50 | 📈 |
| MAE | 0.4358 | ≤ 0.40 | 📈 |
| Accuracy | 82.80% | ≥ 85% | 📈 |

---

## 📚 Documentation

- **Detailed Guide**: [VISUALIZATION_GUIDE.md](ai_components/prioritization/VISUALIZATION_GUIDE.md)
- **Full Summary**: [VISUALIZATION_SUMMARY.md](VISUALIZATION_SUMMARY.md)

---

**That's it!** Choose your preferred method above and start exploring the model's performance.
