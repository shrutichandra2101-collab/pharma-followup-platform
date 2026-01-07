# Validation Dashboard Guide

## Overview

The **Validation Engine Streamlit Dashboard** provides an interactive, web-based interface to explore validation results, analyze data quality, and review performance metrics.

## Features

### 📊 Overview Page
- **Key Metrics**: Total reports, valid/invalid counts, average quality
- **Validation Performance**: Precision, Recall, F1-Score, False Positive Rate
- **Anomaly Detection**: Anomaly precision, recall, F1-Score
- **Status Distribution**: Interactive pie chart of validation outcomes

### 📈 Visualizations Page
- All 7 generated PNG visualizations in grid layout
- Descriptions for each chart
- High-resolution (300 DPI) professional quality

### 🔍 Analysis Page
- **Quality Distribution**: Histogram with interpretation zones
- **Anomaly Analysis**: Risk level breakdown (Low/Medium/High)
- **Status Breakdown**: Detailed statistics by validation status
- **Raw Data**: Filterable table with download option

### 📊 Metrics Page
- Detailed validation metrics (Precision, Recall, F1, Accuracy, FPR)
- Confusion matrix
- Anomaly detection metrics (AUC-ROC)
- Error detection analysis
- Anomaly detection analysis

### 📄 Report Page
- Full text report
- Download option

## Installation

### Prerequisites
```bash
# Python 3.9+
# Virtual environment with dependencies installed
cd /Users/shruti/Projects/pharma-followup-platform
source venv/bin/activate
pip install streamlit plotly pillow pandas
```

### Dependencies
- streamlit >= 1.28.0
- plotly >= 5.14.0
- pandas >= 2.0.0
- pillow >= 9.0.0

All dependencies are in `requirements.txt`.

## Usage

### Method 1: Using Shell Script (Recommended)
```bash
cd /Users/shruti/Projects/pharma-followup-platform/ai_components/validation
chmod +x run_dashboard.sh
./run_dashboard.sh
```

### Method 2: Direct Streamlit Command
```bash
cd /Users/shruti/Projects/pharma-followup-platform
source venv/bin/activate
streamlit run ai_components/validation/dashboard.py
```

### Method 3: From Python
```bash
cd /Users/shruti/Projects/pharma-followup-platform
/Users/shruti/Projects/pharma-followup-platform/venv/bin/streamlit run \
    ai_components/validation/dashboard.py
```

## Dashboard Access

Once running, the dashboard is available at:
```
http://localhost:8501
```

### Default Browser
The dashboard automatically opens in your default browser.

### Manual Browser Access
If it doesn't open automatically:
1. Open http://localhost:8501 in your web browser
2. The dashboard will load automatically

## Navigation

### Sidebar
- **Page Selection**: Switch between 5 main pages
- **Refresh Data**: Reload data from disk
- **About**: Information about the engine

### Pages
1. **Overview** - Key metrics and status distribution
2. **Visualizations** - All 7 generated charts
3. **Analysis** - Detailed breakdowns and filtering
4. **Metrics** - Performance metrics and confusion matrix
5. **Report** - Full text report with download

## Features

### Interactive Elements
- ✅ Filter data by error count, status, quality score
- ✅ Hover tooltips on charts for details
- ✅ Download filtered results as CSV
- ✅ Download full report as TXT
- ✅ Refresh data in real-time

### Visualizations
- ✅ Quality score distribution with mean/median lines
- ✅ Anomaly risk breakdown by level
- ✅ Status distribution pie chart
- ✅ Status vs quality box plots
- ✅ Historical comparison charts
- ✅ Metric summaries and trends

### Data Exploration
- ✅ View all 10,000 validation results
- ✅ Filter by multiple criteria
- ✅ Sort by any column
- ✅ Download subsets for analysis
- ✅ Real-time statistics

## Interpretation Guide

### Metrics
| Metric | Range | Interpretation |
|--------|-------|-----------------|
| **Precision** | 0-1 | Of flagged issues, % that are real (higher=better) |
| **Recall** | 0-1 | Of real issues, % that we caught (higher=better) |
| **F1-Score** | 0-1 | Balance between precision and recall (higher=better) |
| **FPR** | 0-1 | Of clean data, % incorrectly flagged (lower=better) |
| **AUC-ROC** | 0-1 | Anomaly detection discrimination (0.5=random, 1=perfect) |

### Quality Score
| Score | Interpretation |
|-------|-----------------|
| 80-100% | ✅ Excellent - All critical fields present |
| 60-80% | ✅ Good - Most important fields present |
| 40-60% | ⚠️ Fair - Some important fields missing |
| 20-40% | ⚠️ Poor - Many important fields missing |
| <20% | ❌ Critical - Most fields missing |

### Validation Status
| Status | Meaning |
|--------|---------|
| **ACCEPT** | ✅ Valid, high quality - Use as-is |
| **CONDITIONAL_ACCEPT** | ⚠️ Minor issues - May need review |
| **REVIEW** | 🔍 Needs attention - Human review recommended |
| **REJECT** | ❌ Invalid - Do not use |

### Anomaly Risk
| Level | Meaning |
|-------|---------|
| **Low** | ✅ Normal pattern - No concern |
| **Medium** | ⚠️ Unusual - Warrants attention |
| **High** | ❌ Very unusual - Investigate |

## Data Requirements

The dashboard automatically loads data from:
```
data/processed/validation_results.csv         (10,000 rows)
evaluation/validation_metrics.json            (Metrics)
evaluation/VALIDATION_ENGINE_REPORT.txt       (Report)
evaluation/validation_visualizations/*.png    (7 charts)
```

### First Run
If you get warnings about missing data:
1. Run the validation pipeline first:
   ```bash
   python ai_components/validation/model.py
   ```
2. Then launch the dashboard

## Tips & Tricks

### Filter Large Datasets
Use the filters on the Analysis page to focus on specific subsets:
- High error count reports
- Specific validation statuses
- Low quality scores

### Export Results
Download filtered results for external analysis:
1. Go to Analysis → Raw Data tab
2. Set filters as desired
3. Click "Download Filtered Results"

### Share Reports
1. Screenshots work well for quick sharing
2. Download full report for detailed documentation
3. Export metrics JSON for programmatic analysis

### Performance Optimization
- Dashboard auto-reloads if data file changes
- Use filters to reduce displayed data
- Close unused browser tabs for better performance

## Troubleshooting

### Dashboard Won't Start
```bash
# Check if Streamlit is installed
pip install streamlit plotly pillow

# Try running with verbose output
streamlit run dashboard.py --logger.level=debug
```

### "No data found" Warning
```bash
# Run the validation pipeline first
cd /Users/shruti/Projects/pharma-followup-platform
python ai_components/validation/model.py
```

### Port Already in Use
```bash
# Use a different port
streamlit run dashboard.py --server.port 8502
```

### Images Not Displaying
```bash
# Check visualization files exist
ls evaluation/validation_visualizations/
# Should show 7 PNG files (01_* through 07_*)
```

### Slow Performance
- Close other applications
- Clear browser cache (Ctrl+Shift+Delete)
- Reduce data display size using filters
- Use a faster internet connection

## Advanced Usage

### Custom Configuration
Edit dashboard settings in `~/.streamlit/config.toml`:
```toml
[client]
showErrorDetails = true
showWarningOnDirectExecution = false

[logger]
level = "info"
```

### Integration with Other Tools
Export data for use in:
- Excel/Sheets for additional analysis
- SQL databases for storage
- Python/R for advanced analytics
- Power BI/Tableau for dashboards

### Batch Processing
Automate dashboard screenshots:
```bash
# With headless browser support
streamlit run dashboard.py --logger.level=error --headless
```

## Architecture

### Components
```
dashboard.py
├── load_data()           - Load CSV, JSON, TXT
├── load_images()         - Load PNG visualizations
├── render_metric_card()  - Display metric cards
├── page_overview()       - Overview page
├── page_visualizations() - Visualization gallery
├── page_detailed_analysis() - Detailed analysis
├── page_metrics()        - Metrics page
└── page_report()         - Report page
```

### Data Flow
```
CSV/JSON/PNG Files
       ↓
   load_data()
       ↓
  Page Functions
       ↓
Streamlit Components
       ↓
  Web Dashboard
```

## Performance Metrics

### Load Time
- Dashboard startup: ~2-3 seconds
- Page switching: <1 second
- Data refresh: ~1 second
- Chart rendering: ~500ms

### Browser Compatibility
- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ⚠️ Mobile (limited)

## Support & Feedback

### Getting Help
1. Check troubleshooting section
2. Review Streamlit documentation: https://docs.streamlit.io
3. Check component source code with comments

### Reporting Issues
When reporting issues, include:
- Error message (full traceback)
- Browser and version
- Python version
- Steps to reproduce

## Related Documentation

- [Data Validation Engine Summary](../DATA_VALIDATION_ENGINE_SUMMARY.md)
- [Step-by-Step Build Explanation](../DATA_VALIDATION_STEPWISE_EXPLANATION.md)
- [Component Status](../COMPONENT_2_COMPLETE.md)

## Next Steps

### Extend Dashboard
Add custom pages:
```python
def page_custom():
    st.title("Custom Analysis")
    # Your code here
    
# Add to sidebar navigation
```

### Schedule Updates
Auto-refresh dashboard data:
```bash
# Run validation pipeline on schedule
0 * * * * cd /path && python ai_components/validation/model.py
```

### Deploy to Cloud
```bash
# Share via Streamlit Cloud
# 1. Push to GitHub
# 2. Create account on https://share.streamlit.io
# 3. Deploy repository
```

---

**Version**: 1.0  
**Last Updated**: January 7, 2026  
**Status**: ✅ Production Ready
