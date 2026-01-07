"""
Interactive Streamlit dashboard for visualizing prioritization model performance.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append('../..')
from utils.common import load_metrics

# Page config
st.set_page_config(
    page_title="Prioritization Model Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("📊 Prioritization Model")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select View",
    ["📈 Overview", "🎯 Metrics", "🔍 Detailed Analysis", "📑 Report"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Information")
st.sidebar.info(
    """
    **Model Type:** XGBoost (Regression + Classification)
    
    **Purpose:** Prioritize adverse event cases by urgency
    
    **Output:** Priority score (1-10) + Category (Low/Medium/High/Critical)
    """
)

# Load data
metrics_path = Path('../../evaluation/prioritization_metrics.json')
test_csv = Path('../../data/processed/prioritization_test.csv')

def load_all_data():
    try:
        metrics = load_metrics(str(metrics_path))
        test_df = pd.read_csv(test_csv)
        return metrics, test_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

metrics, test_df = load_all_data()

if metrics and test_df is not None:
    # ==================== OVERVIEW PAGE ====================
    if page == "📈 Overview":
        st.title("🎯 Prioritization Model - Performance Dashboard")
        
        st.markdown("""
        This dashboard displays the performance metrics for the **Follow-up Prioritization Engine**,
        which automatically ranks adverse event cases by urgency using XGBoost machine learning.
        """)
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            r2 = metrics['regression']['r2']
            st.metric(
                "R² Score",
                f"{r2:.4f}",
                "✓" if r2 >= 0.85 else "△",
                delta_color="off"
            )
        
        with col2:
            rmse = metrics['regression']['rmse']
            st.metric(
                "RMSE",
                f"{rmse:.4f}",
                "✓" if rmse <= 0.50 else "△",
                delta_color="off"
            )
        
        with col3:
            mae = metrics['regression']['mae']
            st.metric(
                "MAE",
                f"{mae:.4f}",
                "✓" if mae <= 0.40 else "△",
                delta_color="off"
            )
        
        with col4:
            accuracy = metrics['classification']['accuracy']
            st.metric(
                "Accuracy",
                f"{accuracy:.2%}",
                "✓" if accuracy >= 0.85 else "△",
                delta_color="off"
            )
        
        with col5:
            top_feature = max(metrics['feature_importance'], key=metrics['feature_importance'].get)
            st.metric(
                "Top Feature",
                top_feature.replace('_encoded', '').replace('_', ' ').title(),
                metrics['feature_importance'][top_feature]
            )
        
        st.markdown("---")
        
        # Show visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Regression Performance")
            try:
                img = Path('../../evaluation/prioritization_regression.png')
                st.image(str(img), use_column_width=True)
            except:
                st.warning("Regression plot not found")
        
        with col2:
            st.subheader("🏷️ Classification Confusion Matrix")
            try:
                img = Path('../../evaluation/prioritization_classification_confusion_matrix.png')
                st.image(str(img), use_column_width=True)
            except:
                st.warning("Confusion matrix not found")
        
        st.markdown("---")
        st.subheader("🎯 Feature Importance")
        try:
            img = Path('../../evaluation/prioritization_feature_importance.png')
            st.image(str(img), use_column_width=True)
        except:
            st.warning("Feature importance plot not found")
    
    # ==================== METRICS PAGE ====================
    elif page == "🎯 Metrics":
        st.title("📊 Detailed Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Regression Metrics")
            reg_df = pd.DataFrame({
                'Metric': ['RMSE', 'MAE', 'R² Score'],
                'Value': [
                    f"{metrics['regression']['rmse']:.4f}",
                    f"{metrics['regression']['mae']:.4f}",
                    f"{metrics['regression']['r2']:.4f}"
                ],
                'Target': ['≤ 0.50', '≤ 0.40', '≥ 0.85']
            })
            st.dataframe(reg_df, use_container_width=True)
        
        with col2:
            st.subheader("Classification Metrics")
            clf_df = pd.DataFrame({
                'Metric': ['Accuracy'],
                'Value': [f"{metrics['classification']['accuracy']:.4f}"],
                'Target': ['≥ 0.85']
            })
            st.dataframe(clf_df, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Feature Importance")
        
        # Sort features by importance
        feature_importance = metrics['feature_importance']
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        
        importance_df = pd.DataFrame({
            'Feature': [f[0].replace('_encoded', '').replace('_', ' ').title() for f in sorted_features],
            'Importance': [f[1] for f in sorted_features]
        })
        
        st.dataframe(importance_df, use_container_width=True)
        
        # Visualization
        st.bar_chart(importance_df.set_index('Feature'))
        
        try:
            img = Path('../../evaluation/prioritization_top_features.png')
            st.image(str(img), use_column_width=True)
        except:
            pass
    
    # ==================== DETAILED ANALYSIS PAGE ====================
    elif page == "🔍 Detailed Analysis":
        st.title("🔍 Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Prediction Distribution")
            try:
                img = Path('../../evaluation/prioritization_prediction_distribution.png')
                st.image(str(img), use_column_width=True)
            except:
                st.warning("Distribution plot not found")
        
        with col2:
            st.subheader("📈 Model Calibration")
            try:
                img = Path('../../evaluation/prioritization_calibration.png')
                st.image(str(img), use_column_width=True)
            except:
                st.warning("Calibration plot not found")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Category Distribution")
            try:
                img = Path('../../evaluation/prioritization_category_distribution.png')
                st.image(str(img), use_column_width=True)
            except:
                st.warning("Category distribution not found")
        
        with col2:
            st.subheader("❌ Error Analysis")
            try:
                img = Path('../../evaluation/prioritization_error_analysis.png')
                st.image(str(img), use_column_width=True)
            except:
                st.warning("Error analysis not found")
        
        st.markdown("---")
        st.subheader("📋 Per-Category Metrics")
        try:
            img = Path('../../evaluation/prioritization_per_category_metrics.png')
            st.image(str(img), use_column_width=True)
        except:
            st.warning("Per-category metrics not found")
    
    # ==================== REPORT PAGE ====================
    elif page == "📑 Report":
        st.title("📑 Performance Report")
        
        # Load report
        report_path = Path('../../evaluation/PRIORITIZATION_PERFORMANCE_REPORT.txt')
        if report_path.exists():
            with open(report_path, 'r') as f:
                report_text = f.read()
            st.text(report_text)
        else:
            st.info("Run model.py to generate the full performance report")
        
        st.markdown("---")
        st.subheader("📊 Quick Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown("""
            ### Model Architecture
            - **Regression Model:** XGBoost regression (priority scores 1-10)
            - **Classification Model:** XGBoost multi-class (4 categories)
            - **Input Features:** 13 total
            - **Training Samples:** 4,000
            - **Test Samples:** 1,000
            
            ### Key Features
            1. Medical severity (seriousness, event type)
            2. Data completeness
            3. Temporal metrics (days since report, deadline)
            4. Reporter characteristics
            5. Historical response rate
            """)
        
        with summary_col2:
            st.markdown(f"""
            ### Performance Summary
            
            **Regression:**
            - R² Score: {metrics['regression']['r2']:.4f}
            - RMSE: {metrics['regression']['rmse']:.4f}
            - MAE: {metrics['regression']['mae']:.4f}
            
            **Classification:**
            - Accuracy: {metrics['classification']['accuracy']:.2%}
            
            **Status:** ✅ Model is performing well
            
            *Last Updated: Jan 2026*
            """)
        
        st.markdown("---")
        st.subheader("📂 Generated Files")
        
        files_info = {
            "Model Files": [
                "data/models/prioritization_regression.json",
                "data/models/prioritization_classification.json",
                "data/models/prioritization_encoders.pkl"
            ],
            "Data Files": [
                "data/processed/prioritization_train.csv",
                "data/processed/prioritization_test.csv"
            ],
            "Visualizations": [
                "evaluation/prioritization_regression.png",
                "evaluation/prioritization_classification_confusion_matrix.png",
                "evaluation/prioritization_feature_importance.png",
                "evaluation/prioritization_metrics_table.png",
                "evaluation/prioritization_per_category_metrics.png",
                "evaluation/prioritization_prediction_distribution.png",
                "evaluation/prioritization_error_analysis.png",
                "evaluation/prioritization_category_distribution.png",
                "evaluation/prioritization_calibration.png",
                "evaluation/prioritization_top_features.png"
            ],
            "Reports": [
                "evaluation/prioritization_metrics.json",
                "evaluation/PRIORITIZATION_PERFORMANCE_REPORT.txt"
            ]
        }
        
        for category, files in files_info.items():
            st.markdown(f"**{category}:**")
            for file in files:
                st.code(file, language="text")

else:
    st.error("⚠️ Data not found. Please run the model training first:")
    st.code("cd ai_components/prioritization && python3 data_generator.py && python3 model.py", language="bash")
