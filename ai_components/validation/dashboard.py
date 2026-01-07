"""
Data Validation & Gap Detection Engine - Streamlit Dashboard
Interactive visualization and analysis of validation results
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append('../..')

import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Data Validation Engine",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


def load_data():
    """Load validation results and metrics."""
    base_dir = Path(__file__).parent.parent.parent
    
    # Load CSV results
    csv_path = base_dir / 'data' / 'processed' / 'validation_results.csv'
    if csv_path.exists():
        results_df = pd.read_csv(csv_path)
    else:
        results_df = None
    
    # Load metrics JSON
    metrics_path = base_dir / 'evaluation' / 'validation_metrics.json'
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    
    # Load report text
    report_path = base_dir / 'evaluation' / 'VALIDATION_ENGINE_REPORT.txt'
    report_text = ""
    if report_path.exists():
        with open(report_path) as f:
            report_text = f.read()
    
    return results_df, metrics, report_text, base_dir


def load_images(base_dir):
    """Load visualization images."""
    viz_dir = base_dir / 'evaluation' / 'validation_visualizations'
    images = {}
    
    if viz_dir.exists():
        for i in range(1, 8):
            img_path = viz_dir / f'{i:02d}_*.png'
            # Find matching file
            files = list(viz_dir.glob(f'{i:02d}_*.png'))
            if files:
                images[i] = files[0]
    
    return images


def render_metric_card(col, value, label, color="#1f77b4"):
    """Render a metric card."""
    with col:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {color};">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def page_overview():
    """Overview page with key metrics."""
    st.title("📊 Validation Engine Dashboard")
    
    results_df, metrics, report_text, base_dir = load_data()
    
    if results_df is None:
        st.warning("⚠️ No validation results found. Run the pipeline first:")
        st.code("python model.py")
        return
    
    # Key statistics
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    total_reports = len(results_df)
    valid_count = (results_df['is_valid'] == True).sum()
    invalid_count = (results_df['is_valid'] == False).sum()
    avg_quality = results_df['quality_score'].mean()
    
    render_metric_card(col1, f"{total_reports:,}", "Total Reports")
    render_metric_card(col2, f"{valid_count:,}", "Valid Reports", "#2ca02c")
    render_metric_card(col3, f"{invalid_count:,}", "Invalid Reports", "#d62728")
    render_metric_card(col4, f"{avg_quality:.1f}/100", "Avg Quality Score", "#ff7f0e")
    
    # Performance metrics
    st.subheader("Validation Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    val_metrics = metrics.get('validation_metrics', {})
    precision = val_metrics.get('precision', 0)
    recall = val_metrics.get('recall', 0)
    f1 = val_metrics.get('f1', 0)
    fpr = val_metrics.get('false_positive_rate', 0)
    
    render_metric_card(col1, f"{precision:.3f}", "Precision", "#1f77b4")
    render_metric_card(col2, f"{recall:.3f}", "Recall", "#ff7f0e")
    render_metric_card(col3, f"{f1:.3f}", "F1-Score", "#2ca02c")
    render_metric_card(col4, f"{fpr:.4f}", "False Positive Rate", "#d62728")
    
    # Anomaly detection metrics
    st.subheader("Anomaly Detection")
    col1, col2, col3 = st.columns(3)
    
    anom_metrics = metrics.get('anomaly_metrics', {})
    anom_precision = anom_metrics.get('precision', 0)
    anom_recall = anom_metrics.get('recall', 0)
    anom_f1 = anom_metrics.get('f1', 0)
    
    render_metric_card(col1, f"{anom_precision:.3f}", "Anomaly Precision", "#1f77b4")
    render_metric_card(col2, f"{anom_recall:.3f}", "Anomaly Recall", "#ff7f0e")
    render_metric_card(col3, f"{anom_f1:.3f}", "Anomaly F1-Score", "#2ca02c")
    
    # Status distribution
    st.subheader("Validation Status Distribution")
    status_counts = results_df['overall_status'].value_counts()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        colors = {'ACCEPT': '#2ca02c', 'CONDITIONAL_ACCEPT': '#ff7f0e', 
                 'REVIEW': '#ffa500', 'REJECT': '#d62728'}
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            color=status_counts.index,
            color_discrete_map=colors,
            hole=0.3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Status Breakdown")
        for status in ['ACCEPT', 'CONDITIONAL_ACCEPT', 'REVIEW', 'REJECT']:
            count = status_counts.get(status, 0)
            pct = (count / total_reports) * 100
            st.metric(status, f"{count} ({pct:.1f}%)")


def page_visualizations():
    """Display generated visualizations."""
    st.title("📈 Validation Visualizations")
    
    results_df, metrics, report_text, base_dir = load_data()
    images = load_images(base_dir)
    
    if not images:
        st.warning("⚠️ No visualizations found. Run the pipeline first: python model.py")
        return
    
    # Visualization descriptions
    viz_descriptions = {
        1: "Error counts in all reports, comparing detection vs actual errors",
        2: "Quality scores across all reports with interpretation zones",
        3: "Anomaly score distribution by risk level (Low/Medium/High)",
        4: "Overall validation status breakdown (ACCEPT/REJECT/REVIEW)",
        5: "Correlation between quality score and anomaly score",
        6: "Analysis of error types and their impact",
        7: "Performance metrics summary (Precision, Recall, F1, AUC-ROC)"
    }
    
    # Display visualizations in grid
    cols = st.columns(2)
    
    for i in range(1, 8):
        if i in images:
            col_idx = (i - 1) % 2
            with cols[col_idx]:
                st.image(str(images[i]), use_column_width=True)
                st.caption(f"**{i:02d}. {images[i].stem.replace('_', ' ').title()}**\n{viz_descriptions.get(i, '')}")


def page_detailed_analysis():
    """Detailed analysis of validation results."""
    st.title("🔍 Detailed Analysis")
    
    results_df, metrics, report_text, base_dir = load_data()
    
    if results_df is None:
        st.warning("⚠️ No validation results found. Run the pipeline first: python model.py")
        return
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Quality Distribution", "Anomaly Analysis", "Status Breakdown", "Raw Data"]
    )
    
    # Tab 1: Quality Distribution
    with tab1:
        st.subheader("Quality Score Distribution")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.histogram(
                results_df,
                x='quality_score',
                nbins=30,
                title="Quality Score Distribution",
                labels={'quality_score': 'Quality Score (0-100)', 'count': 'Number of Reports'}
            )
            fig.add_vline(
                x=results_df['quality_score'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {results_df['quality_score'].mean():.2f}"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col1:
            # Quality interpretation
            st.markdown("### Quality Score Interpretation")
            quality_ranges = {
                'Excellent (80-100%)': ((results_df['quality_score'] >= 80).sum(), '#2ca02c'),
                'Good (60-80%)': (((results_df['quality_score'] >= 60) & (results_df['quality_score'] < 80)).sum(), '#90ee90'),
                'Fair (40-60%)': (((results_df['quality_score'] >= 40) & (results_df['quality_score'] < 60)).sum(), '#ffa500'),
                'Poor (20-40%)': (((results_df['quality_score'] >= 20) & (results_df['quality_score'] < 40)).sum(), '#ff7f0e'),
                'Critical (<20%)': ((results_df['quality_score'] < 20).sum(), '#d62728')
            }
            
            for label, (count, color) in quality_ranges.items():
                pct = (count / len(results_df)) * 100
                emoji = '🟢' if label.startswith('Excellent') or label.startswith('Good') else '🟠' if label.startswith('Fair') or label.startswith('Poor') else '🔴'
                st.write(f"{emoji} **{label}**: {count:,} ({pct:.1f}%)")
    
    # Tab 2: Anomaly Analysis
    with tab2:
        st.subheader("Anomaly Risk Distribution")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            risk_counts = results_df['anomaly_risk'].value_counts()
            colors_risk = {'Low': '#2ca02c', 'Medium': '#ff7f0e', 'High': '#d62728'}
            
            fig = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                title="Reports by Anomaly Risk Level",
                labels={'x': 'Risk Level', 'y': 'Count'},
                color=risk_counts.index,
                color_discrete_map=colors_risk
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Risk Summary")
            for risk_level in ['Low', 'Medium', 'High']:
                count = (results_df['anomaly_risk'] == risk_level).sum()
                pct = (count / len(results_df)) * 100
                st.metric(f"{risk_level} Risk", f"{count:,} ({pct:.1f}%)")
    
    # Tab 3: Status Breakdown
    with tab3:
        st.subheader("Validation Status Details")
        
        status_data = []
        for status in ['ACCEPT', 'CONDITIONAL_ACCEPT', 'REVIEW', 'REJECT']:
            mask = results_df['overall_status'] == status
            count = mask.sum()
            avg_quality = results_df[mask]['quality_score'].mean() if count > 0 else 0
            avg_errors = results_df[mask]['error_count'].mean() if count > 0 else 0
            status_data.append({
                'Status': status,
                'Count': count,
                'Percentage': (count / len(results_df)) * 100,
                'Avg Quality': avg_quality,
                'Avg Errors': avg_errors
            })
        
        status_df = pd.DataFrame(status_data)
        st.dataframe(status_df, use_container_width=True)
        
        # Status vs Quality
        fig = px.box(
            results_df,
            x='overall_status',
            y='quality_score',
            title="Quality Score by Status",
            color='overall_status',
            color_discrete_map={
                'ACCEPT': '#2ca02c',
                'CONDITIONAL_ACCEPT': '#ff7f0e',
                'REVIEW': '#ffa500',
                'REJECT': '#d62728'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Raw Data
    with tab4:
        st.subheader("Raw Validation Results")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_errors = st.slider("Minimum Error Count", 0, int(results_df['error_count'].max()), 0)
        with col2:
            selected_status = st.multiselect(
                "Validation Status",
                results_df['overall_status'].unique(),
                default=results_df['overall_status'].unique()
            )
        with col3:
            min_quality = st.slider("Minimum Quality Score", 0, 100, 0)
        
        # Apply filters
        filtered_df = results_df[
            (results_df['error_count'] >= min_errors) &
            (results_df['overall_status'].isin(selected_status)) &
            (results_df['quality_score'] >= min_quality)
        ]
        
        st.write(f"**Showing {len(filtered_df):,} of {len(results_df):,} reports**")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download CSV
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Results",
            data=csv,
            file_name="validation_results_filtered.csv",
            mime="text/csv"
        )


def page_metrics():
    """Detailed metrics page."""
    st.title("📊 Performance Metrics")
    
    results_df, metrics, report_text, base_dir = load_data()
    
    if not metrics:
        st.warning("⚠️ No metrics found. Run the pipeline first: python model.py")
        return
    
    # Validation Metrics
    st.subheader("Validation Metrics")
    val_metrics = metrics.get('validation_metrics', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics_list = [
        (col1, "Precision", val_metrics.get('precision', 0)),
        (col2, "Recall", val_metrics.get('recall', 0)),
        (col3, "F1-Score", val_metrics.get('f1', 0)),
        (col4, "Accuracy", val_metrics.get('accuracy', 0)),
        (col5, "FPR", val_metrics.get('false_positive_rate', 0))
    ]
    
    for col, label, value in metrics_list:
        with col:
            st.metric(label, f"{value:.4f}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    if 'confusion_matrix' in val_metrics:
        cm = val_metrics['confusion_matrix']
        cm_df = pd.DataFrame(
            cm,
            columns=['Predicted Negative', 'Predicted Positive'],
            index=['Actual Negative', 'Actual Positive']
        )
        st.dataframe(cm_df, use_container_width=True)
    
    # Anomaly Detection Metrics
    st.subheader("Anomaly Detection Metrics")
    anom_metrics = metrics.get('anomaly_metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    anom_list = [
        (col1, "Precision", anom_metrics.get('precision', 0)),
        (col2, "Recall", anom_metrics.get('recall', 0)),
        (col3, "F1-Score", anom_metrics.get('f1', 0)),
        (col4, "AUC-ROC", anom_metrics.get('auc_roc', 0))
    ]
    
    for col, label, value in anom_list:
        with col:
            st.metric(label, f"{value:.4f}")
    
    # Error Detection Analysis
    st.subheader("Error Detection Analysis")
    error_analysis = metrics.get('error_detection_analysis', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Detection Rate")
        detection_rate = error_analysis.get('error_detection_rate', {})
        st.metric("Detected", f"{detection_rate.get('detected', 0):.1%}")
        st.metric("Missed", f"{detection_rate.get('missed', 0):.1%}")
    
    with col2:
        st.markdown("### False Positives")
        fp_rate = error_analysis.get('false_positive_rate', {})
        st.metric("FP Rate", f"{fp_rate.get('rate', 0):.4f}")
        st.metric("FP Count", f"{fp_rate.get('count', 0)}")
    
    # Anomaly Detection Analysis
    st.subheader("Anomaly Detection Analysis")
    anom_analysis = metrics.get('anomaly_detection_analysis', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Anomalies in Error Reports")
        error_anom = anom_analysis.get('anomalies_in_error_reports', {})
        st.metric("Detected", f"{error_anom.get('detected', 0)}")
        st.metric("Rate", f"{error_anom.get('rate', 0):.1%}")
    
    with col2:
        st.markdown("### False Anomalies in Clean Reports")
        clean_anom = anom_analysis.get('false_anomalies_in_clean_reports', {})
        st.metric("False Positives", f"{clean_anom.get('detected', 0)}")
        st.metric("Rate", f"{clean_anom.get('rate', 0):.1%}")


def page_report():
    """Raw report page."""
    st.title("📄 Validation Report")
    
    results_df, metrics, report_text, base_dir = load_data()
    
    if not report_text:
        st.warning("⚠️ No report found. Run the pipeline first: python model.py")
        return
    
    st.text(report_text)
    
    # Download report
    st.download_button(
        label="📥 Download Full Report",
        data=report_text,
        file_name="VALIDATION_ENGINE_REPORT.txt",
        mime="text/plain"
    )


def main():
    """Main app."""
    # Sidebar
    st.sidebar.title("🔍 Validation Dashboard")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Overview", "📈 Visualizations", "🔍 Analysis", "📊 Metrics", "📄 Report"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        """
        **Data Validation & Gap Detection Engine**
        
        Validates pharmaceutical adverse event reports against ICH E2B(R3) standards.
        
        Features:
        - Rule-based validation (6 checks)
        - Isolation Forest anomaly detection
        - Completeness scoring
        - Comprehensive metrics
        - 7 visualizations
        """
    )
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Route to pages
    if page == "📊 Overview":
        page_overview()
    elif page == "📈 Visualizations":
        page_visualizations()
    elif page == "🔍 Analysis":
        page_detailed_analysis()
    elif page == "📊 Metrics":
        page_metrics()
    elif page == "📄 Report":
        page_report()


if __name__ == "__main__":
    main()
