"""
Geospatial Signal Detection - Interactive Streamlit Dashboard
Real-time monitoring interface for batch anomalies and regional spikes

Step 7: Implement Streamlit dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

# Set page config
st.set_page_config(
    page_title="Signal Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .alert-critical {
        background-color: #ffcccc;
        border-left: 4px solid #d62728;
    }
    .alert-high {
        background-color: #ffe6cc;
        border-left: 4px solid #ff7f0e;
    }
    .alert-medium {
        background-color: #fff9e6;
        border-left: 4px solid #ffbb78;
    }
    .alert-low {
        background-color: #e6f7e6;
        border-left: 4px solid #2ca02c;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load all data files."""
    try:
        df = pd.read_csv('signal_detection_results/signal_detection_data.csv')
        batch_scores = pd.read_csv('signal_detection_results/batch_risk_scores.csv')
        with open('signal_detection_results/signal_detection_metrics.json', 'r') as f:
            metrics = json.load(f)
        return df, batch_scores, metrics
    except FileNotFoundError:
        st.error("Data files not found. Please run signal_detector.py first.")
        return None, None, None


def render_dashboard():
    """Main dashboard rendering function."""
    
    st.title("🔍 Geospatial Signal Detection Dashboard")
    st.markdown("Batch Anomaly Detection & Regional Adverse Event Monitoring")
    
    # Load data
    df, batch_scores, metrics = load_data()
    
    if df is None:
        st.warning("Please run the signal detection pipeline first.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Cluster Explorer", "Batch Investigation", "Alerts & Timeline", "Geographic Map"]
    )
    
    if page == "Overview":
        render_overview(df, batch_scores, metrics)
    elif page == "Cluster Explorer":
        render_cluster_explorer(df, batch_scores)
    elif page == "Batch Investigation":
        render_batch_investigation(df, batch_scores)
    elif page == "Alerts & Timeline":
        render_alerts_timeline(batch_scores, df)
    elif page == "Geographic Map":
        render_geographic_map(df, batch_scores)


def render_overview(df, batch_scores, metrics):
    """Render overview page."""
    st.header("System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Cases",
            f"{len(df):,}",
            delta=f"{df['cluster_id'].nunique()} clusters"
        )
    
    with col2:
        st.metric(
            "Unique Batches",
            df['batch_id'].nunique(),
            delta=f"{metrics['cluster_statistics']['noise_points']} outliers"
        )
    
    with col3:
        critical_count = len(batch_scores[batch_scores['alert_level'] == 'CRITICAL'])
        st.metric(
            "Critical Alerts",
            critical_count,
            delta="High Priority",
            delta_color="inverse"
        )
    
    with col4:
        high_count = len(batch_scores[batch_scores['alert_level'] == 'HIGH'])
        st.metric(
            "High Risk",
            high_count,
            delta="Investigation"
        )
    
    st.divider()
    
    # Clustering quality metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Silhouette Coefficient")
        silhouette = metrics['clustering_quality']['silhouette_coefficient']
        st.gauge(
            value=silhouette,
            min_value=-1.0,
            max_value=1.0,
            title=f"{silhouette:.3f}"
        )
        st.caption("Higher is better (-1 to 1)")
    
    with col2:
        st.subheader("Davies-Bouldin Index")
        dbi = metrics['clustering_quality']['davies_bouldin_index']
        st.gauge(
            value=min(dbi, 3),
            min_value=0,
            max_value=3,
            title=f"{dbi:.3f}"
        )
        st.caption("Lower is better")
    
    with col3:
        st.subheader("Calinski-Harabasz Index")
        chi = metrics['clustering_quality']['calinski_harabasz_index']
        st.gauge(
            value=min(chi / 500, 1.0),  # Normalized for display
            min_value=0,
            max_value=1,
            title=f"{chi:.1f}"
        )
        st.caption("Higher is better")
    
    st.divider()
    
    # Alert distribution
    col1, col2 = st.columns(2)
    
    with col1:
        alert_counts = batch_scores['alert_level'].value_counts()
        fig = px.pie(
            names=alert_counts.index,
            values=alert_counts.values,
            title="Alert Level Distribution",
            color_discrete_map={
                'CRITICAL': '#d62728',
                'HIGH': '#ff7f0e',
                'MEDIUM': '#ffbb78',
                'LOW': '#2ca02c'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        risk_data = batch_scores.groupby('alert_level')['risk_score'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=risk_data.index,
            y=risk_data.values,
            title="Average Risk Score by Alert Level",
            labels={'x': 'Alert Level', 'y': 'Average Risk Score'},
            color=risk_data.index,
            color_discrete_map={
                'CRITICAL': '#d62728',
                'HIGH': '#ff7f0e',
                'MEDIUM': '#ffbb78',
                'LOW': '#2ca02c'
            }
        )
        st.plotly_chart(fig, use_container_width=True)


def render_cluster_explorer(df, batch_scores):
    """Render cluster explorer page."""
    st.header("Cluster Explorer")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_region = st.selectbox(
            "Filter by Region",
            options=['All'] + sorted(df['region'].unique().tolist()),
            key="region_filter"
        )
    
    with col2:
        min_size = st.number_input("Minimum Cluster Size", min_value=1, value=5)
    
    with col3:
        selected_event = st.selectbox(
            "Filter by Event Type",
            options=['All'] + sorted(df['event_type'].unique().tolist()),
            key="event_filter"
        )
    
    # Apply filters
    filtered_df = df[df['cluster_id'] != -1].copy()
    
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['region'] == selected_region]
    
    if selected_event != 'All':
        filtered_df = filtered_df[filtered_df['event_type'] == selected_event]
    
    # Cluster statistics
    cluster_stats = filtered_df.groupby('cluster_id').agg({
        'case_id': 'count',
        'latitude': 'mean',
        'longitude': 'mean',
        'severity': lambda x: (x == 'Severe').sum() + (x == 'Life-threatening').sum(),
        'event_type': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown',
        'drug_name': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
    }).rename(columns={
        'case_id': 'size',
        'severity': 'severe_cases',
        'event_type': 'primary_event',
        'drug_name': 'primary_drug'
    })
    
    cluster_stats = cluster_stats[cluster_stats['size'] >= min_size].sort_values('size', ascending=False)
    
    st.subheader(f"Clusters Found: {len(cluster_stats)}")
    
    # Cluster details
    for cluster_id, row in cluster_stats.head(10).iterrows():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Cluster ID", f"{cluster_id}")
        with col2:
            st.metric("Size", f"{int(row['size'])} cases")
        with col3:
            st.metric("Severe Cases", f"{int(row['severe_cases'])}")
        with col4:
            st.metric("Location", f"{row['latitude']:.2f}, {row['longitude']:.2f}")
        
        st.caption(f"Primary Event: {row['primary_event']} | Primary Drug: {row['primary_drug']}")
        st.divider()


def render_batch_investigation(df, batch_scores):
    """Render batch investigation page."""
    st.header("Batch Investigation")
    
    # Search and filter
    col1, col2 = st.columns(2)
    
    with col1:
        search_batch = st.text_input("Search by Batch ID", key="batch_search")
    
    with col2:
        filter_alert = st.multiselect(
            "Filter by Alert Level",
            options=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
            default=['CRITICAL', 'HIGH']
        )
    
    # Filter batch scores
    filtered_scores = batch_scores.copy()
    
    if search_batch:
        filtered_scores = filtered_scores[filtered_scores['batch_id'].str.contains(search_batch, case=False)]
    
    filtered_scores = filtered_scores[filtered_scores['alert_level'].isin(filter_alert)]
    filtered_scores = filtered_scores.sort_values('risk_score', ascending=False)
    
    st.subheader(f"Batches Found: {len(filtered_scores)}")
    
    # Display batch details
    for _, batch in filtered_scores.head(10).iterrows():
        alert_color_map = {
            'CRITICAL': '#ffcccc',
            'HIGH': '#ffe6cc',
            'MEDIUM': '#fff9e6',
            'LOW': '#e6f7e6'
        }
        
        with st.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"### {batch['batch_id']}")
                st.metric("Risk Score", f"{batch['risk_score']:.3f}")
            
            with col2:
                st.metric("Cases", int(batch['num_cases']))
                st.metric("Alert", batch['alert_level'], label_visibility="collapsed")
            
            with col3:
                st.metric("Region", batch['primary_region'])
                st.metric("Event", batch['primary_event'])
            
            # Risk components
            components_df = pd.DataFrame({
                'Component': [
                    'Geographic', 'Temporal', 'Event Similarity', 'Severity', 'Size', 'Manufacturing'
                ],
                'Score': [
                    batch['geographic_concentration'],
                    batch['temporal_concentration'],
                    batch['event_similarity'],
                    batch['severity_concentration'],
                    batch['size_anomaly'],
                    batch['manufacturing_concentration']
                ]
            })
            
            fig = px.bar(
                components_df,
                x='Component',
                y='Score',
                title=f"Risk Component Breakdown",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()


def render_alerts_timeline(batch_scores, df):
    """Render alerts and timeline page."""
    st.header("Alerts & Timeline")
    
    # Alert summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        critical = len(batch_scores[batch_scores['alert_level'] == 'CRITICAL'])
        st.metric("Critical Alerts (24h)", critical)
    
    with col2:
        high = len(batch_scores[batch_scores['alert_level'] == 'HIGH'])
        st.metric("High Risk (Weekly)", high)
    
    with col3:
        medium = len(batch_scores[batch_scores['alert_level'] == 'MEDIUM'])
        st.metric("Medium Risk (Monitor)", medium)
    
    st.divider()
    
    # Recent alerts
    st.subheader("Recent High-Risk Detections")
    
    recent_alerts = batch_scores[batch_scores['alert_level'].isin(['CRITICAL', 'HIGH'])].nlargest(10, 'risk_score')
    
    alert_table = recent_alerts[[
        'batch_id', 'num_cases', 'risk_score', 'alert_level',
        'primary_region', 'primary_event'
    ]].copy()
    
    alert_table['risk_score'] = alert_table['risk_score'].apply(lambda x: f"{x:.3f}")
    alert_table['num_cases'] = alert_table['num_cases'].astype(int)
    
    st.dataframe(alert_table, use_container_width=True)
    
    st.divider()
    
    # Temporal distribution
    st.subheader("Case Reporting Timeline")
    
    df_copy = df.copy()
    df_copy['date_reported'] = pd.to_datetime(df_copy['date_reported'])
    daily_cases = df_copy.groupby(df_copy['date_reported'].dt.date).size()
    
    fig = px.line(
        x=daily_cases.index,
        y=daily_cases.values,
        title="Daily Case Reports",
        labels={'x': 'Date', 'y': 'Number of Cases'},
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)


def render_geographic_map(df, batch_scores):
    """Render geographic map page."""
    st.header("Geographic Distribution Map")
    
    # Prepare data
    map_df = df[df['cluster_id'] != -1].copy()
    
    # Merge with batch scores
    batch_risk = batch_scores[['batch_id', 'risk_score', 'alert_level']].copy()
    map_df = map_df.merge(batch_risk, on='batch_id', how='left')
    
    # Color mapping
    color_map = {
        'CRITICAL': '#d62728',
        'HIGH': '#ff7f0e',
        'MEDIUM': '#ffbb78',
        'LOW': '#2ca02c'
    }
    
    map_df['color'] = map_df['alert_level'].map(color_map)
    
    # Create map
    fig = go.Figure()
    
    for alert_level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        alert_df = map_df[map_df['alert_level'] == alert_level]
        
        fig.add_trace(go.Scattergeo(
            lon=alert_df['longitude'],
            lat=alert_df['latitude'],
            mode='markers',
            name=alert_level,
            marker=dict(
                size=8,
                color=color_map[alert_level],
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=alert_df['batch_id'],
            hovertemplate='<b>Batch:</b> %{text}<br><b>Alert:</b> ' + alert_level + '<extra></extra>'
        ))
    
    fig.update_layout(
        title="Adverse Events by Geographic Location",
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastcolor='rgb(200, 200, 200)',
            showocean=True,
            oceancolor='rgb(230, 245, 255)'
        ),
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional summary
    st.subheader("Regional Summary")
    
    regional_summary = batch_scores.groupby('primary_region').agg({
        'batch_id': 'count',
        'risk_score': 'mean',
        'num_cases': 'sum'
    }).rename(columns={
        'batch_id': 'batches',
        'risk_score': 'avg_risk',
        'num_cases': 'total_cases'
    }).sort_values('avg_risk', ascending=False)
    
    st.dataframe(regional_summary, use_container_width=True)


if __name__ == "__main__":
    render_dashboard()
