"""
Medical Named Entity Recognition - Streamlit Dashboard
Interactive dashboard for NER model testing and monitoring

Step 6: Create Streamlit dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import sys
sys.path.append('../..')

from model import NERModelTrainer
from evaluation_metrics import NERMetrics
import os
import json


st.set_page_config(
    page_title="Medical NER Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .header-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .entity-box {
        border: 2px solid #667eea;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        background-color: #f8f9ff;
    }
    .entity-text {
        font-weight: bold;
        font-size: 18px;
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load trained NER model."""
    trainer = NERModelTrainer()
    model_path = '../../../data/models/ner_model.pkl'
    if os.path.exists(model_path):
        trainer.load_model(model_path)
        return trainer
    else:
        st.warning("Model file not found. Please run the NER pipeline first.")
        return None


@st.cache_data
def load_metrics():
    """Load pre-computed metrics."""
    metrics_path = '../../../evaluation/ner_metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


@st.cache_data
def load_test_data():
    """Load test data."""
    test_path = '../../../data/processed/ner_test.csv'
    if os.path.exists(test_path):
        return pd.read_csv(test_path)
    return None


def highlight_entities(text: str, entities: List[Dict[str, Any]]) -> str:
    """Create HTML with highlighted entities."""
    # Sort entities by start position (reverse to not mess up indices)
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    entity_colors = {
        'DRUG': '#FF6B6B',
        'DOSAGE': '#4ECDC4',
        'ROUTE': '#45B7D1',
        'DURATION': '#FFA07A',
        'CONDITION': '#98D8C8',
        'OUTCOME': '#F7DC6F',
        'FREQUENCY': '#BB8FCE',
        'SEVERITY': '#EC7063'
    }
    
    for entity in sorted_entities:
        start = entity['start']
        end = entity['end']
        color = entity_colors.get(entity['type'], '#CCCCCC')
        confidence = entity.get('confidence', 1.0)
        
        replacement = f"<mark style='background-color: {color}; padding: 2px 4px; border-radius: 3px; font-weight: bold;' title='{entity['type']} (conf: {confidence:.2f})'>{text[start:end]}</mark>"
        text = text[:start] + replacement + text[end:]
    
    return text


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown("""
    <div class='header-box'>
        <h1>🔬 Medical Named Entity Recognition Dashboard</h1>
        <p>Extract and analyze medical entities from clinical narratives</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and data
    trainer = load_model()
    metrics = load_metrics()
    test_df = load_test_data()
    
    if trainer is None or trainer.model is None:
        st.error("❌ Model not found. Please run the NER pipeline to train the model.")
        return
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Entity Extraction", "Model Performance", "Analytics", "Test Data Explorer"]
    )
    
    # Page 1: Entity Extraction
    if page == "Entity Extraction":
        st.header("🔍 Entity Extraction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Enter Medical Narrative")
            user_text = st.text_area(
                "Paste or type a medical narrative:",
                height=200,
                placeholder="E.g., Patient reported mild recovery after taking Aspirin 500 mg orally for hypertension..."
            )
        
        with col2:
            st.subheader("Quick Examples")
            examples = [
                "Patient recovered after taking Ibuprofen 400 mg orally for 2 weeks for arthritis.",
                "Severe adverse event: hospitalized patient with diabetes on Metformin 1000 mg daily developed sepsis.",
                "Moderate pneumonia treated with Amoxicillin 250 mg three times daily for 1 month."
            ]
            
            if st.button("Load Example 1"):
                user_text = examples[0]
            if st.button("Load Example 2"):
                user_text = examples[1]
            if st.button("Load Example 3"):
                user_text = examples[2]
        
        if user_text:
            # Extract entities
            extracted_entities = trainer.model.extract_entities(user_text)
            
            # Display results
            st.subheader("Extracted Entities")
            
            if extracted_entities:
                # Create HTML visualization
                html_text = user_text
                for entity in sorted(extracted_entities, key=lambda x: x['start'], reverse=True):
                    start = entity['start']
                    end = entity['end']
                    color_map = {
                        'DRUG': '#FF6B6B',
                        'DOSAGE': '#4ECDC4',
                        'ROUTE': '#45B7D1',
                        'DURATION': '#FFA07A',
                        'CONDITION': '#98D8C8',
                        'OUTCOME': '#F7DC6F',
                        'FREQUENCY': '#BB8FCE',
                        'SEVERITY': '#EC7063'
                    }
                    color = color_map.get(entity['type'], '#CCCCCC')
                    replacement = f"<mark style='background-color: {color}; padding: 3px 6px; border-radius: 3px; font-weight: bold; cursor: pointer;'>{user_text[start:end]}</mark>"
                    html_text = html_text[:start] + replacement + html_text[end:]
                
                st.markdown(html_text, unsafe_allow_html=True)
                
                # Entities table
                st.subheader("Entity Details")
                entity_df = pd.DataFrame([
                    {
                        'Entity Type': e['type'],
                        'Text': e['text'],
                        'Confidence': f"{e.get('confidence', 1.0):.2%}",
                        'Position': f"{e['start']}-{e['end']}"
                    }
                    for e in extracted_entities
                ])
                
                st.dataframe(entity_df, use_container_width=True)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Entities", len(extracted_entities))
                with col2:
                    st.metric("Entity Types", len(set(e['type'] for e in extracted_entities)))
                with col3:
                    avg_confidence = np.mean([e.get('confidence', 1.0) for e in extracted_entities])
                    st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                with col4:
                    coverage = len(set(e['type'] for e in extracted_entities)) / 8
                    st.metric("Type Coverage", f"{coverage:.1%}")
            else:
                st.info("ℹ️ No entities found in the provided text.")
    
    # Page 2: Model Performance
    elif page == "Model Performance":
        st.header("📈 Model Performance")
        
        if metrics:
            # Overall metrics
            st.subheader("Overall Performance")
            overall = metrics.get('overall', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Precision",
                    f"{overall.get('precision', 0):.4f}",
                    help="Proportion of extracted entities that are correct"
                )
            with col2:
                st.metric(
                    "Recall",
                    f"{overall.get('recall', 0):.4f}",
                    help="Proportion of true entities that were extracted"
                )
            with col3:
                st.metric(
                    "F1-Score",
                    f"{overall.get('f1', 0):.4f}",
                    help="Harmonic mean of precision and recall"
                )
            
            # Per-entity metrics
            st.subheader("Per-Entity-Type Performance")
            
            by_type = metrics.get('by_entity_type', {})
            if by_type:
                entity_metrics_df = pd.DataFrame([
                    {
                        'Entity Type': entity_type,
                        'Precision': f"{m.get('precision', 0):.4f}",
                        'Recall': f"{m.get('recall', 0):.4f}",
                        'F1-Score': f"{m.get('f1', 0):.4f}",
                        'Support': m.get('support', 0)
                    }
                    for entity_type, m in sorted(by_type.items())
                ])
                
                st.dataframe(entity_metrics_df, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # F1 by entity type
                f1_scores = [by_type.get(et, {}).get('f1', 0) for et in 
                            ['DRUG', 'DOSAGE', 'ROUTE', 'DURATION', 'CONDITION', 'OUTCOME', 'FREQUENCY', 'SEVERITY']]
                st.bar_chart({
                    'Entity Type': ['DRUG', 'DOSAGE', 'ROUTE', 'DURATION', 'CONDITION', 'OUTCOME', 'FREQUENCY', 'SEVERITY'],
                    'F1-Score': f1_scores
                })
            
            with col2:
                # Precision vs Recall
                precisions = [by_type.get(et, {}).get('precision', 0) for et in 
                             ['DRUG', 'DOSAGE', 'ROUTE', 'DURATION', 'CONDITION', 'OUTCOME', 'FREQUENCY', 'SEVERITY']]
                recalls = [by_type.get(et, {}).get('recall', 0) for et in 
                          ['DRUG', 'DOSAGE', 'ROUTE', 'DURATION', 'CONDITION', 'OUTCOME', 'FREQUENCY', 'SEVERITY']]
                
                st.line_chart({
                    'Precision': precisions,
                    'Recall': recalls
                })
        else:
            st.warning("⚠️ Metrics data not available.")
    
    # Page 3: Analytics
    elif page == "Analytics":
        st.header("📊 Analytics & Insights")
        
        if test_df is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Test Cases", len(test_df))
            with col2:
                st.metric("Avg Entities", f"{test_df['entity_count'].mean():.2f}")
            with col3:
                st.metric("Avg Narrative Length", f"{test_df['narrative_length'].mean():.0f} chars")
            with col4:
                complexity_counts = test_df['complexity'].value_counts()
                simple_pct = complexity_counts.get('simple', 0) / len(test_df) * 100
                st.metric("Simple Narratives", f"{simple_pct:.1f}%")
            
            # Complexity breakdown
            st.subheader("Narrative Complexity Distribution")
            complexity_data = test_df['complexity'].value_counts()
            st.bar_chart(complexity_data)
            
            # Entity count distribution
            st.subheader("Entity Count Distribution")
            st.histogram(test_df['entity_count'], bins=20)
            
            # Entity type distribution
            st.subheader("Entity Type Distribution")
            entity_dist = {}
            for entities_list in test_df['entities']:
                for entity in entities_list:
                    entity_type = entity['type']
                    entity_dist[entity_type] = entity_dist.get(entity_type, 0) + 1
            
            if entity_dist:
                st.bar_chart(pd.DataFrame.from_dict(entity_dist, orient='index', columns=['Count']))
    
    # Page 4: Test Data Explorer
    elif page == "Test Data Explorer":
        st.header("🔎 Test Data Explorer")
        
        if test_df is not None:
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                complexity_filter = st.multiselect(
                    "Filter by Complexity:",
                    test_df['complexity'].unique(),
                    default=test_df['complexity'].unique()
                )
            
            with col2:
                min_entities = st.slider("Min Entities:", 0, int(test_df['entity_count'].max()), 0)
            
            with col3:
                max_entities = st.slider("Max Entities:", 0, int(test_df['entity_count'].max()), 
                                        int(test_df['entity_count'].max()))
            
            # Apply filters
            filtered_df = test_df[
                (test_df['complexity'].isin(complexity_filter)) &
                (test_df['entity_count'] >= min_entities) &
                (test_df['entity_count'] <= max_entities)
            ]
            
            st.subheader(f"Showing {len(filtered_df)} of {len(test_df)} test cases")
            
            # Sample selector
            if len(filtered_df) > 0:
                sample_idx = st.selectbox(
                    "Select a sample to view:",
                    range(len(filtered_df)),
                    format_func=lambda i: f"Case {filtered_df.iloc[i]['case_id']} - {filtered_df.iloc[i]['complexity']} ({filtered_df.iloc[i]['entity_count']} entities)"
                )
                
                sample = filtered_df.iloc[sample_idx]
                
                st.subheader("Narrative")
                st.text(sample['narrative'])
                
                st.subheader("Extracted Entities")
                entities = sample['entities']
                
                for entity in entities:
                    st.markdown(f"""
                    <div class='entity-box'>
                        <span class='entity-text'>{entity['text']}</span>
                        <br/>
                        <small>Type: <strong>{entity['type']}</strong> | Position: {entity['start']}-{entity['end']}</small>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; font-size: 12px; color: gray;'>"
        "Medical NER Dashboard | Pharma Follow-up Platform | Component 3"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
