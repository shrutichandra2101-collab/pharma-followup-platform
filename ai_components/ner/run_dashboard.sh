#!/bin/bash
# Medical NER Dashboard - Streamlit App Launcher
# Run: bash ai_components/ner/run_dashboard.sh

echo ""
echo "=========================================="
echo "🔬 Medical NER Dashboard"
echo "=========================================="
echo ""
echo "Starting Streamlit dashboard..."
echo "Open your browser to: http://localhost:8501"
echo ""

cd "$(dirname "$0")"
streamlit run dashboard.py
