#!/bin/bash

# Geospatial Signal Detection - Dashboard Launcher
# Interactive Streamlit interface for batch anomaly monitoring

echo "=================================================="
echo "Geospatial Signal Detection Dashboard"
echo "=================================================="
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed. Installing..."
    pip install streamlit plotly
fi

# Check if signal detection data exists
if [ ! -f "signal_detection_results/signal_detection_data.csv" ]; then
    echo "Signal detection data not found."
    echo "Running pipeline first..."
    python signal_detector.py
    echo ""
fi

echo "Launching Streamlit dashboard..."
echo "Open your browser to http://localhost:8501"
echo ""

streamlit run dashboard.py
