#!/bin/bash

# Validation Dashboard Launcher

echo "======================================"
echo "Validation Engine - Streamlit Dashboard"
echo "======================================"
echo ""

# Check if venv is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source ../../venv/bin/activate
fi

# Launch dashboard
echo "Starting Streamlit dashboard..."
echo ""
echo "📊 Dashboard will open at: http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

/Users/shruti/Projects/pharma-followup-platform/venv/bin/streamlit run dashboard.py \
    --client.showErrorDetails=true \
    --logger.level=info
