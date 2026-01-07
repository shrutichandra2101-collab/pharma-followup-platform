#!/bin/bash

echo "======================================"
echo "Training Prioritization Model"
echo "======================================"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "ERROR: Virtual environment not found!"
    echo "Please run ./setup.sh first"
    exit 1
fi

cd ai_components/prioritization

echo ""
echo "Step 1: Generating training data..."
python3 data_generator.py

echo ""
echo "Step 2: Training models..."
python3 model.py

cd ../..

echo ""
echo "======================================"
echo "Training Complete!"
echo "======================================"
echo ""
echo "View results:"
echo "  - Metrics: evaluation/prioritization_metrics.json"
echo "  - Visualizations: evaluation/prioritization_*.png"
echo ""
echo "To view images (on Mac):"
echo "  open evaluation/prioritization_regression.png"
echo "  open evaluation/prioritization_classification_confusion_matrix.png"
echo "  open evaluation/prioritization_feature_importance.png"
echo ""
