"""Common utilities for data generation and evaluation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import json
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def save_metrics(metrics: Dict[str, Any], output_path: str):
    """Save metrics to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_path}")

def load_metrics(metrics_path: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix", save_path=None):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.tight_layout()
    return fig

def plot_feature_importance(feature_names, importance_values, title="Feature Importance", 
                            top_n=20, save_path=None):
    """Plot feature importance."""
    # Sort features by importance
    indices = np.argsort(importance_values)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importance_values[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.invert_yaxis()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Feature importance plot saved to {save_path}")
    
    plt.tight_layout()
    return fig

def plot_training_curves(train_scores, val_scores, metric_name="Score", save_path=None):
    """Plot training and validation curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_scores) + 1)
    ax.plot(epochs, train_scores, 'b-', label=f'Training {metric_name}', linewidth=2)
    ax.plot(epochs, val_scores, 'r-', label=f'Validation {metric_name}', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Training curves saved to {save_path}")
    
    plt.tight_layout()
    return fig

def print_classification_report(y_true, y_pred, labels=None):
    """Print detailed classification metrics."""
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1-Score: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred, target_names=labels))
    print("="*60 + "\n")

# Pharmacovigilance-specific constants
EVENT_TYPES = [
    "Cardiac Disorders", "Gastrointestinal Disorders", "Nervous System Disorders",
    "Skin Disorders", "Respiratory Disorders", "Blood Disorders", 
    "Psychiatric Disorders", "Renal Disorders", "Hepatic Disorders", "Other"
]

SERIOUSNESS_CRITERIA = [
    "Death", "Life-threatening", "Hospitalization", "Disability", 
    "Congenital Anomaly", "Other Medically Important"
]

REGIONS = ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East", "Africa"]

REPORTER_TYPES = ["Healthcare Professional", "Patient", "Pharmacist", "Other HCP", "Consumer"]

MANDATORY_FIELDS = [
    "patient_age", "patient_gender", "event_description", "drug_name", 
    "dose", "route", "start_date", "event_date", "outcome", "reporter_type"
]
