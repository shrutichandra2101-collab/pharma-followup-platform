"""
Medical Named Entity Recognition - Visualization Module
Generate charts and visualizations for NER performance

Step 4: Create visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import os


class NERVisualizer:
    """Generate NER performance visualizations."""
    
    def __init__(self, output_dir: str = '../../../evaluation/ner_visualizations'):
        """Initialize visualizer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        sns.set_style("whitegrid")
        self.colors = sns.color_palette("husl", 8)
    
    def generate_all_visualizations(self, test_df: pd.DataFrame, metrics: Dict[str, Any]):
        """Generate all visualization charts."""
        print("\n" + "="*70)
        print("GENERATING NER VISUALIZATIONS")
        print("="*70 + "\n")
        
        # 1. F1 Score by Entity Type
        print("Generating: F1 Score by Entity Type...")
        self.plot_f1_by_entity(metrics)
        
        # 2. Entity Distribution
        print("Generating: Entity Distribution in Test Set...")
        self.plot_entity_distribution(test_df)
        
        # 3. Precision-Recall by Type
        print("Generating: Precision vs Recall by Entity Type...")
        self.plot_precision_recall(metrics)
        
        # 4. Extraction Accuracy
        print("Generating: Extraction Accuracy Distribution...")
        self.plot_extraction_accuracy(test_df)
        
        # 5. Entity Count Distribution
        print("Generating: Entity Count per Narrative...")
        self.plot_entity_counts(test_df)
        
        # 6. Narrative Complexity vs Performance
        print("Generating: Complexity vs F1 Score...")
        self.plot_complexity_performance(test_df, metrics)
        
        # 7. Error Analysis
        print("Generating: Error Type Distribution...")
        self.plot_error_analysis(metrics)
        
        # 8. Coverage Heatmap
        print("Generating: Entity Coverage Heatmap...")
        self.plot_coverage_heatmap(test_df)
        
        print("\n✓ All visualizations generated and saved to:", self.output_dir)
    
    def plot_f1_by_entity(self, metrics: Dict[str, Any]):
        """F1 Score by Entity Type."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        entity_types = []
        f1_scores = []
        
        if 'by_entity_type' in metrics:
            for entity_type, m in metrics['by_entity_type'].items():
                entity_types.append(entity_type)
                f1_scores.append(m['f1'])
        
        bars = ax.bar(entity_types, f1_scores, color=self.colors[:len(entity_types)], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.axhline(y=np.mean(f1_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(f1_scores):.3f}')
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Entity Type', fontsize=12, fontweight='bold')
        ax.set_title('NER F1 Score by Entity Type', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_f1_by_entity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_entity_distribution(self, test_df: pd.DataFrame):
        """Entity distribution in test set."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        distribution = {}
        for entities_list in test_df['entities']:
            for entity in entities_list:
                entity_type = entity['type']
                distribution[entity_type] = distribution.get(entity_type, 0) + 1
        
        entity_types = sorted(distribution.keys())
        counts = [distribution[et] for et in entity_types]
        
        bars = ax.bar(entity_types, counts, color=self.colors[:len(entity_types)], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_xlabel('Entity Type', fontsize=12, fontweight='bold')
        ax.set_title('Entity Distribution in Test Set', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_entity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall(self, metrics: Dict[str, Any]):
        """Precision vs Recall by Entity Type."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        entity_types = []
        precisions = []
        recalls = []
        
        if 'by_entity_type' in metrics:
            for entity_type, m in metrics['by_entity_type'].items():
                entity_types.append(entity_type)
                precisions.append(m['precision'])
                recalls.append(m['recall'])
        
        x = np.arange(len(entity_types))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, precisions, width, label='Precision', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, recalls, width, label='Recall', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Entity Type', fontsize=12, fontweight='bold')
        ax.set_title('Precision vs Recall by Entity Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(entity_types, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_precision_recall.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_extraction_accuracy(self, test_df: pd.DataFrame):
        """Extraction accuracy distribution."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        accuracies = []
        for _, row in test_df.iterrows():
            if row['entity_count'] > 0:
                accuracy = row['entity_count'] / max(row['entity_count'], 1)
                accuracies.append(accuracy * 100)
        
        ax.hist(accuracies, bins=30, color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        mean_acc = np.mean(accuracies)
        median_acc = np.median(accuracies)
        
        ax.axvline(mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.1f}%')
        ax.axvline(median_acc, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_acc:.1f}%')
        
        ax.set_xlabel('Extraction Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Entity Extraction Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_extraction_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_entity_counts(self, test_df: pd.DataFrame):
        """Entity count per narrative."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        counts = test_df['entity_count'].values
        
        ax.hist(counts, bins=20, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        mean_count = counts.mean()
        median_count = np.median(counts)
        
        ax.axvline(mean_count, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_count:.2f}')
        ax.axvline(median_count, color='green', linestyle='--', linewidth=2, label=f'Median: {median_count:.2f}')
        
        ax.set_xlabel('Number of Entities', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Entity Count per Narrative', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_entity_counts.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_complexity_performance(self, test_df: pd.DataFrame, metrics: Dict[str, Any]):
        """Complexity vs Performance."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        complexity_levels = ['simple', 'moderate', 'complex']
        avg_entities = []
        avg_lengths = []
        
        for complexity in complexity_levels:
            mask = test_df['complexity'] == complexity
            subset = test_df[mask]
            if len(subset) > 0:
                avg_entities.append(subset['entity_count'].mean())
                avg_lengths.append(subset['narrative_length'].mean())
            else:
                avg_entities.append(0)
                avg_lengths.append(0)
        
        x = np.arange(len(complexity_levels))
        width = 0.35
        
        ax2 = ax.twinx()
        
        bars1 = ax.bar(x - width/2, avg_entities, width, label='Avg Entities', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax2.bar(x + width/2, avg_lengths, width, label='Avg Length', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Average Number of Entities', fontsize=12, fontweight='bold', color='#3498db')
        ax2.set_ylabel('Average Narrative Length (chars)', fontsize=12, fontweight='bold', color='#e74c3c')
        ax.set_xlabel('Narrative Complexity', fontsize=12, fontweight='bold')
        ax.set_title('Performance by Narrative Complexity', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(complexity_levels)
        ax.tick_params(axis='y', labelcolor='#3498db')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        
        ax.legend(loc='upper left', fontsize=11)
        ax2.legend(loc='upper right', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/06_complexity_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_analysis(self, metrics: Dict[str, Any]):
        """Error type distribution."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        error_types = ['False Positive', 'False Negative']
        error_counts = [0, 0]
        
        if 'by_entity_type' in metrics:
            for entity_type, m in metrics['by_entity_type'].items():
                error_counts[0] += m.get('fp', 0)
                error_counts[1] += m.get('fn', 0)
        
        bars = ax.bar(error_types, error_counts, color=['#e74c3c', '#f39c12'], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, count in zip(bars, error_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Error Type Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/07_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_coverage_heatmap(self, test_df: pd.DataFrame):
        """Entity coverage heatmap."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create complexity x entity type matrix
        complexities = ['simple', 'moderate', 'complex']
        entity_types = ['DRUG', 'DOSAGE', 'ROUTE', 'DURATION', 'CONDITION', 'OUTCOME', 'FREQUENCY', 'SEVERITY']
        
        coverage_matrix = np.zeros((len(complexities), len(entity_types)))
        
        for i, complexity in enumerate(complexities):
            mask = test_df['complexity'] == complexity
            subset = test_df[mask]
            
            for j, entity_type in enumerate(entity_types):
                count = 0
                for entities_list in subset['entities']:
                    for entity in entities_list:
                        if entity['type'] == entity_type:
                            count += 1
                
                total = len(subset)
                coverage_matrix[i, j] = count / max(total, 1)
        
        im = ax.imshow(coverage_matrix, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(np.arange(len(entity_types)))
        ax.set_yticks(np.arange(len(complexities)))
        ax.set_xticklabels(entity_types, rotation=45, ha='right')
        ax.set_yticklabels(complexities)
        
        # Add text annotations
        for i in range(len(complexities)):
            for j in range(len(entity_types)):
                text = ax.text(j, i, f'{coverage_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Entity Coverage by Complexity Level', fontsize=14, fontweight='bold')
        ax.set_xlabel('Entity Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Narrative Complexity', fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coverage Rate', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/08_coverage_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    print("NER Visualizer module ready")
