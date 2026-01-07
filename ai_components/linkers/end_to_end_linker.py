"""
End-to-End Pipeline Linker
Orchestrates complete data flow through Components 1 → 2 → 3
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import sys
import os

sys.path.append('../..')
from linkers.prioritization_to_validation import PrioritizationToValidationLinker
from linkers.validation_to_questionnaire import ValidationToQuestionnaireLinker


class EndToEndPipelineLinker:
    """Orchestrates complete pipeline from Prioritization through Questionnaire."""
    
    def __init__(self):
        """Initialize end-to-end linker with component linkers."""
        self.prio_to_val_linker = PrioritizationToValidationLinker()
        self.val_to_quest_linker = ValidationToQuestionnaireLinker()
        
        self.prioritization_output = None
        self.validation_output = None
        self.questionnaire_output = None
        self.pipeline_metrics = {}
    
    def run_pipeline(self, 
                     prioritization_df: pd.DataFrame,
                     validation_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete pipeline linking all components.
        
        Args:
            prioritization_df: Output from Prioritization Model (Component 1)
            validation_df: Output from Validation Model (Component 2)
            
        Returns:
            dict with questionnaire input and pipeline metrics
        """
        print("\n" + "="*70)
        print("END-TO-END PIPELINE LINKER")
        print("Components 1 → 2 → 3 Integration")
        print("="*70 + "\n")
        
        start_time = datetime.now()
        
        # Stage 1: Prioritization → Validation
        print("STAGE 1: Prioritization → Validation")
        print("-" * 70)
        validation_input = self.prio_to_val_linker.transform(prioritization_df)
        self.prioritization_output = prioritization_df
        print(f"✓ Stage 1 complete: {len(validation_input)} records prepared for validation\n")
        
        # Stage 2: Validation → Questionnaire
        print("STAGE 2: Validation → Questionnaire")
        print("-" * 70)
        questionnaire_input = self.val_to_quest_linker.transform(validation_df)
        self.validation_output = validation_df
        self.questionnaire_output = questionnaire_input
        print(f"✓ Stage 2 complete: {len(questionnaire_input)} records prepared for questionnaire generation\n")
        
        # Calculate pipeline metrics
        end_time = datetime.now()
        self._calculate_metrics(prioritization_df, validation_df, questionnaire_input, 
                               end_time - start_time)
        
        # Print pipeline summary
        self._print_summary()
        
        return {
            'questionnaire_input': questionnaire_input,
            'pipeline_metrics': self.pipeline_metrics,
            'validation_input': validation_input,
            'stages_completed': 2
        }
    
    def _calculate_metrics(self, prio_df: pd.DataFrame, val_df: pd.DataFrame, 
                          quest_df: pd.DataFrame, duration) -> None:
        """Calculate pipeline metrics."""
        self.pipeline_metrics = {
            'stage_1_records_in': len(prio_df),
            'stage_1_records_out': len(val_df),
            'stage_2_records_in': len(val_df),
            'stage_2_records_out': len(quest_df),
            'total_records_processed': len(quest_df),
            'pipeline_duration_seconds': duration.total_seconds(),
            'records_per_second': len(quest_df) / duration.total_seconds() if duration.total_seconds() > 0 else 0,
            'timestamp': datetime.now().isoformat(),
            
            # Quality metrics
            'avg_quality_score': val_df['quality_score'].mean() if 'quality_score' in val_df.columns else 0,
            'avg_completeness': val_df['completeness_score'].mean() if 'completeness_score' in val_df.columns else 0,
            'avg_questionnaire_priority': quest_df['questionnaire_priority'].mean() if 'questionnaire_priority' in quest_df.columns else 0,
            
            # Distribution metrics
            'validation_status_distribution': val_df['validation_status'].value_counts().to_dict() if 'validation_status' in val_df.columns else {},
            'questionnaire_type_distribution': quest_df['questionnaire_type'].value_counts().to_dict() if 'questionnaire_type' in quest_df.columns else {},
            'difficulty_distribution': quest_df['expected_difficulty'].value_counts().to_dict() if 'expected_difficulty' in quest_df.columns else {},
            
            # Time metrics
            'avg_expected_completion_minutes': quest_df['expected_completion_minutes'].mean() if 'expected_completion_minutes' in quest_df.columns else 0,
        }
    
    def _print_summary(self) -> None:
        """Print pipeline execution summary."""
        print("\n" + "="*70)
        print("PIPELINE SUMMARY")
        print("="*70 + "\n")
        
        metrics = self.pipeline_metrics
        
        print("Processing Statistics:")
        print(f"  Total records processed: {metrics['total_records_processed']:,}")
        print(f"  Pipeline duration: {metrics['pipeline_duration_seconds']:.2f} seconds")
        print(f"  Throughput: {metrics['records_per_second']:.2f} records/second\n")
        
        print("Quality Metrics:")
        print(f"  Avg quality score: {metrics['avg_quality_score']:.2f}/100")
        print(f"  Avg completeness: {metrics['avg_completeness']:.2f}/100")
        print(f"  Avg questionnaire priority: {metrics['avg_questionnaire_priority']:.2f}/1.0\n")
        
        if metrics['validation_status_distribution']:
            print("Validation Status Distribution:")
            for status, count in metrics['validation_status_distribution'].items():
                pct = (count / metrics['stage_2_records_in']) * 100
                print(f"  {status}: {count} ({pct:.1f}%)")
            print()
        
        if metrics['questionnaire_type_distribution']:
            print("Questionnaire Type Distribution:")
            for qtype, count in metrics['questionnaire_type_distribution'].items():
                pct = (count / metrics['total_records_processed']) * 100
                print(f"  {qtype}: {count} ({pct:.1f}%)")
            print()
        
        if metrics['difficulty_distribution']:
            print("Expected Difficulty Distribution:")
            for difficulty, count in metrics['difficulty_distribution'].items():
                pct = (count / metrics['total_records_processed']) * 100
                print(f"  {difficulty}: {count} ({pct:.1f}%)")
            print()
        
        print(f"Avg expected completion time: {metrics['avg_expected_completion_minutes']:.2f} minutes")
    
    def save_pipeline_report(self, output_dir: str = 'evaluation') -> str:
        """
        Save pipeline report to JSON.
        
        Args:
            output_dir: Directory to save report
            
        Returns:
            Path to saved report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, 'pipeline_linker_report.json')
        
        report = {
            'pipeline_name': 'End-to-End Component Linker (1→2→3)',
            'execution_date': datetime.now().isoformat(),
            'metrics': self.pipeline_metrics,
            'linker_schemas': {
                'stage_1_output': self.prio_to_val_linker.get_schema(),
                'stage_2_output': self.val_to_quest_linker.get_schema()
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Pipeline report saved to {report_path}")
        return report_path
    
    def save_questionnaire_input(self, output_dir: str = 'data/processed') -> str:
        """
        Save questionnaire input to CSV.
        
        Args:
            output_dir: Directory to save data
            
        Returns:
            Path to saved file
        """
        if self.questionnaire_output is None:
            raise ValueError("Pipeline not yet executed. Call run_pipeline() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for CSV (convert lists/dicts to JSON strings)
        export_df = self.questionnaire_output.copy()
        
        for col in export_df.columns:
            if col in ['missing_fields', 'fields_by_category']:
                export_df[col] = export_df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
                )
        
        filepath = os.path.join(output_dir, 'pipeline_questionnaire_input.csv')
        export_df.to_csv(filepath, index=False)
        
        print(f"✓ Questionnaire input saved to {filepath}")
        return filepath
    
    def get_questionnaire_input(self) -> Optional[pd.DataFrame]:
        """Get questionnaire input dataframe."""
        return self.questionnaire_output
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        return self.pipeline_metrics
    
    def validate_schemas(self) -> Dict[str, bool]:
        """
        Validate that outputs match expected schemas.
        
        Returns:
            dict with validation results
        """
        results = {
            'stage_1_schema_valid': self._validate_stage_1_schema(),
            'stage_2_schema_valid': self._validate_stage_2_schema(),
        }
        
        all_valid = all(results.values())
        results['pipeline_valid'] = all_valid
        
        return results
    
    def _validate_stage_1_schema(self) -> bool:
        """Validate stage 1 output schema."""
        if self.validation_output is None:
            return False
        
        required = ['case_id', 'priority_score', 'validation_category']
        return all(col in self.validation_output.columns for col in required)
    
    def _validate_stage_2_schema(self) -> bool:
        """Validate stage 2 output schema."""
        if self.questionnaire_output is None:
            return False
        
        required = ['case_id', 'questionnaire_type', 'expected_difficulty', 
                   'expected_completion_minutes', 'missing_fields']
        return all(col in self.questionnaire_output.columns for col in required)
