"""
Smart Follow-Up Questionnaire Generator - Main Orchestrator Module
Orchestrates all questionnaire components end-to-end

Step 8: Run complete questionnaire generation and evaluation pipeline
"""

import pandas as pd
import json
import sys
import os
sys.path.append('../..')

from data_generator import QuestionnaireDataGenerator
from question_bank import QuestionBank
from selector_engine import QuestionSelector
from questionnaire_builder import QuestionnaireBuilder
from response_predictor import ResponseQualityPredictor
from evaluation_metrics import generate_evaluation_report
from visualizer import QuestionnaireVisualizer


class QuestionnaireGenerator:
    """Complete questionnaire generation pipeline."""
    
    def __init__(self):
        self.question_bank = QuestionBank()
        self.selector = QuestionSelector(self.question_bank)
        self.builder = QuestionnaireBuilder()
        self.predictor = ResponseQualityPredictor()
        self.visualizer = QuestionnaireVisualizer()
        
        self.training_data = None
        self.test_data = None
        self.metrics = None
    
    def run_full_pipeline(self, num_training=4000, num_test=1000, output_json=True):
        """
        Run complete questionnaire generation pipeline.
        
        Args:
            num_training: Number of training cases
            num_test: Number of test cases
            output_json: Save results to JSON
        """
        print("\n" + "="*70)
        print("SMART FOLLOW-UP QUESTIONNAIRE GENERATOR")
        print("="*70 + "\n")
        
        # Step 1: Question Bank Summary
        print("STEP 1: Question Bank Configuration")
        print("-"*70)
        self.question_bank.print_summary()
        
        # Step 2: Data Generation
        print("\n\nSTEP 2: Generating Training Data")
        print("-"*70)
        generator = QuestionnaireDataGenerator(num_samples=num_training)
        full_data = generator.generate_dataset()
        self.training_data, self.test_data = generator.generate_training_test_split(
            full_data, test_size=num_test/(num_training+num_test)
        )
        print(f"✓ Training data: {len(self.training_data)} cases")
        print(f"✓ Test data: {len(self.test_data)} cases")
        
        generator.print_summary(self.training_data)
        
        # Step 3: Question Selection Training
        print("\n\nSTEP 3: Training Selection Models")
        print("-"*70)
        self.selector.train_selector_model(self.training_data)
        self.selector.train_relevance_scorer(self.training_data)
        
        # Step 4: Response Prediction Training
        print("\n\nSTEP 4: Training Response Quality Predictor")
        print("-"*70)
        self.predictor.train(self.training_data)
        
        # Step 5: Generate Test Questionnaires
        print("\n\nSTEP 5: Generating Test Questionnaires")
        print("-"*70)
        self.test_data['questionnaire_id'] = [f"Q_{i}" for i in range(len(self.test_data))]
        
        questionnaires = []
        for idx, row in self.test_data.iterrows():
            # Select questions
            case_dict = row.to_dict()
            selected_questions = self.selector.select_questions(case_dict, max_questions=10)
            
            # Predict response quality
            predicted_quality = self.predictor.predict_probability(case_dict)
            
            # Build questionnaire
            questionnaire = self.builder.build_questionnaire(case_dict, selected_questions)
            questionnaire['predicted_response_quality'] = predicted_quality
            
            questionnaires.append(questionnaire)
        
        print(f"✓ Generated {len(questionnaires)} questionnaires")
        print(f"  Avg questions per questionnaire: {self.test_data['num_selected_questions'].mean():.1f}")
        print(f"  Avg completion time: {self.test_data['actual_completion_time'].mean()/60:.1f} minutes")
        
        # Step 6: Evaluate Questionnaires
        print("\n\nSTEP 6: Evaluating Questionnaire Effectiveness")
        print("-"*70)
        self.metrics = generate_evaluation_report(self.test_data)
        
        self._print_evaluation_summary()
        
        # Step 7: Generate Visualizations
        print("\n\nSTEP 7: Generating Visualizations")
        print("-"*70)
        self.visualizer.generate_all_visualizations(self.test_data, self.metrics)
        
        # Step 8: Save Results
        print("\n\nSTEP 8: Saving Results")
        print("-"*70)
        if output_json:
            self._save_results()
        
        return {
            'training_data': self.training_data,
            'test_data': self.test_data,
            'questionnaires': questionnaires,
            'metrics': self.metrics,
        }
    
    def _print_evaluation_summary(self):
        """Print evaluation summary."""
        metrics = self.metrics['overall_metrics']
        
        print(f"✓ Evaluation Complete")
        
        coverage = metrics['coverage']
        print(f"\n  Field Coverage:")
        print(f"    Average: {coverage['average_field_coverage']:.1%}")
        print(f"    Full Coverage Rate: {coverage['full_coverage_rate']:.1%}")
        
        quality = metrics['response_quality']
        print(f"\n  Response Quality:")
        print(f"    Avg Completion: {quality['average_completion_rate']:.1%}")
        print(f"    Avg Quality: {quality['average_response_quality']:.2f}/5")
        print(f"    High Quality %: {quality['high_quality_percentage']:.1%}")
        print(f"    Avg Satisfaction: {quality['average_satisfaction']:.2f}/5")
        
        roi = metrics['roi']
        print(f"\n  ROI Analysis:")
        print(f"    Average ROI: {roi['average_roi']:.1f}")
        print(f"    High ROI %: {roi['high_roi_percentage']:.1%}")
        
        precision = metrics['selection_precision']
        print(f"\n  Selection Precision:")
        print(f"    Accuracy: {precision['selection_accuracy']:.1%}")
        print(f"    Useful Rate: {precision['useful_question_rate']:.1%}")
    
    def _save_results(self):
        """Save results to files."""
        import os
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(base_dir, 'data', 'processed')
        eval_dir = os.path.join(base_dir, 'evaluation')
        
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        
        # Save training data
        train_path = os.path.join(data_dir, 'questionnaire_train.csv')
        self.training_data.to_csv(train_path, index=False)
        print(f"✓ Saved training data: {train_path}")
        
        # Save test data
        test_path = os.path.join(data_dir, 'questionnaire_test.csv')
        self.test_data.to_csv(test_path, index=False)
        print(f"✓ Saved test data: {test_path}")
        
        # Save metrics
        metrics_path = os.path.join(eval_dir, 'questionnaire_metrics.json')
        
        # Clean metrics for JSON serialization
        clean_metrics = self._clean_metrics_for_json(self.metrics)
        
        with open(metrics_path, 'w') as f:
            json.dump(clean_metrics, f, indent=2)
        print(f"✓ Saved metrics: {metrics_path}")
        
        # Save report
        report_path = os.path.join(eval_dir, 'QUESTIONNAIRE_ENGINE_REPORT.txt')
        with open(report_path, 'w') as f:
            f.write("SMART FOLLOW-UP QUESTIONNAIRE GENERATOR - REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("OVERVIEW\n")
            f.write("-"*70 + "\n")
            f.write(f"Training Cases: {len(self.training_data)}\n")
            f.write(f"Test Cases: {len(self.test_data)}\n")
            f.write(f"Total Questions in Bank: {self.question_bank.get_total_questions()}\n\n")
            
            f.write("QUESTION BANK SUMMARY\n")
            f.write("-"*70 + "\n")
            for category in self.question_bank.get_all_categories():
                questions = self.question_bank.get_questions_by_category(category)
                f.write(f"{category:20s}: {len(questions)} questions\n")
            f.write("\n")
            
            f.write("FIELD COVERAGE RESULTS\n")
            f.write("-"*70 + "\n")
            coverage = self.metrics['overall_metrics']['coverage']
            f.write(f"Average Field Coverage: {coverage['average_field_coverage']:.1%}\n")
            f.write(f"Full Coverage Rate: {coverage['full_coverage_rate']:.1%}\n\n")
            
            f.write("RESPONSE QUALITY RESULTS\n")
            f.write("-"*70 + "\n")
            quality = self.metrics['overall_metrics']['response_quality']
            f.write(f"Average Completion Rate: {quality['average_completion_rate']:.1%}\n")
            f.write(f"Average Response Quality: {quality['average_response_quality']:.2f}/5\n")
            f.write(f"Average Satisfaction: {quality['average_satisfaction']:.2f}/5\n")
            f.write(f"High Quality Rate: {quality['high_quality_percentage']:.1%}\n\n")
            
            f.write("ROI ANALYSIS\n")
            f.write("-"*70 + "\n")
            roi = self.metrics['overall_metrics']['roi']
            f.write(f"Average ROI: {roi['average_roi']:.1f}\n")
            f.write(f"High ROI Cases: {roi['high_roi_percentage']:.1%}\n\n")
            
            f.write("PERFORMANCE BY VALIDATION STATUS\n")
            f.write("-"*70 + "\n")
            for status, stats in self.metrics['analysis_by_status'].items():
                f.write(f"\n{status}:\n")
                f.write(f"  Count: {stats['count']}\n")
                f.write(f"  Avg Effectiveness: {stats['avg_effectiveness']:.1f}/100\n")
                f.write(f"  Time to Complete: {stats['avg_time_minutes']:.1f} min\n")
                f.write(f"  Satisfaction: {stats['avg_satisfaction']:.2f}/5\n")
        
        print(f"✓ Saved report: {report_path}")
    
    def _clean_metrics_for_json(self, metrics: dict) -> dict:
        """Clean metrics dict for JSON serialization."""
        clean = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                clean[key] = self._clean_metrics_for_json(value)
            elif isinstance(value, (int, float, str, bool, type(None))):
                clean[key] = value
            elif isinstance(value, pd.Series):
                clean[key] = value.to_dict()
            else:
                clean[key] = str(value)
        return clean


if __name__ == "__main__":
    # Run full pipeline
    engine = QuestionnaireGenerator()
    results = engine.run_full_pipeline(num_training=4000, num_test=1000)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print("\nQuestionnaire Generator Successfully Built!")
    print("\nGenerated Artifacts:")
    print("  ✓ Question bank with 35+ clinical questions")
    print("  ✓ 5,000 training cases with realistic gaps")
    print("  ✓ Decision tree for question selection")
    print("  ✓ Logistic regression for response quality prediction")
    print("  ✓ 1,000 test questionnaires")
    print("  ✓ Comprehensive evaluation metrics")
    print("  ✓ 8 professional visualizations")
    print("  ✓ Detailed reports and analysis")
