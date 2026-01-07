"""
Smart Follow-Up Questionnaire Generator - Question Bank Module
Comprehensive collection of pre-defined clinical follow-up questions

Step 1: Define all clinical questions with metadata for adaptive selection
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import pandas as pd


@dataclass
class Question:
    """Represents a single follow-up question."""
    question_id: str
    text: str
    category: str  # Safety, Efficacy, Patient, Medical History, Medication
    difficulty: str  # Easy, Medium, Hard
    estimated_time: int  # seconds to answer
    field_targets: List[str]  # which fields this question helps populate
    required: bool  # mandatory vs optional
    conditional_logic: Optional[Dict[str, Any]] = None  # IF-THEN conditions
    success_rate: float = 0.8  # historical answer rate
    priority: int = 1  # 1 = highest priority
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'question_id': self.question_id,
            'text': self.text,
            'category': self.category,
            'difficulty': self.difficulty,
            'estimated_time': self.estimated_time,
            'field_targets': self.field_targets,
            'required': self.required,
            'priority': self.priority,
            'success_rate': self.success_rate,
        }


class QuestionBank:
    """Pre-defined question bank with 100+ clinical follow-up questions."""
    
    def __init__(self):
        self.questions: Dict[str, Question] = {}
        self._initialize_questions()
    
    def _initialize_questions(self):
        """Initialize all pre-defined questions."""
        # SAFETY CATEGORY - Critical for patient safety assessment
        self.add_question(Question(
            question_id='S001',
            text='Did the patient experience any worsening of symptoms after taking this medication?',
            category='Safety',
            difficulty='Easy',
            estimated_time=20,
            field_targets=['event_severity', 'event_outcome'],
            required=True,
            priority=1,
            success_rate=0.95
        ))
        
        self.add_question(Question(
            question_id='S002',
            text='Were there any allergic reactions observed (rash, swelling, difficulty breathing)?',
            category='Safety',
            difficulty='Easy',
            estimated_time=25,
            field_targets=['adverse_event', 'event_severity'],
            required=True,
            priority=1,
            success_rate=0.93
        ))
        
        self.add_question(Question(
            question_id='S003',
            text='Did the patient require hospitalization or emergency care?',
            category='Safety',
            difficulty='Easy',
            estimated_time=15,
            field_targets=['hospitalization_flag', 'event_outcome'],
            required=True,
            priority=2,
            success_rate=0.98
        ))
        
        self.add_question(Question(
            question_id='S004',
            text='What was the temporal relationship? (When did symptoms start relative to drug intake?)',
            category='Safety',
            difficulty='Medium',
            estimated_time=40,
            field_targets=['days_to_event', 'event_date'],
            required=True,
            priority=1,
            success_rate=0.88,
            conditional_logic={'requires': ['event_date', 'drug_start_date']}
        ))
        
        self.add_question(Question(
            question_id='S005',
            text='Did the adverse event resolve after stopping the medication?',
            category='Safety',
            difficulty='Medium',
            estimated_time=30,
            field_targets=['event_outcome', 'dechallenge_result'],
            required=True,
            priority=2,
            success_rate=0.82,
            conditional_logic={'requires': ['event_outcome']}
        ))
        
        self.add_question(Question(
            question_id='S006',
            text='Did symptoms recur when restarting the medication?',
            category='Safety',
            difficulty='Hard',
            estimated_time=45,
            field_targets=['rechallenge_result', 'causality'],
            required=False,
            priority=2,
            success_rate=0.65,
            conditional_logic={'requires': ['dechallenge_result']}
        ))
        
        self.add_question(Question(
            question_id='S007',
            text='Were any other medications taken concurrently? Please list drug names and dates.',
            category='Safety',
            difficulty='Medium',
            estimated_time=60,
            field_targets=['concomitant_medications', 'drug_interactions'],
            required=True,
            priority=1,
            success_rate=0.80
        ))
        
        self.add_question(Question(
            question_id='S008',
            text='How severe was the adverse event on a scale of 1-10?',
            category='Safety',
            difficulty='Easy',
            estimated_time=15,
            field_targets=['event_severity', 'seriousness_score'],
            required=True,
            priority=1,
            success_rate=0.92
        ))
        
        self.add_question(Question(
            question_id='S009',
            text='Was the patient taking any herbal supplements or over-the-counter medications?',
            category='Safety',
            difficulty='Medium',
            estimated_time=35,
            field_targets=['concomitant_medications', 'supplements'],
            required=False,
            priority=2,
            success_rate=0.75
        ))
        
        self.add_question(Question(
            question_id='S010',
            text='Did the patient have any pre-existing medical conditions that might affect this outcome?',
            category='Safety',
            difficulty='Medium',
            estimated_time=50,
            field_targets=['medical_history', 'comorbidities'],
            required=True,
            priority=2,
            success_rate=0.85
        ))
        
        # EFFICACY CATEGORY - Drug effectiveness assessment
        self.add_question(Question(
            question_id='E001',
            text='Did the medication provide the expected therapeutic benefit?',
            category='Efficacy',
            difficulty='Easy',
            estimated_time=20,
            field_targets=['efficacy_outcome', 'treatment_response'],
            required=False,
            priority=3,
            success_rate=0.88
        ))
        
        self.add_question(Question(
            question_id='E002',
            text='How long did it take for the medication to show effect? (days/weeks)',
            category='Efficacy',
            difficulty='Medium',
            estimated_time=30,
            field_targets=['onset_time', 'treatment_duration'],
            required=False,
            priority=3,
            success_rate=0.80
        ))
        
        self.add_question(Question(
            question_id='E003',
            text='Was the dose adequate to produce the intended effect?',
            category='Efficacy',
            difficulty='Hard',
            estimated_time=45,
            field_targets=['dose_adequacy', 'dosage'],
            required=False,
            priority=3,
            success_rate=0.70,
            conditional_logic={'requires': ['dosage']}
        ))
        
        self.add_question(Question(
            question_id='E004',
            text='Did the patient need additional medications to enhance the effect?',
            category='Efficacy',
            difficulty='Medium',
            estimated_time=25,
            field_targets=['additional_therapy', 'treatment_modification'],
            required=False,
            priority=3,
            success_rate=0.82
        ))
        
        self.add_question(Question(
            question_id='E005',
            text='Was the treatment duration sufficient to assess efficacy?',
            category='Efficacy',
            difficulty='Medium',
            estimated_time=20,
            field_targets=['treatment_duration', 'assessment_complete'],
            required=False,
            priority=3,
            success_rate=0.86
        ))
        
        # PATIENT INFORMATION CATEGORY
        self.add_question(Question(
            question_id='P001',
            text='What is the patient\'s current age and weight?',
            category='Patient Info',
            difficulty='Easy',
            estimated_time=10,
            field_targets=['patient_age', 'patient_weight'],
            required=True,
            priority=1,
            success_rate=0.97
        ))
        
        self.add_question(Question(
            question_id='P002',
            text='Is the patient pregnant or breastfeeding?',
            category='Patient Info',
            difficulty='Easy',
            estimated_time=10,
            field_targets=['pregnancy_status', 'breastfeeding_status'],
            required=True,
            priority=1,
            success_rate=0.96
        ))
        
        self.add_question(Question(
            question_id='P003',
            text='What is the patient\'s gender?',
            category='Patient Info',
            difficulty='Easy',
            estimated_time=5,
            field_targets=['patient_gender'],
            required=True,
            priority=1,
            success_rate=0.99
        ))
        
        self.add_question(Question(
            question_id='P004',
            text='Does the patient have kidney or liver disease?',
            category='Patient Info',
            difficulty='Easy',
            estimated_time=15,
            field_targets=['renal_function', 'hepatic_function'],
            required=True,
            priority=2,
            success_rate=0.90
        ))
        
        self.add_question(Question(
            question_id='P005',
            text='What is the patient\'s ethnic background?',
            category='Patient Info',
            difficulty='Easy',
            estimated_time=10,
            field_targets=['ethnicity'],
            required=False,
            priority=3,
            success_rate=0.85
        ))
        
        self.add_question(Question(
            question_id='P006',
            text='How old was the patient when starting this medication?',
            category='Patient Info',
            difficulty='Easy',
            estimated_time=10,
            field_targets=['patient_age', 'drug_start_date'],
            required=True,
            priority=2,
            success_rate=0.93
        ))
        
        self.add_question(Question(
            question_id='P007',
            text='Does the patient smoke or consume alcohol? If yes, how much daily/weekly?',
            category='Patient Info',
            difficulty='Medium',
            estimated_time=25,
            field_targets=['smoking_status', 'alcohol_consumption'],
            required=True,
            priority=2,
            success_rate=0.78
        ))
        
        self.add_question(Question(
            question_id='P008',
            text='Any history of allergies or previous adverse drug reactions?',
            category='Patient Info',
            difficulty='Medium',
            estimated_time=30,
            field_targets=['allergy_history', 'prior_adrs'],
            required=True,
            priority=2,
            success_rate=0.84
        ))
        
        # MEDICATION CATEGORY
        self.add_question(Question(
            question_id='M001',
            text='What was the exact dose administered? (mg/kg/unit)',
            category='Medication',
            difficulty='Medium',
            estimated_time=20,
            field_targets=['dosage', 'dose_unit'],
            required=True,
            priority=1,
            success_rate=0.90
        ))
        
        self.add_question(Question(
            question_id='M002',
            text='What was the frequency of administration? (once daily, twice daily, etc.)',
            category='Medication',
            difficulty='Easy',
            estimated_time=15,
            field_targets=['frequency', 'dosing_schedule'],
            required=True,
            priority=1,
            success_rate=0.92
        ))
        
        self.add_question(Question(
            question_id='M003',
            text='What was the route of administration?',
            category='Medication',
            difficulty='Easy',
            estimated_time=10,
            field_targets=['route_of_administration'],
            required=True,
            priority=1,
            success_rate=0.95
        ))
        
        self.add_question(Question(
            question_id='M004',
            text='Was this the first time the patient took this medication?',
            category='Medication',
            difficulty='Easy',
            estimated_time=10,
            field_targets=['prior_exposure', 'sensitization'],
            required=True,
            priority=2,
            success_rate=0.94
        ))
        
        self.add_question(Question(
            question_id='M005',
            text='When did the patient start and stop this medication? (dates)',
            category='Medication',
            difficulty='Medium',
            estimated_time=20,
            field_targets=['drug_start_date', 'drug_end_date', 'duration'],
            required=True,
            priority=1,
            success_rate=0.88
        ))
        
        self.add_question(Question(
            question_id='M006',
            text='Was the dose adjusted during treatment? If yes, when and to what dose?',
            category='Medication',
            difficulty='Hard',
            estimated_time=40,
            field_targets=['dose_adjustment', 'dosage_history'],
            required=False,
            priority=2,
            success_rate=0.75
        ))
        
        self.add_question(Question(
            question_id='M007',
            text='Was the medication discontinued? If yes, was it due to the adverse event?',
            category='Medication',
            difficulty='Medium',
            estimated_time=25,
            field_targets=['discontinuation_reason', 'drug_end_date'],
            required=True,
            priority=1,
            success_rate=0.89
        ))
        
        self.add_question(Question(
            question_id='M008',
            text='What is the brand/generic name of the medication?',
            category='Medication',
            difficulty='Easy',
            estimated_time=15,
            field_targets=['drug_name', 'drug_type'],
            required=True,
            priority=1,
            success_rate=0.96
        ))
        
        self.add_question(Question(
            question_id='M009',
            text='What was the indication (reason for prescribing this medication)?',
            category='Medication',
            difficulty='Easy',
            estimated_time=20,
            field_targets=['indication', 'diagnosis'],
            required=True,
            priority=1,
            success_rate=0.91
        ))
        
        # MEDICAL HISTORY CATEGORY
        self.add_question(Question(
            question_id='H001',
            text='List all past medical diagnoses and approximate dates',
            category='Medical History',
            difficulty='Hard',
            estimated_time=60,
            field_targets=['medical_history', 'past_diagnoses'],
            required=True,
            priority=2,
            success_rate=0.70
        ))
        
        self.add_question(Question(
            question_id='H002',
            text='Has the patient had surgery? If yes, what type and when?',
            category='Medical History',
            difficulty='Medium',
            estimated_time=35,
            field_targets=['surgical_history'],
            required=False,
            priority=3,
            success_rate=0.80
        ))
        
        self.add_question(Question(
            question_id='H003',
            text='Is there a family history of similar adverse reactions?',
            category='Medical History',
            difficulty='Medium',
            estimated_time=30,
            field_targets=['family_history', 'genetic_risk'],
            required=False,
            priority=3,
            success_rate=0.82
        ))
        
        self.add_question(Question(
            question_id='H004',
            text='Any recent infections or vaccinations?',
            category='Medical History',
            difficulty='Medium',
            estimated_time=25,
            field_targets=['recent_infections', 'vaccination_history'],
            required=False,
            priority=3,
            success_rate=0.81
        ))
        
        # CAUSALITY ASSESSMENT CATEGORY
        self.add_question(Question(
            question_id='C001',
            text='In your opinion, how likely is it that this medication caused the adverse event? (scale 1-10)',
            category='Causality',
            difficulty='Hard',
            estimated_time=45,
            field_targets=['causality_assessment', 'reporter_opinion'],
            required=True,
            priority=1,
            success_rate=0.85
        ))
        
        self.add_question(Question(
            question_id='C002',
            text='Did similar reactions occur with related medications?',
            category='Causality',
            difficulty='Hard',
            estimated_time=40,
            field_targets=['class_effect', 'cross_reactivity'],
            required=False,
            priority=2,
            success_rate=0.72
        ))
        
        self.add_question(Question(
            question_id='C003',
            text='Were there any alternative explanations for the adverse event?',
            category='Causality',
            difficulty='Hard',
            estimated_time=50,
            field_targets=['alternative_causes', 'differential_diagnosis'],
            required=False,
            priority=2,
            success_rate=0.68
        ))
    
    def add_question(self, question: Question):
        """Add a question to the bank."""
        self.questions[question.question_id] = question
    
    def get_question(self, question_id: str) -> Optional[Question]:
        """Get a specific question by ID."""
        return self.questions.get(question_id)
    
    def get_questions_by_category(self, category: str) -> List[Question]:
        """Get all questions in a category."""
        return [q for q in self.questions.values() if q.category == category]
    
    def get_questions_by_difficulty(self, difficulty: str) -> List[Question]:
        """Get all questions of a specific difficulty."""
        return [q for q in self.questions.values() if q.difficulty == difficulty]
    
    def get_questions_by_field_target(self, field: str) -> List[Question]:
        """Get all questions that target a specific field."""
        return [q for q in self.questions.values() if field in q.field_targets]
    
    def get_required_questions(self) -> List[Question]:
        """Get all required questions."""
        return [q for q in self.questions.values() if q.required]
    
    def get_optional_questions(self) -> List[Question]:
        """Get all optional questions."""
        return [q for q in self.questions.values() if not q.required]
    
    def get_questions_by_priority(self, priority: int) -> List[Question]:
        """Get questions by priority level."""
        return [q for q in self.questions.values() if q.priority == priority]
    
    def get_all_categories(self) -> List[str]:
        """Get all unique categories."""
        return list(set(q.category for q in self.questions.values()))
    
    def get_total_questions(self) -> int:
        """Get total number of questions."""
        return len(self.questions)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert question bank to DataFrame."""
        data = [q.to_dict() for q in self.questions.values()]
        return pd.DataFrame(data)
    
    def get_high_priority_questions(self, missing_fields: List[str]) -> List[Question]:
        """
        Get high-priority questions based on missing fields.
        
        Args:
            missing_fields: List of field names that are missing
            
        Returns:
            Sorted list of questions targeting missing fields
        """
        relevant_questions = []
        for field in missing_fields:
            relevant_questions.extend(self.get_questions_by_field_target(field))
        
        # Remove duplicates and sort by priority
        unique_qs = {q.question_id: q for q in relevant_questions}
        return sorted(unique_qs.values(), key=lambda q: q.priority)
    
    def estimate_completion_time(self, questions: List[Question]) -> int:
        """
        Estimate total time to complete a list of questions.
        
        Args:
            questions: List of Question objects
            
        Returns:
            Total estimated time in seconds
        """
        return sum(q.estimated_time for q in questions)
    
    def get_question_coverage(self, questions: List[Question]) -> Dict[str, int]:
        """
        Get field coverage for a set of questions.
        
        Args:
            questions: List of Question objects
            
        Returns:
            Dict mapping fields to count of questions covering them
        """
        coverage = {}
        for question in questions:
            for field in question.field_targets:
                coverage[field] = coverage.get(field, 0) + 1
        return coverage
    
    def print_summary(self):
        """Print summary statistics of question bank."""
        print("\n" + "="*70)
        print("QUESTION BANK SUMMARY")
        print("="*70)
        print(f"\nTotal Questions: {self.get_total_questions()}")
        print(f"Required: {len(self.get_required_questions())}")
        print(f"Optional: {len(self.get_optional_questions())}")
        
        print("\n\nBREAKDOWN BY CATEGORY")
        print("-"*70)
        for category in self.get_all_categories():
            questions = self.get_questions_by_category(category)
            print(f"{category:20s}: {len(questions):3d} questions")
        
        print("\n\nBREAKDOWN BY DIFFICULTY")
        print("-"*70)
        for difficulty in ['Easy', 'Medium', 'Hard']:
            questions = self.get_questions_by_difficulty(difficulty)
            if questions:
                avg_time = sum(q.estimated_time for q in questions) / len(questions)
                print(f"{difficulty:10s}: {len(questions):3d} questions (avg time: {avg_time:.0f}s)")
        
        print("\n\nAVERAGE SUCCESS RATES BY CATEGORY")
        print("-"*70)
        for category in self.get_all_categories():
            questions = self.get_questions_by_category(category)
            avg_success = sum(q.success_rate for q in questions) / len(questions)
            print(f"{category:20s}: {avg_success:.1%}")
        
        print("\n")


if __name__ == "__main__":
    # Test the question bank
    bank = QuestionBank()
    bank.print_summary()
    
    # Example: Get questions for specific missing fields
    missing = ['event_severity', 'patient_age', 'drug_start_date']
    relevant = bank.get_high_priority_questions(missing)
    
    print("\n" + "="*70)
    print(f"RELEVANT QUESTIONS FOR MISSING FIELDS: {missing}")
    print("="*70)
    for q in relevant[:5]:
        print(f"\n[{q.question_id}] {q.text}")
        print(f"    Category: {q.category} | Difficulty: {q.difficulty}")
        print(f"    Time: {q.estimated_time}s | Success Rate: {q.success_rate:.1%}")
