"""
Smart Follow-Up Questionnaire Generator - Questionnaire Builder Module
Assemble adaptive questionnaires with branching logic and context embedding

Step 4: Build actual questionnaire documents from selected questions
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime


class QuestionnaireBuilder:
    """Build adaptive questionnaires from selected questions."""
    
    def __init__(self):
        """Initialize questionnaire builder."""
        self.case_context_template = """
CASE CONTEXT
============
Case ID: {case_id}
Initial Report Date: {report_date}
Current Status: {status}
Quality Score: {quality_score}/100
Completeness: {completeness}/100
Priority: {priority}

Data Gaps Identified:
{missing_fields_text}

This questionnaire is designed to obtain critical missing information.
Your responses will help us better assess the safety and effectiveness of this report.
"""
    
    def build_questionnaire(self, case: Dict[str, Any], questions: List[Dict[str, Any]],
                          include_context: bool = True) -> Dict[str, Any]:
        """
        Build a complete questionnaire.
        
        Args:
            case: Case information
            questions: List of selected questions
            include_context: Whether to include case context
            
        Returns:
            Complete questionnaire dict
        """
        questionnaire = {
            'questionnaire_id': f"Q_{case['case_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'case_id': case['case_id'],
            'creation_date': datetime.now().isoformat(),
            'total_questions': len(questions),
        }
        
        if include_context:
            questionnaire['context'] = self._build_context(case)
        
        # Build question sections
        questionnaire['sections'] = self._organize_into_sections(questions)
        
        # Add branching logic
        questionnaire['branching_logic'] = self._generate_branching_logic(questions)
        
        # Calculate time estimate
        questionnaire['estimated_completion_time_seconds'] = self._estimate_time(questions)
        questionnaire['estimated_completion_time_minutes'] = questionnaire['estimated_completion_time_seconds'] / 60
        
        # Add instructions
        questionnaire['instructions'] = self._generate_instructions(len(questions))
        
        return questionnaire
    
    def _build_context(self, case: Dict[str, Any]) -> str:
        """Build case context text."""
        missing_fields = case.get('missing_fields', [])
        missing_text = '\n'.join([f"  - {field}" for field in missing_fields])
        
        context = self.case_context_template.format(
            case_id=case['case_id'],
            report_date=case.get('report_date', 'Unknown'),
            status=case.get('validation_status', 'N/A'),
            quality_score=case.get('quality_score', 0),
            completeness=case.get('completeness_score', 0),
            priority=case.get('review_priority', 'N/A'),
            missing_fields_text=missing_text if missing_text else "None identified"
        )
        
        return context
    
    def _organize_into_sections(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Organize questions into logical sections."""
        # Group by category from field targets
        category_map = {
            'Safety': 'Patient Safety Assessment',
            'Efficacy': 'Medication Effectiveness',
            'Patient Info': 'Patient Demographics & History',
            'Medication': 'Medication Details',
            'Causality': 'Causality Assessment',
        }
        
        sections = {}
        section_order = []
        
        for question in questions:
            # Infer category from question ID (S=Safety, E=Efficacy, P=Patient, M=Med, H=History, C=Causality)
            prefix = question['question_id'][0]
            category_map_reverse = {
                'S': 'Safety',
                'E': 'Efficacy',
                'P': 'Patient Info',
                'M': 'Medication',
                'H': 'Patient Info',
                'C': 'Causality',
            }
            category = category_map_reverse.get(prefix, 'Other')
            
            if category not in sections:
                sections[category] = {
                    'section_name': category_map.get(category, category),
                    'questions': [],
                }
                section_order.append(category)
            
            sections[category]['questions'].append({
                'question_id': question['question_id'],
                'text': question.get('text', 'Question text not available'),
                'required': question.get('required', False),
                'difficulty': question.get('difficulty', 'Medium'),
                'estimated_time': question.get('estimated_time', 30),
            })
        
        return [sections[cat] for cat in section_order if cat in sections]
    
    def _generate_branching_logic(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate conditional branching logic.
        
        Example: If S003 (hospitalization) is YES, skip E2-E4 (efficacy)
        """
        branching = {
            'enabled': True,
            'rules': [
                {
                    'condition': 'If S003 (hospitalization) = YES',
                    'then': 'Expand safety section and add dechallenge questions',
                    'skip': [],
                },
                {
                    'condition': 'If validation_status = ACCEPT',
                    'then': 'Reduce to essential questions only',
                    'skip': ['E001', 'E002', 'E003'],
                },
                {
                    'condition': 'If num_missing_fields <= 2',
                    'then': 'Use quick form (5-7 minutes)',
                    'skip': [],
                },
            ]
        }
        
        return branching
    
    def _estimate_time(self, questions: List[Dict[str, Any]]) -> int:
        """Estimate total completion time."""
        base_time = sum(q.get('estimated_time', 30) for q in questions)
        
        # Add overhead (intro, context reading, review)
        overhead = min(len(questions) * 5, 60)
        
        return int(base_time + overhead)
    
    def _generate_instructions(self, num_questions: int) -> str:
        """Generate user-friendly instructions."""
        instructions = f"""
INSTRUCTIONS
============
Thank you for taking the time to respond to this follow-up questionnaire.

1. OVERVIEW
   This questionnaire contains {num_questions} questions designed to gather
   critical missing information about the adverse event report.

2. HOW TO RESPOND
   - Answer all questions marked with * (required) if possible
   - Provide as much detail as you can
   - If unsure about an answer, please provide your best estimate
   - You can save and resume your response later

3. TIME ESTIMATE
   This questionnaire should take approximately 10-20 minutes to complete.

4. CONFIDENTIALITY
   All responses are confidential and will be used for pharmacovigilance
   purposes only, in compliance with regulatory requirements.

5. NEED HELP?
   If you have questions, please contact: [contact_email]

Please click "START QUESTIONNAIRE" to begin.
"""
        return instructions
    
    def export_questionnaire(self, questionnaire: Dict[str, Any], format: str = 'json') -> str:
        """
        Export questionnaire in specified format.
        
        Args:
            questionnaire: Built questionnaire dict
            format: 'json', 'html', or 'text'
            
        Returns:
            Formatted questionnaire string
        """
        if format == 'text':
            return self._export_text(questionnaire)
        elif format == 'html':
            return self._export_html(questionnaire)
        else:  # json
            import json
            return json.dumps(questionnaire, indent=2)
    
    def _export_text(self, questionnaire: Dict[str, Any]) -> str:
        """Export as plain text."""
        text = f"QUESTIONNAIRE: {questionnaire['questionnaire_id']}\n"
        text += f"Case ID: {questionnaire['case_id']}\n"
        text += f"Total Questions: {questionnaire['total_questions']}\n"
        text += f"Estimated Time: {questionnaire['estimated_completion_time_minutes']:.1f} minutes\n\n"
        
        if 'context' in questionnaire:
            text += questionnaire['context'] + "\n\n"
        
        text += questionnaire['instructions'] + "\n\n"
        
        q_num = 1
        for section in questionnaire['sections']:
            text += f"\n{section['section_name'].upper()}\n"
            text += "=" * 50 + "\n"
            for q in section['questions']:
                required = "*" if q['required'] else " "
                text += f"\n{q_num}{required}. {q['text']}\n"
                text += f"   (Question ID: {q['question_id']})\n"
                q_num += 1
        
        return text
    
    def _export_html(self, questionnaire: Dict[str, Any]) -> str:
        """Export as HTML."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Questionnaire {questionnaire['questionnaire_id']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
        .section {{ margin-top: 20px; border-left: 4px solid #007bff; padding-left: 15px; }}
        .question {{ margin-top: 15px; padding: 10px; background-color: #f9f9f9; }}
        .required {{ color: red; }}
        textarea {{ width: 100%; height: 80px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Follow-up Questionnaire</h1>
        <p>Case ID: {questionnaire['case_id']}</p>
        <p>Estimated Time: {questionnaire['estimated_completion_time_minutes']:.1f} minutes</p>
    </div>
"""
        
        q_num = 1
        for section in questionnaire['sections']:
            html += f'<div class="section"><h2>{section["section_name"]}</h2>'
            for q in section['questions']:
                required_tag = '<span class="required">*</span>' if q['required'] else ''
                html += f"""
    <div class="question">
        <p><strong>{q_num}{required_tag}. {q['text']}</strong></p>
        <textarea placeholder="Enter your response here..."></textarea>
    </div>
"""
                q_num += 1
            html += '</div>'
        
        html += """
    <button onclick="submitQuestionnaire()">Submit Questionnaire</button>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    # Test builder
    builder = QuestionnaireBuilder()
    
    test_case = {
        'case_id': '12345',
        'report_date': '2026-01-01',
        'validation_status': 'REVIEW',
        'quality_score': 55,
        'completeness_score': 60,
        'review_priority': 'High',
        'missing_fields': ['event_severity', 'hospitalization_flag', 'patient_age'],
    }
    
    test_questions = [
        {'question_id': 'S001', 'text': 'Did symptoms worsen?', 'required': True, 
         'difficulty': 'Easy', 'estimated_time': 20},
        {'question_id': 'S008', 'text': 'Severity on scale 1-10?', 'required': True,
         'difficulty': 'Easy', 'estimated_time': 15},
        {'question_id': 'P001', 'text': 'What is patient age and weight?', 'required': True,
         'difficulty': 'Easy', 'estimated_time': 10},
    ]
    
    questionnaire = builder.build_questionnaire(test_case, test_questions)
    
    print("QUESTIONNAIRE BUILT:")
    print("=" * 70)
    print(f"ID: {questionnaire['questionnaire_id']}")
    print(f"Questions: {questionnaire['total_questions']}")
    print(f"Time: {questionnaire['estimated_completion_time_minutes']:.1f} min")
    
    # Export as text
    text = builder.export_questionnaire(questionnaire, format='text')
    print("\n\n" + text)
