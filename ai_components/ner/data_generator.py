"""
Medical Named Entity Recognition - Data Generator Module
Generates synthetic medical narratives with labeled entities for training

Step 1: Generate training data for NER model
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from datetime import datetime, timedelta
import random


class MedicalEntity:
    """Represents a medical entity with position and type."""
    
    def __init__(self, text: str, entity_type: str, start: int, end: int):
        self.text = text
        self.entity_type = entity_type
        self.start = start
        self.end = end
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'type': self.entity_type,
            'start': self.start,
            'end': self.end
        }


class MedicalNarrativeGenerator:
    """Generate synthetic medical narratives with labeled entities."""
    
    def __init__(self, seed=42):
        """Initialize narrative generator."""
        random.seed(seed)
        np.random.seed(seed)
        
        # Entity libraries
        self.drugs = [
            'Aspirin', 'Ibuprofen', 'Amoxicillin', 'Metformin', 'Lisinopril',
            'Atorvastatin', 'Omeprazole', 'Sertraline', 'Albuterol', 'Levothyroxine',
            'Warfarin', 'Metoprolol', 'Amlodipine', 'Fluoxetine', 'Simvastatin',
            'Clopidogrel', 'Insulin glargine', 'Furosemide', 'Enalapril', 'Tramadol',
            'Ciprofloxacin', 'Prednisone', 'Azathioprine', 'Gabapentin', 'Celecoxib'
        ]
        
        self.dosages = [
            '5 mg', '10 mg', '25 mg', '50 mg', '100 mg', '250 mg', '500 mg', '1000 mg',
            '1 gram', '2 grams', '5 grams', '0.5 mg', '2.5 mg', '75 mg', '150 mg',
            '200 mg', '400 mg', '600 mg', '800 mg', '1 mg/kg', '2 mg/kg'
        ]
        
        self.routes = [
            'orally', 'intravenously', 'intramuscularly', 'subcutaneously',
            'topically', 'rectally', 'intranasal', 'inhaled', 'transdermal',
            'by mouth', 'IV', 'IM', 'SC', 'PO'
        ]
        
        self.durations = [
            '1 day', '3 days', '1 week', '2 weeks', '1 month', '3 months', '6 months',
            '1 year', '2 years', 'daily', 'twice daily', 'three times daily',
            'once weekly', 'twice weekly'
        ]
        
        self.conditions = [
            'hypertension', 'diabetes mellitus', 'hyperlipidemia', 'heart failure',
            'atrial fibrillation', 'pneumonia', 'urinary tract infection',
            'gastric ulcer', 'depression', 'anxiety disorder', 'asthma', 'COPD',
            'rheumatoid arthritis', 'osteoarthritis', 'migraine', 'seizure disorder',
            'thyroid disorder', 'chronic kidney disease', 'sepsis', 'stroke'
        ]
        
        self.outcomes = [
            'recovered', 'recovering', 'not recovered', 'fatal', 'hospitalized',
            'life-threatening', 'disabled', 'congenital anomaly', 'spontaneously resolved',
            'resolved after treatment discontinuation', 'ongoing', 'improved',
            'worsened', 'stable condition'
        ]
        
        self.frequencies = [
            'once daily', 'twice daily', 'three times daily', 'four times daily',
            'once weekly', 'twice weekly', 'once monthly', 'as needed',
            'every 6 hours', 'every 8 hours', 'every 12 hours', 'every 24 hours',
            'continuously'
        ]
        
        self.severities = [
            'mild', 'moderate', 'severe', 'life-threatening', 'critical',
            'minimal', 'substantial', 'serious'
        ]
        
        self.narrative_templates = [
            "Patient reported {severity} {outcome} after taking {drug} {dosage} {route} for {condition}. Treatment duration was {duration} with frequency of {frequency}.",
            "A {severity} adverse event of {outcome} was observed in patient with {condition} treated with {drug} {dosage}. Route of administration: {route}. Duration: {duration}.",
            "{drug} {dosage} administered {route} {frequency} resulted in {outcome} in patient diagnosed with {condition}. Event severity: {severity}. Treatment period: {duration}.",
            "Patient with history of {condition} received {drug} {dosage} {route} for {duration}. Frequency of administration: {frequency}. Outcome: {outcome} ({severity}).",
            "Following administration of {drug} {dosage} {route} {frequency}, patient with {condition} experienced {outcome}. Severity rating: {severity}. Total treatment duration: {duration}.",
            "Case report: {severity} {outcome} in patient treated with {drug} {dosage} via {route} at {frequency} for management of {condition}. Duration of therapy: {duration}.",
            "Patient on {drug} {dosage} administered {route} {frequency} developed {outcome} related to underlying {condition}. Severity: {severity}. Medication duration: {duration}.",
        ]
    
    def generate_narrative(self) -> Tuple[str, List[MedicalEntity]]:
        """Generate a single medical narrative with entities."""
        template = random.choice(self.narrative_templates)
        
        # Select entities
        drug = random.choice(self.drugs)
        dosage = random.choice(self.dosages)
        route = random.choice(self.routes)
        duration = random.choice(self.durations)
        condition = random.choice(self.conditions)
        outcome = random.choice(self.outcomes)
        frequency = random.choice(self.frequencies)
        severity = random.choice(self.severities)
        
        # Fill template
        narrative = template.format(
            drug=drug, dosage=dosage, route=route, duration=duration,
            condition=condition, outcome=outcome, frequency=frequency,
            severity=severity
        )
        
        # Extract entities with positions
        entities = []
        
        # Find and mark entities
        for entity_type, value in [
            ('DRUG', drug),
            ('DOSAGE', dosage),
            ('ROUTE', route),
            ('DURATION', duration),
            ('CONDITION', condition),
            ('OUTCOME', outcome),
            ('FREQUENCY', frequency),
            ('SEVERITY', severity)
        ]:
            start_idx = narrative.find(value)
            if start_idx != -1:
                end_idx = start_idx + len(value)
                entities.append(MedicalEntity(value, entity_type, start_idx, end_idx))
        
        return narrative, entities
    
    def generate_dataset(self, num_samples: int = 5000) -> pd.DataFrame:
        """Generate dataset of narratives with entities."""
        data = {
            'case_id': [],
            'narrative': [],
            'entities': [],
            'entity_count': [],
            'drug_count': [],
            'route_count': [],
            'condition_count': [],
            'outcome_count': [],
            'narrative_length': [],
            'complexity': []
        }
        
        for i in range(num_samples):
            narrative, entities = self.generate_narrative()
            
            # Count entity types
            entity_types = [e.entity_type for e in entities]
            drug_count = entity_types.count('DRUG')
            route_count = entity_types.count('ROUTE')
            condition_count = entity_types.count('CONDITION')
            outcome_count = entity_types.count('OUTCOME')
            
            # Determine complexity
            if len(entities) <= 4:
                complexity = 'simple'
            elif len(entities) <= 6:
                complexity = 'moderate'
            else:
                complexity = 'complex'
            
            data['case_id'].append(f"NER_{i:05d}")
            data['narrative'].append(narrative)
            data['entities'].append([e.to_dict() for e in entities])
            data['entity_count'].append(len(entities))
            data['drug_count'].append(drug_count)
            data['route_count'].append(route_count)
            data['condition_count'].append(condition_count)
            data['outcome_count'].append(outcome_count)
            data['narrative_length'].append(len(narrative))
            data['complexity'].append(complexity)
        
        return pd.DataFrame(data)
    
    def generate_ner_format(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert to standard NER training format (CoNLL-style)."""
        ner_format = []
        
        for _, row in df.iterrows():
            narrative = row['narrative']
            entities = row['entities']
            
            # Tokenize narrative (simple word tokenization)
            words = narrative.split()
            
            # Create token-level labels
            tokens = []
            current_pos = 0
            
            for word in words:
                word_start = narrative.find(word, current_pos)
                word_end = word_start + len(word)
                
                label = 'O'  # Outside any entity
                
                # Check if word is part of any entity
                for entity in entities:
                    if word_start >= entity['start'] and word_end <= entity['end']:
                        label = f"B-{entity['type']}"  # Beginning
                        if word_start > entity['start']:
                            label = f"I-{entity['type']}"  # Inside
                        break
                
                tokens.append({'text': word, 'label': label})
                current_pos = word_end
            
            ner_format.append({
                'case_id': row['case_id'],
                'text': narrative,
                'tokens': tokens,
                'entities': entities,
                'num_entities': row['entity_count']
            })
        
        return ner_format


class NERDataGenerator:
    """Main NER data generator orchestrator."""
    
    def __init__(self, seed=42):
        self.generator = MedicalNarrativeGenerator(seed=seed)
    
    def generate_and_split(self, num_training: int = 4000, num_test: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate training and test datasets."""
        print(f"\nGenerating {num_training + num_test} medical narratives with entities...")
        
        # Generate all data
        df = self.generator.generate_dataset(num_training + num_test)
        
        # Split
        train_df = df.iloc[:num_training].reset_index(drop=True)
        test_df = df.iloc[num_training:].reset_index(drop=True)
        
        print(f"✓ Generated {len(train_df)} training narratives")
        print(f"✓ Generated {len(test_df)} test narratives")
        
        # Print statistics
        print(f"\nTraining Data Statistics:")
        print(f"  Average narrative length: {train_df['narrative_length'].mean():.0f} characters")
        print(f"  Average entities per narrative: {train_df['entity_count'].mean():.2f}")
        print(f"  Complexity distribution:")
        for complexity, count in train_df['complexity'].value_counts().items():
            print(f"    • {complexity}: {count} ({count/len(train_df)*100:.1f}%)")
        
        return train_df, test_df


if __name__ == "__main__":
    gen = NERDataGenerator()
    train_df, test_df = gen.generate_and_split(num_training=4000, num_test=1000)
    
    # Show samples
    print("\n\nSample Narratives:")
    print("-" * 80)
    for idx in range(3):
        print(f"\nNarrative {idx + 1}:")
        print(f"Text: {train_df.iloc[idx]['narrative']}")
        print(f"Entities: {train_df.iloc[idx]['entities']}")
