"""
Geospatial Signal Detection - Population Data Generator
Generate synthetic adverse event cases with geographic and batch information

Step 1: Generate population-level surveillance data
"""

import pandas as pd
import numpy as np
from typing import Tuple
from datetime import datetime, timedelta
import random


class PopulationDataGenerator:
    """Generate synthetic adverse event data with geographic and batch clustering."""
    
    def __init__(self, seed=42):
        """Initialize data generator."""
        random.seed(seed)
        np.random.seed(seed)
        
        # Geographic regions (lat/long boundaries)
        self.regions = {
            'North_America': {'lat': (25, 50), 'lng': (-130, -60)},
            'Europe': {'lat': (35, 70), 'lng': (-10, 45)},
            'Asia': {'lat': (10, 50), 'lng': (60, 150)},
            'South_America': {'lat': (-55, 15), 'lng': (-85, -30)},
            'Africa': {'lat': (-35, 35), 'lng': (-20, 55)}
        }
        
        # Drug information
        self.drugs = [
            'Aspirin', 'Ibuprofen', 'Amoxicillin', 'Metformin', 'Lisinopril',
            'Atorvastatin', 'Omeprazole', 'Sertraline', 'Albuterol', 'Levothyroxine'
        ]
        
        # Manufacturing sites
        self.manufacturing_sites = [
            'Site_A_Germany', 'Site_B_India', 'Site_C_USA', 'Site_D_China', 'Site_E_Brazil'
        ]
        
        # Adverse event types
        self.event_types = [
            'Rash', 'Nausea', 'Headache', 'Fatigue', 'Dizziness',
            'Allergic_Reaction', 'Stomach_Pain', 'Fever', 'Joint_Pain', 'Insomnia'
        ]
        
        self.severities = ['Mild', 'Moderate', 'Severe', 'Life-threatening']
    
    def generate_batch_info(self) -> dict:
        """Generate batch/lot information."""
        site = random.choice(self.manufacturing_sites)
        batch_num = random.randint(1, 999)
        lot_num = random.randint(1, 10000)
        
        return {
            'batch_id': f"BATCH_{site}_{batch_num:03d}",
            'lot_number': f"LOT_{lot_num:05d}",
            'manufacturing_site': site,
            'manufacturing_date': (datetime.now() - timedelta(days=random.randint(30, 180))).strftime('%Y-%m-%d'),
            'expiration_date': (datetime.now() + timedelta(days=random.randint(180, 730))).strftime('%Y-%m-%d')
        }
    
    def generate_geographic_location(self, region: str = None) -> dict:
        """Generate geographic coordinates."""
        if region is None:
            region = random.choice(list(self.regions.keys()))
        
        bounds = self.regions[region]
        lat = np.random.uniform(bounds['lat'][0], bounds['lat'][1])
        lng = np.random.uniform(bounds['lng'][0], bounds['lng'][1])
        
        return {
            'region': region,
            'latitude': lat,
            'longitude': lng,
            'country': self._get_country_from_coords(lat, lng)
        }
    
    def _get_country_from_coords(self, lat: float, lng: float) -> str:
        """Map coordinates to country."""
        countries_map = {
            'North_America': ['USA', 'Canada', 'Mexico'],
            'Europe': ['Germany', 'UK', 'France', 'Italy', 'Spain'],
            'Asia': ['India', 'China', 'Japan', 'Thailand'],
            'South_America': ['Brazil', 'Argentina', 'Chile'],
            'Africa': ['Egypt', 'Nigeria', 'South Africa']
        }
        
        for region, countries in countries_map.items():
            bounds = self.regions[region]
            if bounds['lat'][0] <= lat <= bounds['lat'][1] and bounds['lng'][0] <= lng <= bounds['lng'][1]:
                return random.choice(countries)
        
        return 'Unknown'
    
    def generate_adverse_event(self, cluster_info: dict = None) -> dict:
        """Generate single adverse event case."""
        batch_info = self.generate_batch_info()
        
        # Generate geographic location (with clustering if specified)
        if cluster_info and random.random() < 0.7:  # 70% chance to cluster
            geo = {
                'region': cluster_info['region'],
                'latitude': cluster_info['center_lat'] + np.random.normal(0, 0.5),
                'longitude': cluster_info['center_lng'] + np.random.normal(0, 0.5),
                'country': cluster_info['region']
            }
        else:
            geo = self.generate_geographic_location()
        
        # Event information
        drug = random.choice(self.drugs)
        event_type = random.choice(self.event_types)
        severity = random.choice(self.severities)
        
        # Reporter information
        reporter_type = random.choice(['HCP', 'Patient', 'Pharmacist'])
        
        return {
            'case_id': f"CASE_{int(datetime.now().timestamp() * 1000)}_{random.randint(0, 9999)}",
            'date_reported': (datetime.now() - timedelta(days=random.randint(0, 90))).strftime('%Y-%m-%d'),
            'drug_name': drug,
            'batch_id': batch_info['batch_id'],
            'lot_number': batch_info['lot_number'],
            'manufacturing_site': batch_info['manufacturing_site'],
            'manufacturing_date': batch_info['manufacturing_date'],
            'event_type': event_type,
            'severity': severity,
            'severity_numeric': {'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Life-threatening': 4}[severity],
            'latitude': geo['latitude'],
            'longitude': geo['longitude'],
            'region': geo['region'],
            'country': geo['country'],
            'reporter_type': reporter_type,
            'days_since_exposure': random.randint(0, 90),
            'quality_score': np.random.uniform(40, 100),
            'completeness_score': np.random.uniform(30, 100)
        }
    
    def generate_dataset(self, num_cases: int = 5000, anomalous_batches: int = 5) -> pd.DataFrame:
        """Generate population-level dataset with natural clusters."""
        cases = []
        
        print(f"\nGenerating {num_cases} adverse event cases...")
        print(f"Creating {anomalous_batches} anomalous batch clusters...\n")
        
        # Define anomalous cluster centers
        anomalous_clusters = []
        for i in range(anomalous_batches):
            region = random.choice(list(self.regions.keys()))
            bounds = self.regions[region]
            cluster = {
                'id': i,
                'region': region,
                'center_lat': np.random.uniform(bounds['lat'][0], bounds['lat'][1]),
                'center_lng': np.random.uniform(bounds['lng'][0], bounds['lng'][1]),
                'drug': random.choice(self.drugs),
                'size': random.randint(15, 40)
            }
            anomalous_clusters.append(cluster)
            print(f"  Cluster {i+1}: {cluster['drug']} in {cluster['region']} ({cluster['size']} cases)")
        
        # Generate cases
        cluster_case_counts = {i: 0 for i in range(anomalous_batches)}
        
        for i in range(num_cases):
            # Determine if this case is part of anomalous cluster
            cluster_id = None
            for cluster in anomalous_clusters:
                if cluster_case_counts[cluster['id']] < cluster['size']:
                    if random.random() < 0.6:  # 60% of cases go to clusters
                        cluster_id = cluster['id']
                        cluster_case_counts[cluster['id']] += 1
                        case = self.generate_adverse_event(cluster)
                        break
            
            # If not assigned to cluster, generate random case
            if cluster_id is None:
                case = self.generate_adverse_event()
            
            cases.append(case)
        
        df = pd.DataFrame(cases)
        
        print(f"\n✓ Generated {len(df)} cases")
        print(f"  Regions: {df['region'].nunique()}")
        print(f"  Drugs: {df['drug_name'].nunique()}")
        print(f"  Batches: {df['batch_id'].nunique()}")
        print(f"  Event types: {df['event_type'].nunique()}")
        
        return df
    
    def generate_train_test(self, num_cases: int = 5000, anomalous_batches: int = 5) -> Tuple[pd.DataFrame, list]:
        """Generate dataset and return with cluster information."""
        df = self.generate_dataset(num_cases, anomalous_batches)
        return df


if __name__ == "__main__":
    gen = PopulationDataGenerator()
    df = gen.generate_train_test(num_cases=5000, anomalous_batches=5)
    
    print("\n\nDataset sample:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
