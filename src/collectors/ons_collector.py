import pandas as pd
import numpy as np
import logging
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class ONSDataCollector:
    def __init__(self, data_dir: Path, config: Dict):
        self.data_dir = data_dir
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config['user_agent']
        })
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.get('log_level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def download_housing_affordability_data(self) -> pd.DataFrame:
        """Download ONS housing affordability ratios (using sample data for development)"""
        self.logger.info("Creating sample housing affordability data for development...")
        
        # For Week 1-2, we use realistic sample data
        sample_data = self._create_sample_affordability_data()
        
        # Cache the data
        cache_path = self.data_dir / 'cache' / 'ons_affordability_sample.csv'
        cache_path.parent.mkdir(exist_ok=True)
        sample_data.to_csv(cache_path, index=False)
        
        self.logger.info(f"Sample affordability data created: {len(sample_data)} records")
        self.logger.info(f"Data saved to: {cache_path}")
        
        return sample_data
    
    def _create_sample_affordability_data(self) -> pd.DataFrame:
        """Create realistic sample data based on actual UK housing patterns"""
        
        # Real UK local authorities with known affordability patterns
        authorities_data = [
            # London boroughs (high ratios)
            ('Westminster', 'London', 11.5, 850000),
            ('Kensington and Chelsea', 'London', 13.2, 920000),
            ('Camden', 'London', 9.8, 720000),
            ('Hackney', 'London', 8.9, 580000),
            ('Tower Hamlets', 'London', 7.8, 520000),
            ('Southwark', 'London', 8.4, 650000),
            
            # Southeast (medium-high ratios)
            ('Brighton and Hove', 'South East', 8.7, 480000),
            ('Oxford', 'South East', 9.1, 520000),
            ('Cambridge', 'South East', 8.8, 510000),
            ('Bath and North East Somerset', 'South West', 8.2, 470000),
            
            # Major cities (medium ratios)
            ('Birmingham', 'West Midlands', 5.8, 220000),
            ('Manchester', 'North West', 5.2, 190000),
            ('Liverpool', 'North West', 4.1, 150000),
            ('Leeds', 'Yorkshire and Humber', 5.5, 210000),
            ('Sheffield', 'Yorkshire and Humber', 4.8, 180000),
            ('Newcastle upon Tyne', 'North East', 4.2, 160000),
            ('Bristol', 'South West', 7.1, 380000),
            ('Nottingham', 'East Midlands', 4.9, 175000),
            ('Leicester', 'East Midlands', 5.1, 185000),
            ('Coventry', 'West Midlands', 5.3, 195000),
            
            # Additional areas
            ('Cardiff', 'Wales', 5.7, 240000),
            ('Swansea', 'Wales', 4.6, 180000),
            ('Edinburgh', 'Scotland', 6.8, 320000),  # For comparison
            ('Glasgow', 'Scotland', 4.9, 170000),   # For comparison
        ]
        
        years = list(range(2018, 2025))
        data = []
        
        for authority_name, region, base_ratio, base_price in authorities_data:
            for year in years:
                # Add year-over-year variation (housing market trends)
                year_factor = 1 + (year - 2018) * 0.04  # 4% annual price growth
                covid_boost = 1.15 if year in [2021, 2022] else 1.0  # COVID price boost
                recent_cooling = 0.95 if year >= 2023 else 1.0  # Recent market cooling
                
                # Calculate price with trends
                current_price = base_price * year_factor * covid_boost * recent_cooling
                
                # Add some randomness but keep realistic
                price_variance = np.random.uniform(0.92, 1.08)
                current_price *= price_variance
                
                # Calculate earnings (more stable than prices)
                base_earnings = current_price / base_ratio
                earnings_growth = 1 + (year - 2018) * 0.025  # 2.5% annual earnings growth
                current_earnings = base_earnings * earnings_growth
                
                # Calculate final ratio
                ratio = current_price / current_earnings
                
                # Add small random variation
                ratio *= np.random.uniform(0.95, 1.05)
                
                data.append({
                    'local_authority': authority_name,
                    'region': region,
                    'year': year,
                    'price_to_earnings_ratio': round(ratio, 2),
                    'median_house_price': round(current_price, -3),  # Round to nearest 1000
                    'median_earnings': round(current_earnings, -2),   # Round to nearest 100
                    'data_source': 'SAMPLE_DATA',
                    'data_quality': 'HIGH',
                    'created_date': datetime.now().isoformat()
                })
        
        df = pd.DataFrame(data)
        
        # Add some derived metrics for testing
        df['affordability_category'] = pd.cut(
            df['price_to_earnings_ratio'],
            bins=[0, 4, 6, 8, 10, float('inf')],
            labels=['Very Affordable', 'Affordable', 'Stretched', 'Unaffordable', 'Crisis']
        )
        
        return df
    
    def download_earnings_data(self) -> pd.DataFrame:
        """Download ASHE earnings data (sample for development)"""
        self.logger.info("Creating sample earnings data...")
        
        sample_earnings = self._create_sample_earnings_data()
        
        # Cache the data
        cache_path = self.data_dir / 'cache' / 'ons_earnings_sample.csv'
        sample_earnings.to_csv(cache_path, index=False)
        
        return sample_earnings
    
    def _create_sample_earnings_data(self) -> pd.DataFrame:
        """Create sample earnings data with realistic patterns"""
        
        # Earnings data typically available for larger authorities
        earnings_areas = [
            ('Westminster', 65000),
            ('Camden', 58000), 
            ('Birmingham', 32000),
            ('Manchester', 35000),
            ('Liverpool', 31000),
            ('Leeds', 34000),
            ('Bristol', 38000),
            ('Brighton and Hove', 36000),
            ('Oxford', 42000),
            ('Cambridge', 45000)
        ]
        
        years = list(range(2018, 2025))
        data = []
        
        for authority, base_earnings in earnings_areas:
            for year in years:
                # Earnings grow more steadily than house prices
                growth_factor = 1 + (year - 2018) * 0.025  # 2.5% annual growth
                current_earnings = base_earnings * growth_factor
                
                # Add small variance
                current_earnings *= np.random.uniform(0.98, 1.02)
                
                data.append({
                    'local_authority': authority,
                    'year': year,
                    'median_earnings': round(current_earnings, -2),
                    'mean_earnings': round(current_earnings * 1.15, -2),  # Mean typically 15% higher
                    'data_source': 'ASHE_SAMPLE',
                    'employment_rate': round(np.random.uniform(0.72, 0.85), 3)
                })
        
        return pd.DataFrame(data)
    
    def get_data_summary(self) -> Dict:
        """Get summary of available data"""
        
        affordability_data = self.download_housing_affordability_data()
        earnings_data = self.download_earnings_data()
        
        return {
            'affordability_records': len(affordability_data),
            'earnings_records': len(earnings_data),
            'authorities_count': affordability_data['local_authority'].nunique(),
            'year_range': f"{affordability_data['year'].min()}-{affordability_data['year'].max()}",
            'regions': affordability_data['region'].unique().tolist(),
            'data_quality': 'SAMPLE_HIGH_QUALITY',
            'last_updated': datetime.now().isoformat()
        }