import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from scipy import stats

class HousingDataProcessor:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.processed_dir = data_dir / 'processed'
        self.processed_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_comprehensive_metrics(self, affordability_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive affordability metrics and analysis"""
        
        self.logger.info("Calculating comprehensive affordability metrics...")
        
        # Create working copy
        df = affordability_data.copy()
        
        # 1. Basic affordability metrics
        df = self._add_basic_metrics(df)
        
        # 2. Time series analysis
        df = self._add_time_series_analysis(df)
        
        # 3. Regional comparisons
        df = self._add_regional_analysis(df)
        
        # 4. Crisis area identification
        df = self._identify_crisis_areas(df)
        
        # 5. Housing burden calculations
        df = self._calculate_housing_burden(df)
        
        # Save processed data
        output_path = self.processed_dir / 'comprehensive_affordability_metrics.csv'
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Processed data saved: {output_path}")
        self.logger.info(f"Total records processed: {len(df)}")
        
        return df
    
    def _add_basic_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic affordability metrics"""
        
        # Affordability index (0-100, higher = more affordable)
        df['affordability_index'] = np.clip(
            100 - ((df['price_to_earnings_ratio'] - 3) * 10), 0, 100
        ).round(1)
        
        # First-time buyer accessibility (4.5x income lending limit)
        df['ftb_accessible'] = df['price_to_earnings_ratio'] <= 4.5
        df['ftb_accessibility_gap'] = np.maximum(0, df['price_to_earnings_ratio'] - 4.5)
        
        # Housing stress levels
        df['stress_level'] = pd.cut(
            df['price_to_earnings_ratio'],
            bins=[0, 3.5, 4.5, 7, 10, float('inf')],
            labels=['No Stress', 'Low Stress', 'Moderate Stress', 'High Stress', 'Severe Stress']
        )
        
        return df
    
    def _add_time_series_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time series analysis metrics"""
        
        # Sort by authority and year
        df = df.sort_values(['local_authority', 'year'])
        
        # Year-over-year changes
        df['price_change_yoy'] = df.groupby('local_authority')['median_house_price'].pct_change() * 100
        df['earnings_change_yoy'] = df.groupby('local_authority')['median_earnings'].pct_change() * 100
        df['ratio_change_yoy'] = df.groupby('local_authority')['price_to_earnings_ratio'].pct_change() * 100
        
        # 3-year trends
        df['price_change_3yr'] = df.groupby('local_authority')['median_house_price'].pct_change(periods=3) * 100
        df['ratio_change_3yr'] = df.groupby('local_authority')['price_to_earnings_ratio'].pct_change(periods=3) * 100
        
        # Trend direction
        df['price_trend'] = df['price_change_yoy'].apply(self._classify_trend)
        df['affordability_trend'] = df['ratio_change_yoy'].apply(self._classify_trend)
        
        # Volatility (rolling standard deviation)
        df['price_volatility'] = df.groupby('local_authority')['price_change_yoy'].rolling(3).std().reset_index(0, drop=True)
        
        return df
    
    def _add_regional_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regional comparison metrics"""
        
        # Regional rankings
        df['regional_affordability_rank'] = df.groupby(['region', 'year'])['price_to_earnings_ratio'].rank(method='min')
        df['regional_price_rank'] = df.groupby(['region', 'year'])['median_house_price'].rank(method='min', ascending=False)
        
        # Regional comparisons
        regional_medians = df.groupby(['region', 'year'])['price_to_earnings_ratio'].median()
        df = df.set_index(['region', 'year'])
        df['regional_median_ratio'] = df.index.map(regional_medians)
        df = df.reset_index()
        
        df['vs_regional_median'] = df['price_to_earnings_ratio'] - df['regional_median_ratio']
        df['above_regional_median'] = df['vs_regional_median'] > 0
        
        return df
    
    def _identify_crisis_areas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify and classify crisis areas"""
        
        # Multiple crisis indicators
        df['high_ratio'] = df['price_to_earnings_ratio'] > 8.0
        df['rapid_growth'] = df['price_change_yoy'] > 15.0
        df['earnings_lag'] = (df['price_change_yoy'] - df['earnings_change_yoy']) > 10.0
        
        # Crisis score (0-3 based on indicators)
        df['crisis_score'] = (
            df['high_ratio'].astype(int) + 
            df['rapid_growth'].astype(int) + 
            df['earnings_lag'].astype(int)
        )
        
        # Crisis classification
        crisis_map = {0: 'Stable', 1: 'Watch', 2: 'Concern', 3: 'Crisis'}
        df['crisis_status'] = df['crisis_score'].map(crisis_map)
        
        # Overall crisis area flag
        df['is_crisis_area'] = df['crisis_score'] >= 2
        
        return df
    
    def _calculate_housing_burden(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate housing cost burden metrics"""
        
        # Monthly mortgage calculation (simplified)
        # Assume 25-year mortgage at 5% interest
        monthly_rate = 0.05 / 12
        num_payments = 25 * 12
        
        # Mortgage payment formula
        df['monthly_mortgage'] = df['median_house_price'] * (
            monthly_rate * (1 + monthly_rate) ** num_payments
        ) / ((1 + monthly_rate) ** num_payments - 1)
        
        df['monthly_income'] = df['median_earnings'] / 12
        df['housing_cost_ratio'] = df['monthly_mortgage'] / df['monthly_income']
        
        # Housing burden categories
        df['housing_burden'] = pd.cut(
            df['housing_cost_ratio'],
            bins=[0, 0.28, 0.35, 0.50, float('inf')],
            labels=['Affordable', 'Moderate Burden', 'High Burden', 'Severe Burden']
        )
        
        # Deposit requirements (10% deposit)
        df['required_deposit'] = df['median_house_price'] * 0.10
        df['deposit_years_saving'] = df['required_deposit'] / (df['median_earnings'] * 0.20)  # 20% savings rate
        
        return df
    
    def _classify_trend(self, change: float) -> str:
        """Classify trend based on percentage change"""
        if pd.isna(change):
            return 'Unknown'
        elif change > 10:
            return 'Rising Fast'
        elif change > 5:
            return 'Rising'
        elif change > -5:
            return 'Stable'
        elif change > -10:
            return 'Declining'
        else:
            return 'Declining Fast'
    
    def generate_policy_insights(self, processed_data: pd.DataFrame) -> Dict:
        """Generate policy-relevant insights from the analysis"""
        
        latest_year = processed_data['year'].max()
        latest_data = processed_data[processed_data['year'] == latest_year]
        
        insights = {
            'executive_summary': self._generate_executive_summary(latest_data),
            'crisis_areas': self._identify_priority_areas(latest_data),
            'regional_patterns': self._analyze_regional_patterns(processed_data),
            'trend_analysis': self._analyze_key_trends(processed_data),
            'policy_recommendations': self._generate_policy_recommendations(latest_data)
        }
        
        return insights
    
    def _generate_executive_summary(self, latest_data: pd.DataFrame) -> Dict:
        """Generate executive summary statistics"""
        
        total_areas = len(latest_data)
        crisis_areas = len(latest_data[latest_data['is_crisis_area']])
        median_ratio = latest_data['price_to_earnings_ratio'].median()
        
        return {
            'total_areas_analyzed': total_areas,
            'crisis_areas': crisis_areas,
            'crisis_percentage': round((crisis_areas / total_areas) * 100, 1),
            'median_affordability_ratio': round(median_ratio, 2),
            'areas_above_healthy_threshold': len(latest_data[latest_data['price_to_earnings_ratio'] > 7]),
            'first_time_buyer_accessible_areas': len(latest_data[latest_data['ftb_accessible']]),
            'median_house_price': int(latest_data['median_house_price'].median()),
            'median_earnings': int(latest_data['median_earnings'].median())
        }
    
    def _identify_priority_areas(self, latest_data: pd.DataFrame) -> List[Dict]:
        """Identify areas requiring immediate policy intervention"""
        
        priority_areas = latest_data[
            (latest_data['crisis_score'] >= 2) & 
            (latest_data['price_to_earnings_ratio'] > 8)
        ].sort_values('price_to_earnings_ratio', ascending=False)
        
        return [
            {
                'local_authority': row['local_authority'],
                'region': row['region'],
                'affordability_ratio': row['price_to_earnings_ratio'],
                'crisis_score': row['crisis_score'],
                'median_house_price': int(row['median_house_price']),
                'housing_burden': row['housing_burden']
            }
            for _, row in priority_areas.head(10).iterrows()
        ]
    
    def _analyze_regional_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze regional affordability patterns"""
        
        latest_year = data['year'].max()
        regional_data = data[data['year'] == latest_year].groupby('region').agg({
            'price_to_earnings_ratio': ['median', 'mean', 'std'],
            'is_crisis_area': 'sum',
            'local_authority': 'count'
        }).round(2)
        
        regional_summary = {}
        for region in regional_data.index:
            regional_summary[region] = {
                'median_ratio': regional_data.loc[region, ('price_to_earnings_ratio', 'median')],
                'mean_ratio': regional_data.loc[region, ('price_to_earnings_ratio', 'mean')],
                'crisis_areas': regional_data.loc[region, ('is_crisis_area', 'sum')],
                'total_areas': regional_data.loc[region, ('local_authority', 'count')],
                'crisis_percentage': round(
                    (regional_data.loc[region, ('is_crisis_area', 'sum')] / 
                     regional_data.loc[region, ('local_authority', 'count')]) * 100, 1
                )
            }
        
        return regional_summary
    
    def _analyze_key_trends(self, data: pd.DataFrame) -> Dict:
        """Analyze key market trends"""
        
        # National trends over time
        national_trends = data.groupby('year').agg({
            'price_to_earnings_ratio': 'median',
            'median_house_price': 'median',
            'median_earnings': 'median',
            'is_crisis_area': 'mean'
        }).round(2)
        
        # Calculate overall trends
        latest_ratio = national_trends['price_to_earnings_ratio'].iloc[-1]
        five_year_ago_ratio = national_trends['price_to_earnings_ratio'].iloc[0] if len(national_trends) >= 5 else latest_ratio
        
        return {
            'current_national_median_ratio': latest_ratio,
            'five_year_ratio_change': round(latest_ratio - five_year_ago_ratio, 2),
            'percentage_areas_in_crisis': round(national_trends['is_crisis_area'].iloc[-1] * 100, 1),
            'trend_direction': 'Worsening' if latest_ratio > five_year_ago_ratio else 'Improving',
            'annual_trends': national_trends.to_dict('index')
        }
    
    def _generate_policy_recommendations(self, latest_data: pd.DataFrame) -> List[str]:
        """Generate evidence-based policy recommendations"""
        
        recommendations = []
        
        crisis_percentage = (len(latest_data[latest_data['is_crisis_area']]) / len(latest_data)) * 100
        median_ratio = latest_data['price_to_earnings_ratio'].median()
        
        if crisis_percentage > 30:
            recommendations.append("Implement national housing emergency measures")
        
        if median_ratio > 8:
            recommendations.extend([
                "Expand Help to Buy scheme eligibility",
                "Consider shared ownership programs in high-ratio areas"
            ])
        
        london_areas = latest_data[latest_data['region'] == 'London']
        if len(london_areas) > 0 and london_areas['price_to_earnings_ratio'].median() > 10:
            recommendations.append("Accelerate London housing supply initiatives")
            recommendations.append("Review London housing allowance rates")
        
        high_growth_areas = len(latest_data[latest_data['price_change_yoy'] > 15])
        if high_growth_areas > 5:
            recommendations.append("Monitor and potentially cool overheated markets")
        
        recommendations.append("Target social housing development in crisis areas")
        recommendations.append("Review planning permissions in high-demand areas")
        
        return recommendations
    def create_enhanced_sample_data(self) -> pd.DataFrame:
        """Create comprehensive sample data with regional classification"""
        import numpy as np
        
        # Extended local authorities with regions
        authorities_data = [
            # London
            ('Westminster', 'London', 14.2, 850000),
            ('Kensington and Chelsea', 'London', 13.8, 920000),
            ('Camden', 'London', 10.8, 680000),
            ('Islington', 'London', 10.2, 650000),
            ('Hackney', 'London', 9.1, 580000),
            ('Tower Hamlets', 'London', 8.9, 560000),
            
            # South East
            ('Brighton and Hove', 'South East', 8.9, 420000),
            ('Oxford', 'South East', 9.2, 480000),
            ('Cambridge', 'South East', 8.7, 520000),
            ('Reading', 'South East', 7.8, 390000),
            
            # North
            ('Manchester', 'North West', 5.8, 220000),
            ('Liverpool', 'North West', 4.9, 180000),
            ('Leeds', 'Yorkshire', 5.4, 210000),
            ('Sheffield', 'Yorkshire', 4.2, 160000),
            ('Newcastle', 'North East', 4.1, 155000),
            
            # Midlands
            ('Birmingham', 'West Midlands', 5.9, 210000),
            ('Nottingham', 'East Midlands', 5.2, 185000),
            ('Leicester', 'East Midlands', 5.1, 190000),
            ('Coventry', 'West Midlands', 5.7, 200000),
            
            # Wales & Southwest
            ('Cardiff', 'Wales', 6.2, 240000),
            ('Bristol', 'South West', 7.1, 350000),
            ('Bath', 'South West', 8.3, 480000),
        ]
        
        years = list(range(2018, 2025))
        
        data = []
        for authority, region, base_ratio, base_price in authorities_data:
            for year in years:
                # Add year-over-year variation
                year_factor = 1 + (year - 2020) * 0.05  # 5% annual growth
                price_variation = np.random.uniform(0.95, 1.05)
                ratio_variation = np.random.uniform(0.95, 1.05)
                
                current_price = base_price * year_factor * price_variation
                current_ratio = base_ratio * ratio_variation
                current_earnings = current_price / current_ratio
                
                data.append({
                    'local_authority': authority,
                    'region': region,
                    'year': year,
                    'price_to_earnings_ratio': round(current_ratio, 2),
                    'median_house_price': round(current_price, -3),
                    'median_earnings': round(current_earnings, -2),
                    'data_source': 'SAMPLE_ENHANCED'
                })
        
        df = pd.DataFrame(data)
        
        # Add calculated fields
        df['affordability_category'] = pd.cut(
            df['price_to_earnings_ratio'],
            bins=[0, 4, 6, 8, 10, float('inf')],
            labels=['Very Affordable', 'Affordable', 'Stretched', 'Unaffordable', 'Crisis']
        )
        
        df['is_crisis_area'] = df['price_to_earnings_ratio'] > 8.0
        
        return df

def calculate_regional_stats(self, data: pd.DataFrame) -> pd.DataFrame:
    """Calculate regional summary statistics"""
    
    latest_year = data['year'].max()
    latest_data = data[data['year'] == latest_year]
    
    regional_stats = latest_data.groupby('region').agg({
        'price_to_earnings_ratio': ['mean', 'median', 'std'],
        'median_house_price': ['mean', 'median'],
        'median_earnings': ['mean', 'median'],
        'is_crisis_area': 'sum',
        'local_authority': 'count'
    }).round(2)
    
    # Flatten column names
    regional_stats.columns = ['_'.join(col).strip() for col in regional_stats.columns]
    regional_stats = regional_stats.reset_index()
    
    # Calculate crisis percentage
    regional_stats['crisis_percentage'] = (
        regional_stats['is_crisis_area_sum'] / regional_stats['local_authority_count'] * 100
    ).round(1)
    
    return regional_stats

def create_time_series_data(self, data: pd.DataFrame) -> pd.DataFrame:
    """Create time series aggregations"""
    
    national_trends = data.groupby('year').agg({
        'price_to_earnings_ratio': 'median',
        'median_house_price': 'median',
        'median_earnings': 'median',
        'is_crisis_area': ['sum', 'count']
    }).round(2)
    
    # Flatten columns
    national_trends.columns = ['_'.join(col) if col[1] else col[0] for col in national_trends.columns]
    national_trends = national_trends.reset_index()
    
    # Calculate national crisis percentage
    national_trends['crisis_percentage'] = (
        national_trends['is_crisis_area_sum'] / national_trends['is_crisis_area_count'] * 100
    ).round(1)
    
    return national_trends

