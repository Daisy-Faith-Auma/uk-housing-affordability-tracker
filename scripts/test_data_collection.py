#!/usr/bin/env python3
"""
Test script for housing data collection and processing
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))  # Add this line for config access

from collectors.ons_collector import ONSDataCollector
from processors.housing_processor import HousingDataProcessor
from config.settings import DATA_DIRS, API_CONFIG

def main():
    print("🏠 Testing Housing Data Collection & Processing")
    print("=" * 50)
    
    # Initialize collectors and processors
    data_dir = project_root / 'data'
    collector = ONSDataCollector(data_dir, API_CONFIG)
    processor = HousingDataProcessor(data_dir)
    
    try:
        # 1. Test data collection
        print("\n1. Testing Data Collection...")
        affordability_data = collector.download_housing_affordability_data()
        earnings_data = collector.download_earnings_data()
        
        print(f"✅ Affordability data: {len(affordability_data)} records")
        print(f"✅ Earnings data: {len(earnings_data)} records")
        
        # 2. Test data processing
        print("\n2. Testing Data Processing...")
        processed_data = processor.calculate_comprehensive_metrics(affordability_data)
        print(f"✅ Processed data: {len(processed_data)} records with {len(processed_data.columns)} columns")
        
        # 3. Display sample results
        print("\n3. Sample Results:")
        print("-" * 30)
        latest_year = processed_data['year'].max()
        latest_sample = processed_data[processed_data['year'] == latest_year].head(5)
        
        display_columns = ['local_authority', 'price_to_earnings_ratio', 'affordability_index', 
                          'crisis_status', 'housing_burden']
        print(latest_sample[display_columns].to_string(index=False))
        
        # 4. Generate policy insights
        print("\n4. Policy Insights:")
        print("-" * 30)
        insights = processor.generate_policy_insights(processed_data)
        
        summary = insights['executive_summary']
        print(f"• Total areas analyzed: {summary['total_areas_analyzed']}")
        print(f"• Crisis areas: {summary['crisis_areas']} ({summary['crisis_percentage']}%)")
        print(f"• Median affordability ratio: {summary['median_affordability_ratio']}")
        print(f"• Areas above healthy threshold (7x): {summary['areas_above_healthy_threshold']}")
        
        print(f"\n📊 Top 3 Crisis Areas:")
        for i, area in enumerate(insights['crisis_areas'][:3], 1):
            print(f"{i}. {area['local_authority']} ({area['region']}): {area['affordability_ratio']}x ratio")
        
        # 5. Test data summary
        print("\n5. Data Summary:")
        print("-" * 30)
        summary_stats = collector.get_data_summary()
        for key, value in summary_stats.items():
            print(f"• {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n✅ All tests passed successfully!")
        print(f"📁 Processed data saved to: {project_root}/data/processed/")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)