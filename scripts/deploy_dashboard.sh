#!/bin/bash

echo "Deploying Housing Affordability Dashboard..."

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run data processing to ensure cache is populated
python -c "
from src.processors.housing_processor import HousingDataProcessor
from pathlib import Path

data_dir = Path('data')
processor = HousingDataProcessor(data_dir)
data = processor.create_enhanced_sample_data()
print(f'Generated {len(data)} sample records')
"

# Start streamlit
echo "Starting Streamlit dashboard..."
streamlit run src/dashboard/enhanced_app.py --server.port 8501 --server.address 0.0.0.0