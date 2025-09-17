import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables if .env file exists
try:
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIRS = {
    'raw': BASE_DIR / 'data' / 'raw',
    'processed': BASE_DIR / 'data' / 'processed',
    'cache': BASE_DIR / 'data' / 'cache',
    'geographic': BASE_DIR / 'data' / 'geographic'
}

# Ensure directories exist
for dir_path in DATA_DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)

# API Configuration
API_CONFIG = {
    'ons_base_url': 'https://api.beta.ons.gov.uk/v1/datasets/',
    'land_registry_base': 'http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com',
    'request_timeout': 60,
    'rate_limit_delay': 1,  # seconds between requests
    'user_agent': 'UK-Housing-Tracker/1.0 (Educational Project)'
}

# Analysis Parameters
ANALYSIS_CONFIG = {
    'affordability_threshold': 7.0,  # 7x earnings = unaffordable
    'crisis_threshold': 8.0,         # 8x earnings = crisis
    'start_year': 2002,
    'end_year': 2024,
    'min_transactions': 10           # Minimum transactions for reliable data
}

# Geographic Configuration
GEO_CONFIG = {
    'crs': 'EPSG:4326',  # WGS84 for web mapping
    'uk_bounds': {
        'north': 60.9,
        'south': 49.8,
        'east': 2.0,
        'west': -8.6
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': BASE_DIR / 'logs' / 'housing_tracker.log'
}

# Ensure logs directory exists
(BASE_DIR / 'logs').mkdir(exist_ok=True)