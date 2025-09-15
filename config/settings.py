import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# Logs directory
LOGS_DIR = BASE_DIR / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

# API Configuration
API_CONFIG = {
    'ons_base_url': 'https://api.beta.ons.gov.uk/v1/datasets/',
    'land_registry_base': 'http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com',
    'request_timeout': 60,
    'rate_limit_delay': 1,  # seconds between requests
    'user_agent': 'UK-Housing-Tracker/1.0 (Educational Project)',
    'max_retries': 3,
    'backoff_factor': 2
}

# Analysis Parameters
ANALYSIS_CONFIG = {
    'affordability_threshold': 7.0,  # 7x earnings = traditional lending limit
    'crisis_threshold': 8.0,         # 8x earnings = crisis level
    'start_year': 2002,              # First year of comprehensive data
    'end_year': 2024,                # Current year
    'min_transactions': 10,          # Minimum transactions for reliable data
    'outlier_threshold': 3,          # Standard deviations for outlier detection
    'confidence_level': 0.95         # Statistical confidence level
}

# Geographic Configuration
GEO_CONFIG = {
    'crs': 'EPSG:4326',  # WGS84 for web mapping
    'uk_bounds': {
        'north': 60.9,
        'south': 49.8,
        'east': 2.0,
        'west': -8.6
    },
    'london_center': {
        'lat': 51.5074,
        'lon': -0.1278
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'housing_tracker.log',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    'page_title': 'UK Housing Affordability Tracker',
    'page_icon': '🏠',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Environment-specific settings
if os.getenv('ENVIRONMENT') == 'production':
    API_CONFIG['rate_limit_delay'] = 2  # Be more conservative in production
    LOGGING_CONFIG['level'] = 'WARNING'
elif os.getenv('ENVIRONMENT') == 'development':
    ANALYSIS_CONFIG['min_transactions'] = 5  # Lower threshold for testing
    LOGGING_CONFIG['level'] = 'DEBUG'

# Validation
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required directories exist
    for name, path in DATA_DIRS.items():
        if not path.exists():
            errors.append(f"Missing directory: {path}")
    
    # Check year range makes sense
    if ANALYSIS_CONFIG['start_year'] >= ANALYSIS_CONFIG['end_year']:
        errors.append("Start year must be before end year")
    
    # Check thresholds are positive
    if ANALYSIS_CONFIG['affordability_threshold'] <= 0:
        errors.append("Affordability threshold must be positive")
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")
    
    return True

# Run validation on import
validate_config()