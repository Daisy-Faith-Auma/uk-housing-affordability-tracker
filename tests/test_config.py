import unittest
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from config.settings import (
    DATA_DIRS, API_CONFIG, ANALYSIS_CONFIG, 
    GEO_CONFIG, LOGGING_CONFIG, validate_config
)

class TestConfiguration(unittest.TestCase):
    
    def test_data_directories_exist(self):
        """Test that all data directories are created"""
        for name, path in DATA_DIRS.items():
            with self.subTest(directory=name):
                self.assertTrue(path.exists(), f"Directory {name} does not exist at {path}")
                self.assertTrue(path.is_dir(), f"Path {path} is not a directory")
    
    def test_api_config_structure(self):
        """Test that API configuration has required keys"""
        required_keys = ['ons_base_url', 'land_registry_base', 'request_timeout', 'user_agent']
        
        for key in required_keys:
            with self.subTest(key=key):
                self.assertIn(key, API_CONFIG, f"Missing API config key: {key}")
    
    def test_analysis_config_values(self):
        """Test that analysis configuration values are reasonable"""
        self.assertGreater(ANALYSIS_CONFIG['affordability_threshold'], 0)
        self.assertGreater(ANALYSIS_CONFIG['crisis_threshold'], 0)
        self.assertGreater(ANALYSIS_CONFIG['end_year'], ANALYSIS_CONFIG['start_year'])
        self.assertGreaterEqual(ANALYSIS_CONFIG['min_transactions'], 1)
    
    def test_geographic_config(self):
        """Test geographic configuration"""
        bounds = GEO_CONFIG['uk_bounds']
        self.assertGreater(bounds['north'], bounds['south'])
        self.assertGreater(bounds['east'], bounds['west'])
        self.assertEqual(GEO_CONFIG['crs'], 'EPSG:4326')
    
    def test_config_validation(self):
        """Test that configuration validation passes"""
        try:
            validate_config()
        except ValueError as e:
            self.fail(f"Configuration validation failed: {e}")
    
    def test_logging_config(self):
        """Test logging configuration"""
        self.assertIn('level', LOGGING_CONFIG)
        self.assertIn('format', LOGGING_CONFIG)
        self.assertIn('file', LOGGING_CONFIG)

if __name__ == '__main__':
    unittest.main()