"""
Unit tests for data collection module
"""

import pytest
import pandas as pd
from src.data_collection import ReviewScraper
from src.preprocessing import DataPreprocessor

class TestDataCollection:
    """Test data collection functionality"""
    
    def test_scraper_initialization(self):
        """Test scraper initialization"""
        scraper = ReviewScraper()
        assert scraper is not None
        assert hasattr(scraper, 'banks_config')
        assert len(scraper.banks_config) == 3
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
    
    def test_data_quality_validation(self, sample_data):
        """Test data quality validation"""
        from src.utils import validate_data_quality
        
        # Test with good data
        assert validate_data_quality(sample_data, min_reviews=5, max_missing_pct=10.0)
        
        # Test with insufficient data
        insufficient_data = sample_data.head(2)
        assert not validate_data_quality(insufficient_data, min_reviews=5, max_missing_pct=10.0)

@pytest.fixture
def sample_data():
    """Sample data for testing"""
    return pd.DataFrame({
        'review_id': [1, 2, 3, 4, 5],
        'review': ['Great app!', 'Needs improvement', 'Awesome', 'Bad experience', 'Good'],
        'rating': [5, 2, 5, 1, 4],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'bank': ['CBE', 'Awash', 'CBE', 'Dashen', 'Awash'],
        'source': ['Google Play'] * 5
    })