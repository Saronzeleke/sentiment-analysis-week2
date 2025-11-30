"""
Data Preprocessing Module for cleaning and preparing review data
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocess and clean review data"""
    
    def __init__(self):
        self.processed_data = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load raw data from CSV"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded data with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate reviews based on review_id"""
        initial_count = len(df)
        df = df.drop_duplicates(subset=['review_id'], keep='first')
        removed_count = initial_count - len(df)
        logger.info(f"Removed {removed_count} duplicate reviews")
        return df
    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        initial_count = len(df)
        
        # Remove rows with missing critical fields
        df = df.dropna(subset=['content', 'score', 'at'])
        
        # Fill other missing values
        df['review_created_version'] = df['review_created_version'].fillna('Unknown')
        df['reply_content'] = df['reply_content'].fillna('')
        df['replied_at'] = df['replied_at'].fillna('')
        
        removed_count = initial_count - len(df)
        logger.info(f"Removed {removed_count} rows with missing critical data")
        
        return df
    
    def normalize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize date formats to YYYY-MM-DD"""
        try:
            # Convert to datetime and format
            df['at'] = pd.to_datetime(df['at']).dt.strftime('%Y-%m-%d')
            logger.info("Date normalization completed")
        except Exception as e:
            logger.error(f"Error normalizing dates: {str(e)}")
        
        return df
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize review text"""
        if pd.isna(text):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Trim whitespace
        text = text.strip()
        
        return text
    
    def preprocess_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply text cleaning to all reviews"""
        df['content_cleaned'] = df['content'].apply(self.clean_text)
        logger.info("Text cleaning completed")
        return df
    
    def calculate_data_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate data quality metrics"""
        total_reviews = len(df)
        
        metrics = {
            'total_reviews': total_reviews,
            'missing_content': df['content'].isna().sum(),
            'missing_ratings': df['score'].isna().sum(),
            'missing_dates': df['at'].isna().sum(),
            'duplicate_reviews': df.duplicated(subset=['review_id']).sum(),
            'reviews_per_bank': df['bank'].value_counts().to_dict()
        }
        
        # Calculate percentage metrics
        metrics['missing_content_pct'] = (metrics['missing_content'] / total_reviews) * 100
        metrics['missing_ratings_pct'] = (metrics['missing_ratings'] / total_reviews) * 100
        metrics['missing_dates_pct'] = (metrics['missing_dates'] / total_reviews) * 100
        metrics['duplicate_reviews_pct'] = (metrics['duplicate_reviews'] / total_reviews) * 100
        
        logger.info("Data quality metrics calculated")
        return metrics
    
    def run_pipeline(self, input_file: str) -> Tuple[pd.DataFrame, Dict]:
        """Run complete preprocessing pipeline"""
        logger.info("Starting data preprocessing pipeline")
        
        # Load data
        df = self.load_data(input_file)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Handle missing data
        df = self.handle_missing_data(df)
        
        # Normalize dates
        df = self.normalize_dates(df)
        
        # Clean text
        df = self.preprocess_reviews(df)
        
        # Select and rename final columns
        final_columns = {
            'review_id': 'review_id',
            'content_cleaned': 'review',
            'score': 'rating',
            'at': 'date',
            'bank': 'bank',
            'source': 'source'
        }
        
        df_final = df[list(final_columns.keys())].rename(columns=final_columns)
        
        # Calculate quality metrics
        metrics = self.calculate_data_quality_metrics(df_final)
        
        logger.info("Preprocessing pipeline completed")
        return df_final, metrics

def save_processed_data(df: pd.DataFrame, filename: str = "processed_reviews.csv"):
    """Save processed data to CSV"""
    try:
        df.to_csv(f"data/processed/{filename}", index=False)
        logger.info(f"Processed data saved to data/processed/{filename}")
    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}")

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    processed_df, quality_metrics = preprocessor.run_pipeline("data/raw/raw_reviews.csv")
    
    # Log quality metrics
    logger.info(f"Data Quality Metrics: {quality_metrics}")
    
    # Save processed data
    save_processed_data(processed_df)