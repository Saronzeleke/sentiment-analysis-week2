# main.py
"""
Main Execution Script for Ethiopian Bank Reviews Analysis
Run this file to execute the complete pipeline
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection import PlayStoreScraper
from data_preprocessing import ReviewPreprocessor
from sentiment_analysis import SentimentAnalyzer
import pandas as pd


def main():
    """Execute complete analysis pipeline"""
    print("=" * 70)
    print("ETHIOPIAN BANK MOBILE APP REVIEWS ANALYSIS")
    print("=" * 70)
    
    # Step 1: Data Collection
    print("\n🚀 STEP 1: Data Collection")
    print("-" * 40)
    scraper = PlayStoreScraper()
    raw_df = scraper.scrape_all_banks()
    
    if raw_df.empty:
        print("❌ Data collection failed. Exiting.")
        return
    
    # Step 2: Data Preprocessing
    print("\n🔧 STEP 2: Data Preprocessing")
    print("-" * 40)
    preprocessor = ReviewPreprocessor()
    processed_df = preprocessor.process()
    
    if processed_df is None:
        print("❌ Data preprocessing failed. Exiting.")
        return
    
    # Step 3: Sentiment & Thematic Analysis
    print("\n📊 STEP 3: Sentiment & Thematic Analysis")
    print("-" * 40)
    analyzer = SentimentAnalyzer()
    analysis_success = analyzer.analyze()
    
    if analysis_success:
        print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("📁 Check the 'data/processed_data' folder for results")
    else:
        print("\n⚠️  Analysis completed with some issues")


if __name__ == "__main__":
    main()