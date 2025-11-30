"""
Main Execution Script - Complete Project Pipeline
Runs both Task 1 (Data Collection & Preprocessing) and Task 2 (Analysis)
"""

import os
import sys
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run complete project pipeline"""
    print("=" * 70)
    print("ETHIOPIAN BANK REVIEWS ANALYSIS - COMPLETE PIPELINE")
    print("=" * 70)
    
    try:
        # Task 1: Data Collection & Preprocessing
        print("\n📊 TASK 1: DATA COLLECTION & PREPROCESSING")
        print("-" * 50)
        
        from data_collection import PlayStoreScraper
        from preprocessing import ReviewPreprocessor
        from config.banks_config import DATA_PATHS
        
        # Step 1: Data Collection
        print("1. Collecting data...")
        scraper = PlayStoreScraper()
        raw_df = scraper.scrape_all_banks()  # ✅ Fixed: was 'run_complete_collection'
        
        if raw_df.empty:
            print("⚠️ No data collected. Skipping preprocessing and analysis.")
            return None
        
        # Step 2: Data Preprocessing
        print("\n2. Preprocessing data...")
        preprocessor = ReviewPreprocessor()
        processed_df, quality_metrics = preprocessor.run_pipeline(
            DATA_PATHS['raw_reviews'],
            DATA_PATHS['processed_reviews']
        )
        
        # Task 2: Analysis (Partial)
        print("\n😊 TASK 2: SENTIMENT & THEMATIC ANALYSIS (Partial)")
        print("-" * 50)
        
        # Import analysis modules
        try:
            from sentiment_analysis import SentimentAnalyzer
            from thematic_analysis import ThematicAnalyzer
            
            print("1. Performing sentiment analysis...")
            sentiment_analyzer = SentimentAnalyzer()
            sentiment_results = sentiment_analyzer.analyze_dataframe(processed_df)
            
            print("2. Performing thematic analysis...")
            thematic_analyzer = ThematicAnalyzer()
            thematic_results = thematic_analyzer.run_analysis(sentiment_results)
            
            print("✅ Analysis completed successfully!")
            
        except ImportError:
            print("⚠️ Analysis modules not available - Task 2 ready for implementation")
        
        # Final Summary
        print("\n" + "=" * 70)
        print("🎉 PROJECT EXECUTION SUMMARY")
        print("=" * 70)
        
        print(f"✅ Task 1 Completed:")
        print(f"   - Data Collection: {len(raw_df)} reviews collected")
        print(f"   - Data Preprocessing: {len(processed_df)} clean reviews")
        print(f"   - Data Quality: {100 - quality_metrics.get('missing_content_pct', 0):.1f}%")
        
        print(f"📁 Files Created:")
        print(f"   - Raw Data: {DATA_PATHS['raw_reviews']}")
        print(f"   - Processed Data: {DATA_PATHS['processed_reviews']}")
        
        print(f"\n🚀 Next Steps:")
        print(f"   - Run notebooks/02_sentiment_analysis.ipynb for detailed analysis")
        print(f"   - Run notebooks/03_thematic_analysis.ipynb for theme exploration")
        print(f"   - Check data/results/ for output files")
        
        return processed_df
        
    except Exception as e:
        print(f"❌ Error in pipeline execution: {e}")
        raise


if __name__ == "__main__":
    result_df = main()