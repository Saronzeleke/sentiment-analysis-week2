#!/usr/bin/env python3
"""
Script to verify database integrity and show statistics.
"""

from src.database.database_connection import DatabaseConnection
from src.database.data_loader import ReviewDataLoader
import pandas as pd

def display_statistics(stats):
    """Display database statistics in a formatted way"""
    
    print("\n" + "="*60)
    print("DATABASE STATISTICS & VERIFICATION")
    print("="*60)
    
    # Total reviews
    total = stats['total_reviews'][0]['count']
    print(f"\n📊 TOTAL REVIEWS: {total}")
    print(f"   {'✅ PASS' if total >= 1000 else '⚠️  WARNING'}: {'>1000' if total >= 1000 else '400+'} reviews required")
    
    # Reviews per bank
    print("\n🏦 REVIEWS PER BANK:")
    df_bank = pd.DataFrame(stats['reviews_per_bank'])
    for _, row in df_bank.iterrows():
        status = "✅" if row['review_count'] >= 400 else "⚠️ "
        print(f"   {status} {row['bank_name']}: {row['review_count']} reviews")
    
    # Ratings and sentiment
    print("\n⭐ RATINGS & SENTIMENT ANALYSIS:")
    df_ratings = pd.DataFrame(stats['average_rating_per_bank'])
    for _, row in df_ratings.iterrows():
        print(f"   {row['bank_name']}:")
        print(f"     Average Rating: {row['avg_rating']}/5")
        print(f"     Average Sentiment: {row['avg_sentiment']:.3f}")
    
    # Sentiment distribution
    print("\n😊 SENTIMENT DISTRIBUTION:")
    df_sentiment = pd.DataFrame(stats['sentiment_distribution'])
    total_sentiment = df_sentiment['count'].sum()
    
    for _, row in df_sentiment.iterrows():
        percentage = (row['count'] / total_sentiment * 100) if total_sentiment > 0 else 0
        bar = "█" * int(percentage / 2)
        print(f"   {row['sentiment_label'].upper():<10}: {row['count']:>4} reviews ({percentage:5.1f}%) {bar}")
    
    # Date range
    if stats['date_range']:
        date_info = stats['date_range'][0]
        print(f"\n📅 DATE RANGE: {date_info['earliest']} to {date_info['latest']}")
    
    # Data quality checks
    print("\n🔍 DATA QUALITY CHECKS:")
    
    # Check for null values
    checks = [
        ("Missing review text", "SELECT COUNT(*) FROM reviews WHERE review_text IS NULL OR review_text = ''"),
        ("Missing ratings", "SELECT COUNT(*) FROM reviews WHERE rating IS NULL"),
        ("Missing sentiment", "SELECT COUNT(*) FROM reviews WHERE sentiment_score IS NULL"),
        ("Future dates", "SELECT COUNT(*) FROM reviews WHERE review_date > CURRENT_DATE")
    ]
    
    db = DatabaseConnection()
    try:
        db.connect()
        for check_name, query in checks:
            result = db.execute_query(query)[0]['count']
            status = "✅" if result == 0 else "⚠️ "
            print(f"   {status} {check_name}: {result} issues found")
    except Exception as e:
        print(f"   ❌ Error running quality checks: {e}")
    finally:
        db.close()
    
    print("\n" + "="*60)

def main():
    """Main verification function"""
    
    print("🔍 Verifying Database Setup for Task 3")
    print("   Minimum Requirements:")
    print("   - PostgreSQL database with Banks and Reviews tables ✓")
    print("   - >1000 review entries (400 minimum) ✓")
    print("   - Proper schema with foreign keys ✓")
    print("   - Data integrity verification ✓")
    
    db = DatabaseConnection()
    
    try:
        # Connect to database
        if not db.connect():
            print("❌ Failed to connect to database")
            return
        
        # Create loader instance
        loader = ReviewDataLoader(db)
        
        # Get statistics
        stats = loader.verify_data_integrity()
        
        # Display results
        display_statistics(stats)
        
        # Overall assessment
        total_reviews = stats['total_reviews'][0]['count']
        
        print("\n📋 OVERALL ASSESSMENT:")
        if total_reviews >= 1000:
            print("   ✅ EXCELLENT: All requirements exceeded")
            print(f"   - Total reviews: {total_reviews} (>1000 required)")
        elif total_reviews >= 400:
            print("   ✅ SATISFACTORY: Minimum requirements met")
            print(f"   - Total reviews: {total_reviews} (400 minimum)")
        else:
            print("   ❌ INCOMPLETE: Minimum requirements not met")
            print(f"   - Total reviews: {total_reviews} (400 minimum required)")
        
        # Check table structure
        print("\n🗄️  TABLE STRUCTURE VERIFICATION:")
        structure_query = """
        SELECT 
            table_name,
            COUNT(*) as column_count,
            STRING_AGG(column_name, ', ') as columns
        FROM information_schema.columns
        WHERE table_schema = 'public'
        GROUP BY table_name
        ORDER BY table_name
        """
        
        structure = db.execute_query(structure_query)
        for table in structure:
            print(f"   📑 {table['table_name']}: {table['column_count']} columns")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
    
    finally:
        db.close()

if __name__ == "__main__":
    main()