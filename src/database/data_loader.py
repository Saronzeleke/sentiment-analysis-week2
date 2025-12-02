import os 
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.resolve("src")))
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Any
from src.database.database_connection import DatabaseConnection, create_tables, insert_banks_data
logger = logging.getLogger(__name__)

class ReviewDataLoader:
    """Load processed review data into PostgreSQL database"""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
        
    def load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """Load and validate CSV data"""
        try:
            df = pd.read_csv(csv_path)
            
            # Data validation
            required_columns = ['bank_name', 'review_text', 'rating', 'review_date', 
                              'sentiment_label', 'sentiment_score']
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Clean data
            df['review_text'] = df['review_text'].fillna('').astype(str)
            df['sentiment_label'] = df['sentiment_label'].fillna('neutral').astype(str)
            df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce').fillna(0)
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
            
            # Parse dates
            df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce').dt.date
            
            logger.info(f"Loaded {len(df)} records from {csv_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            raise
    
    def get_bank_mapping(self) -> Dict[str, int]:
        """Get mapping of bank names to bank_ids"""
        try:
            query = "SELECT bank_id, bank_name FROM banks"
            results = self.db.execute_query(query)
            
            return {row['bank_name']: row['bank_id'] for row in results}
            
        except Exception as e:
            logger.error(f"Failed to get bank mapping: {e}")
            return {}
    
    def insert_reviews(self, df: pd.DataFrame, batch_size: int = 100) -> int:
        """Insert reviews in batches for better performance"""
        
        bank_mapping = self.get_bank_mapping()
        
        # Prepare data for insertion
        records = []
        for _, row in df.iterrows():
            bank_id = bank_mapping.get(row['bank_name'])
            
            if bank_id:  # Only insert if bank exists
                record = (
                    bank_id,
                    row['review_text'][:5000],  # Truncate if too long
                    float(row['rating']),
                    row['review_date'],
                    row['sentiment_label'],
                    float(row['sentiment_score']),
                    'Google Play Store',
                    row.get('cleaned_text', '')[:5000],
                    row.get('keywords', '').split(',')[:10] if isinstance(row.get('keywords'), str) else [],
                    row.get('theme', 'general')
                )
                records.append(record)
        
        # Insert in batches
        insert_query = """
        INSERT INTO reviews 
        (bank_id, review_text, rating, review_date, sentiment_label, 
         sentiment_score, source, cleaned_text, keywords, theme)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        total_inserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            try:
                with self.db.connection.cursor() as cursor:
                    cursor.executemany(insert_query, batch)
                    self.db.connection.commit()
                    
                    batch_inserted = len(batch)
                    total_inserted += batch_inserted
                    logger.info(f"Inserted batch {i//batch_size + 1}: {batch_inserted} records")
                    
            except Exception as e:
                logger.error(f"Failed to insert batch {i//batch_size + 1}: {e}")
                self.db.connection.rollback()
        
        logger.info(f"Total reviews inserted: {total_inserted}")
        return total_inserted
    
    def verify_data_integrity(self) -> Dict[str, Any]:
        """Run integrity checks and return statistics"""
        
        queries = {
            'total_reviews': "SELECT COUNT(*) as count FROM reviews",
            'reviews_per_bank': """
                SELECT b.bank_name, COUNT(r.review_id) as review_count
                FROM banks b
                LEFT JOIN reviews r ON b.bank_id = r.bank_id
                GROUP BY b.bank_name
                ORDER BY review_count DESC
            """,
            'average_rating_per_bank': """
                SELECT b.bank_name, 
                       ROUND(AVG(r.rating), 2) as avg_rating,
                       ROUND(AVG(r.sentiment_score), 3) as avg_sentiment
                FROM banks b
                LEFT JOIN reviews r ON b.bank_id = r.bank_id
                GROUP BY b.bank_name
                ORDER BY avg_rating DESC
            """,
            'sentiment_distribution': """
                SELECT sentiment_label, COUNT(*) as count,
                       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
                FROM reviews
                GROUP BY sentiment_label
                ORDER BY count DESC
            """,
            'date_range': """
                SELECT MIN(review_date) as earliest, MAX(review_date) as latest
                FROM reviews
            """
        }
        
        results = {}
        for name, query in queries.items():
            try:
                results[name] = self.db.execute_query(query)
            except Exception as e:
                logger.error(f"Failed to execute {name} query: {e}")
                results[name] = []
        
        return results
    
    def generate_sql_dump(self, output_path: str) -> None:
        """Generate SQL dump of the database"""
        import subprocess
        
        dump_cmd = [
            'pg_dump',
            '-h', self.db.config['host'],
            '-U', self.db.config['user'],
            '-d', self.db.config['dbname'],
            '-f', output_path,
            '--schema-only'  # Change to '--data-only' or remove for full dump
        ]
        
        try:
            # Set PGPASSWORD environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = self.db.config['password']
            
            subprocess.run(dump_cmd, env=env, check=True)
            logger.info(f"SQL dump generated: {output_path}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate SQL dump: {e}")
            raise

def main():
    """Main function to load data into database"""
    # Configuration
    CSV_PATH = "../data/processed/processed_reviews.csv"
    SQL_DUMP_PATH = "../data/sql_dump/bank_reviews.sql"
    
    # Initialize database connection
    db = DatabaseConnection()
    
    try:
        # Connect to database
        if not db.connect():
            raise ConnectionError("Failed to connect to database")
        
        # Create tables
        create_tables(db)
        
        # Insert bank data
        insert_banks_data(db)
        
        # Load and insert review data
        loader = ReviewDataLoader(db)
        df = loader.load_csv_data(CSV_PATH)
        
        # Check if we have enough data
        if len(df) < 400:
            logger.warning(f"Only {len(df)} reviews available, minimum 400 required")
        
        inserted_count = loader.insert_reviews(df)
        
        if inserted_count >= 1000:
            logger.info(f"✅ Successfully inserted {inserted_count} reviews (Exceeds 1000 requirement)")
        elif inserted_count >= 400:
            logger.info(f"✅ Successfully inserted {inserted_count} reviews (Meets minimum requirement)")
        else:
            logger.warning(f"Only inserted {inserted_count} reviews (Below minimum requirement)")
        
        # Verify data integrity
        stats = loader.verify_data_integrity()
        
        # Print verification results
        print("\n" + "="*50)
        print("DATA INTEGRITY VERIFICATION")
        print("="*50)
        
        print(f"\nTotal Reviews: {stats['total_reviews'][0]['count']}")
        
        print("\nReviews per Bank:")
        for row in stats['reviews_per_bank']:
            print(f"  {row['bank_name']}: {row['review_count']}")
        
        print("\nAverage Ratings:")
        for row in stats['average_rating_per_bank']:
            print(f"  {row['bank_name']}: Rating={row['avg_rating']}, Sentiment={row['avg_sentiment']}")
        
        print("\nSentiment Distribution:")
        for row in stats['sentiment_distribution']:
            print(f"  {row['sentiment_label']}: {row['count']} ({row['percentage']}%)")
        
        # Generate SQL dump
        loader.generate_sql_dump(SQL_DUMP_PATH)
        
        print("\n" + "="*50)
        print("TASK 3 COMPLETED SUCCESSFULLY")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise
        
    finally:
        db.close()

if __name__ == "__main__":
    main()