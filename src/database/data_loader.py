import os
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import pandas as pd
import logging
from typing import Dict, Any
from database.database_connection import DatabaseConnection, create_tables, insert_banks_data
from typing import List, Dict, Any
from .database_connection import DatabaseConnection
import os 
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ReviewDataLoader:
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection

    def load_csv_data(self, csv_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['bank_name', 'review_text', 'rating', 'review_date', 
                              'sentiment_label', 'sentiment_score']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            df['review_text'] = df['review_text'].fillna('').astype(str)
            df['sentiment_label'] = df['sentiment_label'].fillna('neutral').astype(str)
            df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce').fillna(0)
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
            df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce').dt.date
            
            logger.info(f"Loaded {len(df)} records from {csv_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            raise

    def ensure_banks_exist(self, bank_names: list) -> Dict[str, int]:
        """Ensure all banks in the list exist; create missing ones with app_name = NULL (if allowed)"""
        # Get existing banks
        query = "SELECT bank_id, bank_name FROM banks"
        existing = self.db.execute_query(query)
        name_to_id = {row['bank_name']: row['bank_id'] for row in existing}

        # Insert missing banks
        missing = set(bank_names) - set(name_to_id.keys())
        for bank_name in sorted(missing):
            try:
                # Insert with app_name = '' (empty string) to satisfy NOT NULL if needed
                # If your schema allows NULL, use: VALUES (%s, NULL)
                insert_query = "INSERT INTO banks (bank_name, app_name) VALUES (%s, %s) RETURNING bank_id"
                result = self.db.execute_query(insert_query, params=(bank_name, ""))
                bank_id = result[0]['bank_id']
                name_to_id[bank_name] = bank_id
                logger.info(f"Auto-created bank '{bank_name}' with ID {bank_id}")
            except Exception as e:
                logger.error(f"Failed to auto-create bank '{bank_name}': {repr(e)}")
        return name_to_id

    def insert_reviews(self, df: pd.DataFrame, batch_size: int = 100) -> int:
        bank_mapping = self.ensure_banks_exist(df['bank_name'].unique().tolist())
        records = []

        for _, row in df.iterrows():
            bank_id = bank_mapping.get(row['bank_name'])
            if bank_id is None:
                logger.warning(f"Skipping review: bank '{row['bank_name']}' has no ID")
                continue

            record = (
                bank_id,
                row['review_text'][:5000],
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
                    total_inserted += len(batch)
                    logger.info(f"Inserted batch {i//batch_size + 1}: {len(batch)} records")
            except Exception as e:
                logger.error(f"Failed to insert batch {i//batch_size + 1}: {repr(e)}")
                self.db.connection.rollback()

        logger.info(f"✅ Total reviews inserted: {total_inserted}")
        return total_inserted

    def verify_data_integrity(self) -> Dict[str, Any]:
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
                       ROUND(AVG(r.rating)::numeric, 2) as avg_rating,
                       ROUND(AVG(r.sentiment_score)::numeric, 3) as avg_sentiment
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
            'date_range': "SELECT MIN(review_date) as earliest, MAX(review_date) as latest FROM reviews"
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
        import subprocess
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dump_cmd = [
            'pg_dump',
            '-h', self.db.config['host'],
            '-U', self.db.config['user'],
            '-d', self.db.config['dbname'],
            '-f', output_path,
            '--schema-only'
        ]
        try:
            env = os.environ.copy()
            env['PGPASSWORD'] = self.db.config['password']
            subprocess.run(dump_cmd, env=env, check=True)
            logger.info(f"SQL dump generated: {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate SQL dump: {e}")
            raise

def main():
    CSV_PATH = r"C:\Users\admin\sentiment-analysis-week2\data\processed_data\reviews_with_sentiment.csv"
    SQL_DUMP_PATH = r"C:\Users\admin\sentiment-analysis-week2\data\sql_dump\bank_reviews.sql"

    db = DatabaseConnection()
    try:
        if not db.connect():
            raise ConnectionError("Failed to connect to database")

        # Ensure clean slate
        db.execute_query("DROP TABLE IF EXISTS reviews CASCADE", fetch=False)
        db.execute_query("DROP TABLE IF EXISTS banks CASCADE", fetch=False)

        create_tables(db)
        # Note: skip insert_banks_data() — we auto-create banks from CSV

        loader = ReviewDataLoader(db)
        df = loader.load_csv_data(CSV_PATH)

        if len(df) < 400:
            logger.warning(f"Only {len(df)} reviews available — below 400 minimum")

        inserted_count = loader.insert_reviews(df)

        stats = loader.verify_data_integrity()
        print("\n" + "="*50)
        print("✅ DATA INTEGRITY REPORT")
        print("="*50)
        print(f"Total Reviews: {stats['total_reviews'][0]['count']}")
        for row in stats['reviews_per_bank']:
            print(f"  🏦 {row['bank_name']}: {row['review_count']} reviews")
        print("\nSentiment Distribution:")
        for row in stats['sentiment_distribution']:
            print(f"  {row['sentiment_label']}: {row['count']} ({row['percentage']}%)")

        loader.generate_sql_dump(SQL_DUMP_PATH)
        print(f"\n✅ SQL dump saved to: {SQL_DUMP_PATH}")

    except Exception as e:
        logger.error(f"Main execution failed: {repr(e)}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    main()