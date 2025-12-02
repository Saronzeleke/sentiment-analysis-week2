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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReviewDataLoader:
    """Load processed review data into PostgreSQL database"""

    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection

    def load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """Load and validate CSV data"""
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

    def get_bank_mapping(self) -> Dict[str, int]:
        """Get mapping of normalized bank names to bank_ids"""
        try:
            query = "SELECT bank_id, bank_name FROM banks"
            results = self.db.execute_query(query)
            return {row['bank_name'].strip().lower(): row['bank_id'] for row in results}
        except Exception as e:
            logger.error(f"Failed to get bank mapping: {e}")
            return {}

    def insert_or_create_bank(self, bank_name: str) -> int:
        """Insert a new bank if missing and return its bank_id"""
        try:
            # FIX: Include app_name to satisfy NOT NULL constraint
            insert_query = "INSERT INTO banks (bank_name, app_name) VALUES (%s, %s) RETURNING bank_id"
            with self.db.connection.cursor() as cursor:
                cursor.execute(insert_query, (bank_name, ""))  # empty string as placeholder
                bank_id = cursor.fetchone()[0]
                self.db.connection.commit()
                logger.info(f"Inserted new bank: {bank_name} with ID {bank_id}")
                return bank_id
        except Exception as e:
            logger.error(f"Failed to insert new bank '{bank_name}': {e}")
            self.db.connection.rollback()
            return None

    def insert_reviews(self, df: pd.DataFrame, batch_size: int = 100) -> int:
        """Insert reviews in batches with automatic bank creation"""

        bank_mapping = self.get_bank_mapping()
        records = []

        for _, row in df.iterrows():
            row_bank_name = str(row['bank_name']).strip()
            normalized_name = row_bank_name.lower()
            bank_id = bank_mapping.get(normalized_name)

            if not bank_id:
                bank_id = self.insert_or_create_bank(row_bank_name)
                if bank_id:
                    bank_mapping[normalized_name] = bank_id
                else:
                    logger.warning(f"Failed to insert bank for review: {row_bank_name}")
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
                logger.error(f"Failed to insert batch {i//batch_size + 1}: {e}")
                self.db.connection.rollback()

        logger.info(f"Total reviews inserted: {total_inserted}")
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
                       COALESCE(ROUND(AVG(r.rating)::numeric, 2), 0) as avg_rating,
                       COALESCE(ROUND(AVG(r.sentiment_score)::numeric, 3), 0) as avg_sentiment
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
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate SQL dump: {e}")
            raise


def main():
    CSV_PATH = r"C:\Users\admin\sentiment-analysis-week2\data\processed_data\reviews_with_sentiment.csv"
    SQL_DUMP_PATH = r"C:\Users\admin\sentiment-analysis-week2\data\sql_dump\bank_reviews.sql"

    db = DatabaseConnection()
    try:
        if not db.connect():
            raise ConnectionError("Failed to connect to database")

        # Optional: Comment out insert_banks_data to avoid pre-defining only 3 banks
        create_tables(db)
        # insert_banks_data(db)  # ← optional: remove if you auto-create all banks

        loader = ReviewDataLoader(db)
        df = loader.load_csv_data(CSV_PATH)

        if len(df) < 400:
            logger.warning(f"Only {len(df)} reviews available, minimum 400 required")

        inserted_count = loader.insert_reviews(df)
        logger.info(f"✅ Reviews inserted: {inserted_count}")

        stats = loader.verify_data_integrity()
        print("\nDATA INTEGRITY STATS:")
        for key, value in stats.items():
            print(f"{key}: {value}")

        loader.generate_sql_dump(SQL_DUMP_PATH)
        print(f"\nSQL dump saved to: {SQL_DUMP_PATH}")

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()