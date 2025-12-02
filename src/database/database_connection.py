import os
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
import logging
from typing import Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'bank_reviews')


def create_database_if_not_exists():
    """Create PostgreSQL database if it does not exist (autocommit)."""
    try:
        # Connect to default 'postgres' database
        conn = psycopg2.connect(
            dbname='postgres',
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Check if DB exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (DB_NAME,))
        exists = cur.fetchone()
        if not exists:
            cur.execute(f"CREATE DATABASE {DB_NAME};")
            logger.info(f"✅ Database '{DB_NAME}' created successfully")
        else:
            logger.info(f"ℹ️ Database '{DB_NAME}' already exists")

        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"❌ Failed to create database: {e}")
        raise


class DatabaseConnection:
    """PostgreSQL database connection manager with error handling."""
    
    def __init__(self):
        self.config = {
            'dbname': DB_NAME,
            'user': DB_USER,
            'password': DB_PASSWORD,
            'host': DB_HOST,
            'port': DB_PORT
        }
        self.connection: Optional[psycopg2.extensions.connection] = None

    def connect(self) -> Optional[psycopg2.extensions.connection]:
        """Establish database connection."""
        try:
            self.connection = psycopg2.connect(
                **self.config,
                cursor_factory=RealDictCursor
            )
            logger.info("✅ Database connection established successfully")
            return self.connection
        except psycopg2.Error as e:
            logger.error(f"❌ Database connection failed: {e}")
            return None

    def execute_query(self, query: str, params: tuple = None, fetch: bool = True) -> Any:
        """Execute SQL query with optional fetch."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params or ())
                if fetch:
                    if query.strip().upper().startswith('SELECT') or 'RETURNING' in query.upper():
                        return cursor.fetchall()
                self.connection.commit()
                return cursor.rowcount
        except psycopg2.Error as e:
            logger.error(f"❌ Query execution failed: {e}")
            self.connection.rollback()
            raise

    def close(self):
        """Close database connection."""
        if self.connection and not self.connection.closed:
            self.connection.close()
            logger.info("ℹ️ Database connection closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_tables(db: DatabaseConnection):
    """Create tables if they do not exist."""
    try:
        # Banks table
        db.execute_query("""
        CREATE TABLE IF NOT EXISTS banks (
            bank_id SERIAL PRIMARY KEY,
            bank_name VARCHAR(100) NOT NULL,
            app_name VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """, fetch=False)

        # Reviews table
        db.execute_query("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id SERIAL PRIMARY KEY,
            bank_id INTEGER REFERENCES banks(bank_id) ON DELETE CASCADE,
            review_text TEXT,
            rating FLOAT,
            review_date DATE,
            sentiment_label VARCHAR(50),
            sentiment_score FLOAT,
            source VARCHAR(50),
            cleaned_text TEXT,
            keywords TEXT[],
            theme VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """, fetch=False)

        # Indexes
        db.execute_query("CREATE INDEX IF NOT EXISTS idx_reviews_bank_id ON reviews(bank_id);", fetch=False)
        db.execute_query("CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews(sentiment_score);", fetch=False)
        db.execute_query("CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews(review_date);", fetch=False)
        db.execute_query("CREATE INDEX IF NOT EXISTS idx_banks_name ON banks(bank_name);", fetch=False)

        # View
        db.execute_query("""
        CREATE OR REPLACE VIEW bank_sentiment_summary AS
        SELECT 
            b.bank_name,
            COUNT(r.review_id) AS total_reviews,
            AVG(r.rating) AS avg_rating,
            AVG(r.sentiment_score) AS avg_sentiment,
            SUM(CASE WHEN r.sentiment_label = 'positive' THEN 1 ELSE 0 END) AS positive_count,
            SUM(CASE WHEN r.sentiment_label = 'negative' THEN 1 ELSE 0 END) AS negative_count,
            SUM(CASE WHEN r.sentiment_label = 'neutral' THEN 1 ELSE 0 END) AS neutral_count
        FROM banks b
        LEFT JOIN reviews r ON b.bank_id = r.bank_id
        GROUP BY b.bank_name;
        """, fetch=False)

        logger.info("✅ Database tables and view created successfully")

    except Exception as e:
        logger.error(f"❌ Failed to create tables: {e}")
        raise


def insert_banks_data(db: DatabaseConnection):
    """Insert initial bank data."""
    banks = [
        ('Commercial Bank of Ethiopia', 'CBE Birr'),
        ('Bank of Abyssinia', 'BOA Mobile'),
        ('Dashen Bank', 'Dashen Mobile')
    ]
    try:
        # Clear existing data
        db.execute_query("DELETE FROM banks;", fetch=False)

        # Insert new data and fetch RETURNING bank_id
        insert_query = "INSERT INTO banks (bank_name, app_name) VALUES (%s, %s) RETURNING bank_id;"
        for bank in banks:
            result = db.execute_query(insert_query, bank, fetch=True)
            logger.info(f"Inserted bank: {bank[0]} with ID: {result[0]['bank_id']}")

    except Exception as e:
        logger.error(f"❌ Failed to insert bank data: {e}")
        raise


if __name__ == "__main__":
    # Step 1: Create database if it does not exist
    create_database_if_not_exists()

    # Step 2: Connect and create tables / insert data
    with DatabaseConnection() as db:
        create_tables(db)
        insert_banks_data(db)
