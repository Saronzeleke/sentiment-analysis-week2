"""
Data Collection Module for scraping bank app reviews from Google Play Store
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from google_play_scraper import app, reviews, Sort
import time
from tqdm import tqdm
import logging
import sys
import os
from datetime import datetime

# Add config to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CORRECT Ethiopian Bank App IDs - Verified working IDs
ETHIOPIAN_BANKS = {
    "cbe": {
        "app_id": "com.cbe.mobilebanking",  # Commercial Bank of Ethiopia
        "name": "Commercial Bank of Ethiopia",
        "country": "et",
        "language": "en"
    },
    "awash": {
        "app_id": "com.awash.bank.mobilebanking",  # Awash Bank
        "name": "Awash Bank", 
        "country": "et",
        "language": "en"
    },
    "dashen": {
        "app_id": "com.dashenaymariam",  # Dashen Bank
        "name": "Dashen Bank",
        "country": "et", 
        "language": "en"
    }
}

class EthiopianBankReviewScraper:
    """Scrape reviews from Google Play Store for Ethiopian banks"""
    
    def __init__(self, banks_config: dict = ETHIOPIAN_BANKS):
        self.banks_config = banks_config
        self.reviews_data = []
        
    def verify_app_exists(self, bank_key: str) -> bool:
        """
        Verify that the app exists and is accessible
        """
        bank_config = self.banks_config[bank_key]
        
        try:
            logger.info(f"Verifying app for {bank_config['name']}...")
            app_info = app(
                bank_config['app_id'],
                lang=bank_config['language'],
                country=bank_config['country']
            )
            logger.info(f"✓ App found: {app_info.get('title', 'Unknown')}")
            logger.info(f"  - Score: {app_info.get('score', 'N/A')}")
            logger.info(f"  - Installs: {app_info.get('installs', 'N/A')}")
            return True
        except Exception as e:
            logger.error(f"❌ App not found for {bank_config['name']}: {str(e)}")
            return False
    
    def scrape_bank_reviews(self, bank_key: str, count: int = 400) -> list:
        """
        Scrape reviews for a specific Ethiopian bank
        
        Args:
            bank_key: Key identifier for the bank
            count: Number of reviews to scrape
            
        Returns:
            List of review dictionaries
        """
        bank_config = self.banks_config[bank_key]
        
        try:
            logger.info(f"📱 Scraping reviews for {bank_config['name']}...")
            
            # Try multiple sorting methods to get more reviews
            sort_methods = [Sort.NEWEST, Sort.MOST_RELEVANT, Sort.RATING]
            scraped_reviews = []
            continuation_token = None
            
            for sort_method in sort_methods:
                if len(scraped_reviews) >= count:
                    break
                    
                try:
                    logger.info(f"  Trying sort method: {sort_method.name}")
                    
                    while len(scraped_reviews) < count:
                        batch_count = min(100, count - len(scraped_reviews))
                        
                        result, continuation_token = reviews(
                            bank_config['app_id'],
                            lang=bank_config['language'],
                            country=bank_config['country'],
                            sort=sort_method,
                            count=batch_count,
                            continuation_token=continuation_token
                        )
                        
                        if not result:
                            break
                            
                        # Add bank information to each review
                        for review in result:
                            # Check if we already have this review
                            existing_ids = [r['review_id'] for r in scraped_reviews]
                            if review['reviewId'] not in existing_ids:
                                review_data = {
                                    'review_id': review['reviewId'],
                                    'content': review['content'],
                                    'rating': review['score'],  # Changed from 'score' to 'rating'
                                    'thumbs_up_count': review['thumbsUpCount'],
                                    'review_created_version': review.get('reviewCreatedVersion', ''),
                                    'date': review['at'].strftime('%Y-%m-%d'),  # Changed from 'at' to 'date'
                                    'reply_content': review.get('replyContent', ''),
                                    'replied_at': review.get('repliedAt', ''),
                                    'bank': bank_config['name'],
                                    'bank_key': bank_key,
                                    'source': 'Google Play Store'
                                }
                                scraped_reviews.append(review_data)
                        
                        logger.info(f"  Collected {len(scraped_reviews)} reviews so far...")
                        
                        if continuation_token is None:
                            break
                            
                        # Rate limiting
                        time.sleep(1)
                        
                except Exception as e:
                    logger.warning(f"  Sort method {sort_method.name} failed: {str(e)}")
                    continue
            
            logger.info(f"✅ Successfully collected {len(scraped_reviews)} reviews for {bank_config['name']}")
            return scraped_reviews
            
        except Exception as e:
            logger.error(f"❌ Error scraping reviews for {bank_config['name']}: {str(e)}")
            return []
    
    def scrape_all_banks(self, reviews_per_bank: int = 400) -> pd.DataFrame:
        """
        Scrape reviews for all configured Ethiopian banks
        
        Args:
            reviews_per_bank: Number of reviews to scrape per bank
            
        Returns:
            DataFrame containing all reviews
        """
        logger.info("🚀 Starting review scraping for all Ethiopian banks...")
        
        all_reviews = []
        successful_banks = 0
        
        for bank_key in tqdm(self.banks_config.keys(), desc="Scraping banks"):
            # Verify app exists first
            if self.verify_app_exists(bank_key):
                bank_reviews = self.scrape_bank_reviews(bank_key, reviews_per_bank)
                
                if bank_reviews:
                    all_reviews.extend(bank_reviews)
                    successful_banks += 1
                    logger.info(f"✓ {self.banks_config[bank_key]['name']}: {len(bank_reviews)} reviews")
                else:
                    logger.warning(f"⚠️  No reviews collected for {self.banks_config[bank_key]['name']}")
            else:
                logger.warning(f"⚠️  Skipping {self.banks_config[bank_key]['name']} - app not found")
            
            # Rate limiting between banks
            time.sleep(2)
        
        # Create DataFrame
        if all_reviews:
            df = pd.DataFrame(all_reviews)
            logger.info(f"🎉 Total reviews scraped: {len(df)} from {successful_banks} banks")
            
            # Log distribution
            bank_counts = df['bank'].value_counts()
            for bank, count in bank_counts.items():
                logger.info(f"  - {bank}: {count} reviews")
                
            return df
        else:
            logger.error("❌ No reviews were collected from any bank!")
            return pd.DataFrame()
    
    def save_raw_data(self, df: pd.DataFrame, filename: str = "raw_reviews.csv"):
        """Save raw scraped data to CSV with proper error handling"""
        try:
            # Create directories if they don't exist
            os.makedirs('data/raw', exist_ok=True)
            
            filepath = f"data/raw/{filename}"
            df.to_csv(filepath, index=False)
            logger.info(f"💾 Raw data saved to {filepath}")
            
            # Also save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/raw/raw_reviews_{timestamp}.csv"
            df.to_csv(backup_path, index=False)
            logger.info(f"📦 Backup saved to {backup_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving raw data: {str(e)}")

def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("ETHIOPIAN BANK REVIEWS DATA COLLECTION")
    logger.info("=" * 60)
    
    # Initialize scraper
    scraper = EthiopianBankReviewScraper()
    
    # Scrape all banks
    reviews_df = scraper.scrape_all_banks(reviews_per_bank=400)
    
    if not reviews_df.empty:
        # Save raw data
        scraper.save_raw_data(reviews_df)
        
        # Display summary
        logger.info("\n📊 DATA COLLECTION SUMMARY:")
        logger.info(f"   • Total Reviews: {len(reviews_df)}")
        logger.info(f"   • Banks Covered: {reviews_df['bank'].nunique()}")
        logger.info(f"   • Date Range: {reviews_df['date'].min()} to {reviews_df['date'].max()}")
        logger.info(f"   • Rating Distribution:")
        for rating in sorted(reviews_df['rating'].unique()):
            count = len(reviews_df[reviews_df['rating'] == rating])
            logger.info(f"     - {rating} stars: {count} reviews")
        
        logger.info("✅ DATA COLLECTION COMPLETED SUCCESSFULLY!")
        return reviews_df
    else:
        logger.error("❌ DATA COLLECTION FAILED - No reviews collected")
        return pd.DataFrame()

if __name__ == "__main__":
    reviews_df = main()