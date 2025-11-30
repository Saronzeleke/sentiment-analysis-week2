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

# Set up logging without emojis for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)  # Use stdout for better encoding
    ]
)
logger = logging.getLogger(__name__)

# Try multiple possible app IDs for Ethiopian banks
ETHIOPIAN_BANKS = {
    "cbe": {
        "app_ids": [
            "com.cbe.mobilebanking",  # Primary
            "com.cbe.ethiopia",       # Alternative
            "com.cbe.digital",        # Alternative  
            "com.combank.ethiopia"    # Alternative
        ],
        "name": "Commercial Bank of Ethiopia"
    },
    "awash": {
        "app_ids": [
            "com.awash.bank",         # Primary
            "com.awash.mobile",       # Alternative
            "com.awash.digital",      # Alternative
            "com.awash.ethiopia"      # Alternative
        ],
        "name": "Awash Bank"
    },
    "dashen": {
        "app_ids": [
            "com.dashen.sc",          # Primary
            "com.dashen.bank",        # Alternative
            "com.dashen.mobile",      # Alternative
            "com.dashen.ethiopia"     # Alternative
        ],
        "name": "Dashen Bank"
    }
}

class EthiopianBankReviewScraper:
    """Scrape reviews from Google Play Store for Ethiopian banks"""
    
    def __init__(self, banks_config: dict = ETHIOPIAN_BANKS):
        self.banks_config = banks_config
        self.reviews_data = []
        self.working_app_ids = {}  # Store working app IDs
    
    def find_working_app_id(self, bank_key: str) -> str:
        """
        Try multiple app IDs to find one that works
        """
        bank_config = self.banks_config[bank_key]
        
        logger.info(f"Searching for working app ID for {bank_config['name']}...")
        
        for app_id in bank_config["app_ids"]:
            try:
                logger.info(f"  Trying: {app_id}")
                app_info = app(app_id, lang="en", country="us")  # Try US store
                logger.info(f"  SUCCESS - Found: {app_info.get('title', 'Unknown')}")
                logger.info(f"    Rating: {app_info.get('score', 'N/A')}")
                logger.info(f"    Installs: {app_info.get('installs', 'N/A')}")
                self.working_app_ids[bank_key] = app_id
                return app_id
            except Exception as e:
                logger.info(f"  FAILED - {app_id}: {str(e)}")
                continue
        
        logger.warning(f"  No working app ID found for {bank_config['name']}")
        return None
    
    def scrape_bank_reviews(self, bank_key: str, count: int = 100) -> list:
        """
        Scrape reviews for a specific bank
        """
        if bank_key not in self.working_app_ids:
            logger.warning(f"No working app ID for {self.banks_config[bank_key]['name']}")
            return []
        
        app_id = self.working_app_ids[bank_key]
        bank_name = self.banks_config[bank_key]["name"]
        
        try:
            logger.info(f"Scraping reviews for {bank_name}...")
            
            scraped_reviews = []
            continuation_token = None
            
            while len(scraped_reviews) < count:
                try:
                    batch_count = min(100, count - len(scraped_reviews))
                    
                    result, continuation_token = reviews(
                        app_id,
                        lang="en",
                        country="us",  # Try US store
                        sort=Sort.NEWEST,
                        count=batch_count,
                        continuation_token=continuation_token
                    )
                    
                    if not result:
                        logger.info(f"No more reviews available for {bank_name}")
                        break
                    
                    # Process reviews
                    for review in result:
                        review_data = {
                            'review_id': review['reviewId'],
                            'content': review['content'],
                            'rating': review['score'],
                            'thumbs_up_count': review['thumbsUpCount'],
                            'review_created_version': review.get('reviewCreatedVersion', ''),
                            'date': review['at'].strftime('%Y-%m-%d'),
                            'reply_content': review.get('replyContent', ''),
                            'replied_at': review.get('repliedAt', ''),
                            'bank': bank_name,
                            'bank_key': bank_key,
                            'source': 'Google Play Store'
                        }
                        scraped_reviews.append(review_data)
                    
                    logger.info(f"Collected {len(scraped_reviews)} reviews for {bank_name}")
                    
                    if continuation_token is None:
                        break
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Error in batch scraping: {str(e)}")
                    break
            
            logger.info(f"Successfully collected {len(scraped_reviews)} reviews for {bank_name}")
            return scraped_reviews
            
        except Exception as e:
            logger.error(f"Error scraping reviews for {bank_name}: {str(e)}")
            return []
    
    def scrape_all_banks(self, reviews_per_bank: int = 100) -> pd.DataFrame:
        """
        Scrape reviews for all configured banks
        """
        logger.info("Starting review scraping for all Ethiopian banks...")
        
        all_reviews = []
        successful_banks = 0
        
        for bank_key in tqdm(self.banks_config.keys(), desc="Scraping banks"):
            # Find working app ID first
            working_app_id = self.find_working_app_id(bank_key)
            
            if working_app_id:
                bank_reviews = self.scrape_bank_reviews(bank_key, reviews_per_bank)
                
                if bank_reviews:
                    all_reviews.extend(bank_reviews)
                    successful_banks += 1
                    logger.info(f"SUCCESS - {self.banks_config[bank_key]['name']}: {len(bank_reviews)} reviews")
                else:
                    logger.warning(f"WARNING - No reviews collected for {self.banks_config[bank_key]['name']}")
            else:
                logger.warning(f"SKIPPING - No working app found for {self.banks_config[bank_key]['name']}")
            
            # Rate limiting between banks
            time.sleep(2)
        
        # Create DataFrame
        if all_reviews:
            df = pd.DataFrame(all_reviews)
            logger.info(f"TOTAL - Collected {len(df)} reviews from {successful_banks} banks")
            
            # Log distribution
            bank_counts = df['bank'].value_counts()
            for bank, count in bank_counts.items():
                logger.info(f"  - {bank}: {count} reviews")
                
            return df
        else:
            logger.error("FAILED - No reviews were collected from any bank!")
            return pd.DataFrame()
    
    def save_raw_data(self, df: pd.DataFrame, filename: str = "raw_reviews.csv"):
        """Save raw scraped data to CSV"""
        try:
            # Create directories if they don't exist
            os.makedirs('data/raw', exist_ok=True)
            
            filepath = f"data/raw/{filename}"
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"SAVED - Raw data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"ERROR saving raw data: {str(e)}")

def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("ETHIOPIAN BANK REVIEWS DATA COLLECTION")
    logger.info("=" * 60)
    
    # Initialize scraper
    scraper = EthiopianBankReviewScraper()
    
    # Start with small batch for testing
    reviews_df = scraper.scrape_all_banks(reviews_per_bank=50)
    
    if not reviews_df.empty:
        # Save raw data
        scraper.save_raw_data(reviews_df)
        
        # Display summary
        logger.info("\nDATA COLLECTION SUMMARY:")
        logger.info(f"   Total Reviews: {len(reviews_df)}")
        logger.info(f"   Banks Covered: {reviews_df['bank'].nunique()}")
        logger.info(f"   Date Range: {reviews_df['date'].min()} to {reviews_df['date'].max()}")
        
        logger.info("SUCCESS - Data collection completed!")
        return reviews_df
    else:
        logger.error("FAILED - No reviews collected")
        return pd.DataFrame()

if __name__ == "__main__":
    reviews_df = main()