"""
Data Collection Module for scraping bank app reviews from Google Play Store
"""

import pandas as pd
from google_play_scraper import app, reviews, Sort
import time
from tqdm import tqdm
import logging
from typing import Dict, List, Optional
from config.banks_config import BANKS_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewScraper:
    """Scrape reviews from Google Play Store for multiple banks"""
    
    def __init__(self, banks_config: Dict = BANKS_CONFIG):
        self.banks_config = banks_config
        self.reviews_data = []
        
    def scrape_bank_reviews(self, bank_key: str, count: int = 400) -> List[Dict]:
        """
        Scrape reviews for a specific bank
        
        Args:
            bank_key: Key identifier for the bank
            count: Number of reviews to scrape
            
        Returns:
            List of review dictionaries
        """
        bank_config = self.banks_config[bank_key]
        
        try:
            logger.info(f"Scraping reviews for {bank_config['name']}")
            
            # Scrape reviews with continuation token for pagination
            continuation_token = None
            scraped_reviews = []
            
            while len(scraped_reviews) < count:
                batch_count = min(200, count - len(scraped_reviews))
                
                result, continuation_token = reviews(
                    bank_config['app_id'],
                    lang=bank_config['language'],
                    country=bank_config['country'],
                    sort=Sort.NEWEST,
                    count=batch_count,
                    continuation_token=continuation_token
                )
                
                if not result:
                    break
                    
                # Add bank information to each review
                for review in result:
                    review_data = {
                        'review_id': review['reviewId'],
                        'content': review['content'],
                        'score': review['score'],
                        'thumbs_up_count': review['thumbsUpCount'],
                        'review_created_version': review['reviewCreatedVersion'],
                        'at': review['at'].strftime('%Y-%m-%d'),
                        'reply_content': review.get('replyContent', ''),
                        'replied_at': review.get('repliedAt', ''),
                        'bank': bank_config['name'],
                        'bank_key': bank_key,
                        'source': 'Google Play Store'
                    }
                    scraped_reviews.append(review_data)
                
                logger.info(f"Scraped {len(scraped_reviews)} reviews for {bank_config['name']}")
                
                if continuation_token is None:
                    break
                    
                # Rate limiting
                time.sleep(2)
                
            return scraped_reviews
            
        except Exception as e:
            logger.error(f"Error scraping reviews for {bank_config['name']}: {str(e)}")
            return []
    
    def scrape_all_banks(self, reviews_per_bank: int = 400) -> pd.DataFrame:
        """
        Scrape reviews for all configured banks
        
        Args:
            reviews_per_bank: Number of reviews to scrape per bank
            
        Returns:
            DataFrame containing all reviews
        """
        all_reviews = []
        
        for bank_key in self.banks_config.keys():
            bank_reviews = self.scrape_bank_reviews(bank_key, reviews_per_bank)
            all_reviews.extend(bank_reviews)
            
            # Rate limiting between banks
            time.sleep(3)
        
        df = pd.DataFrame(all_reviews)
        logger.info(f"Total reviews scraped: {len(df)}")
        
        return df

def save_raw_data(df: pd.DataFrame, filename: str = "raw_reviews.csv"):
    """Save raw scraped data to CSV"""
    try:
        df.to_csv(f"data/raw/{filename}", index=False)
        logger.info(f"Raw data saved to data/raw/{filename}")
    except Exception as e:
        logger.error(f"Error saving raw data: {str(e)}")

if __name__ == "__main__":
    scraper = ReviewScraper()
    reviews_df = scraper.scrape_all_banks(reviews_per_bank=400)
    save_raw_data(reviews_df)