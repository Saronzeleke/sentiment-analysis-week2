# src/data_collection.py
"""
Enhanced Google Play Store Review Scraper
With better error handling and multiple fallback options
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google_play_scraper import app, Sort, reviews
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm
import logging
from config.banks_config import APP_IDS, BANK_NAMES, SCRAPING_CONFIG, DATA_PATHS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PlayStoreScraper:
    """Enhanced scraper class with better error handling"""

    def __init__(self):
        self.app_ids = APP_IDS
        self.bank_names = BANK_NAMES
        self.reviews_per_bank = SCRAPING_CONFIG['reviews_per_bank']
        self.lang = SCRAPING_CONFIG['lang']
        self.country = SCRAPING_CONFIG['country']
        self.max_retries = SCRAPING_CONFIG['max_retries']
        self.collected_data = []

    def test_app_connection(self, app_id):
        """Test if we can connect to the app"""
        try:
            app_info = app(app_id, lang=self.lang, country=self.country)
            logger.info(f"✅ Successfully connected to {app_info.get('title', 'Unknown')}")
            return True
        except Exception as e:
            logger.warning(f"❌ Cannot connect to {app_id}: {str(e)}")
            return False

    def scrape_reviews_with_fallback(self, app_id, bank_code, count=400):
        """Scrape reviews with multiple fallback strategies"""
        logger.info(f"📥 Scraping reviews for {self.bank_names[bank_code]}...")
        
        all_reviews = []
        continuation_token = None
        
        for attempt in range(self.max_retries):
            try:
                # Try to get reviews in batches
                batch_size = min(100, count - len(all_reviews))
                
                result, continuation_token = reviews(
                    app_id,
                    lang=self.lang,
                    country=self.country,
                    sort=Sort.NEWEST,
                    count=batch_size,
                    continuation_token=continuation_token
                )
                
                if result:
                    all_reviews.extend(result)
                    logger.info(f"✅ Batch {attempt + 1}: Collected {len(result)} reviews")
                
                # If we have enough reviews or no more available, break
                if len(all_reviews) >= count or continuation_token is None:
                    break
                    
                # Be polite to the server
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    logger.info("🔄 Retrying after 3 seconds...")
                    time.sleep(3)
                else:
                    logger.error(f"❌ Failed after {self.max_retries} attempts")
                    break
        
        return all_reviews

    def process_reviews(self, reviews_data, bank_code):
        """Process raw review data into clean format"""
        processed = []
        
        for review in reviews_data:
            try:
                processed.append({
                    'review_id': review.get('reviewId', f'{bank_code}_{len(processed)}'),
                    'review_text': review.get('content', '').strip(),
                    'rating': review.get('score', 0),
                    'review_date': review.get('at', datetime.now()),
                    'user_name': review.get('userName', 'Anonymous'),
                    'thumbs_up': review.get('thumbsUpCount', 0),
                    'reply_content': review.get('replyContent', ''),
                    'bank_code': bank_code,
                    'bank_name': self.bank_names[bank_code],
                    'app_version': review.get('reviewCreatedVersion', 'N/A'),
                    'source': 'Google Play Store'
                })
            except Exception as e:
                logger.warning(f"⚠️ Error processing review: {str(e)}")
                continue
                
        return processed

    def scrape_all_banks(self):
        """Main orchestration method to scrape all banks"""
        logger.info("🎯 Starting Google Play Store Review Scraper")
        logger.info("=" * 60)
        
        all_reviews = []
        app_info_list = []

        # Step 1: Test connections and get app info
        logger.info("🔍 Step 1: Testing app connections...")
        for bank_code, app_id in self.app_ids.items():
            bank_name = self.bank_names[bank_code]
            logger.info(f"\n🏦 Processing {bank_name}...")
            
            if self.test_app_connection(app_id):
                try:
                    app_info = app(app_id, lang=self.lang, country=self.country)
                    app_info_list.append({
                        'bank_code': bank_code,
                        'bank_name': bank_name,
                        'app_id': app_id,
                        'app_title': app_info.get('title', 'N/A'),
                        'rating': app_info.get('score', 0),
                        'reviews_count': app_info.get('reviews', 0),
                        'installs': app_info.get('installs', 'N/A'),
                        'version': app_info.get('version', 'N/A')
                    })
                except Exception as e:
                    logger.warning(f"⚠️ Could not get app info for {bank_name}: {str(e)}")

        # Step 2: Scrape reviews
        logger.info("\n📥 Step 2: Scraping reviews...")
        for bank_code, app_id in tqdm(self.app_ids.items(), desc="Scraping Banks"):
            bank_name = self.bank_names[bank_code]
            
            reviews_data = self.scrape_reviews_with_fallback(app_id, bank_code, self.reviews_per_bank)
            
            if reviews_data:
                processed_reviews = self.process_reviews(reviews_data, bank_code)
                all_reviews.extend(processed_reviews)
                logger.info(f"✅ {bank_name}: Collected {len(processed_reviews)} reviews")
            else:
                logger.warning(f"❌ {bank_name}: No reviews collected")
            
            # Be polite between banks
            time.sleep(2)

        # Step 3: Save results
        if all_reviews:
            df = pd.DataFrame(all_reviews)
            
            # Ensure directories exist
            os.makedirs(DATA_PATHS['raw'], exist_ok=True)
            
            # Save raw reviews
            df.to_csv(DATA_PATHS['raw_reviews'], index=False, encoding='utf-8')
            
            # Save app info if available
            if app_info_list:
                app_info_df = pd.DataFrame(app_info_list)
                app_info_df.to_csv(f"{DATA_PATHS['raw']}/app_info.csv", index=False)
            
            # Generate summary
            self.generate_scraping_summary(df, app_info_list)
            
            return df
        else:
            logger.error("❌ No reviews were collected from any bank!")
            return pd.DataFrame()

    def generate_scraping_summary(self, df, app_info_list):
        """Generate a comprehensive scraping summary"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 SCRAPING SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"✅ Total Reviews Collected: {len(df):,}")
        
        # Reviews by bank
        logger.info("\n📈 Reviews by Bank:")
        bank_counts = df['bank_name'].value_counts()
        for bank, count in bank_counts.items():
            logger.info(f"   🏦 {bank}: {count} reviews")
        
        # Rating distribution
        logger.info("\n⭐ Rating Distribution:")
        rating_counts = df['rating'].value_counts().sort_index(ascending=False)
        for rating, count in rating_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"   {''.join(['⭐'] * int(rating))}: {count} ({percentage:.1f}%)")
        
        # Date range
        if 'review_date' in df.columns:
            date_range = f"{df['review_date'].min()} to {df['review_date'].max()}"
            logger.info(f"\n📅 Date Range: {date_range}")
        
        logger.info(f"\n💾 Data saved to: {DATA_PATHS['raw_reviews']}")

    def display_sample_data(self, df, num_samples=2):
        """Display sample reviews for verification"""
        logger.info("\n" + "=" * 60)
        logger.info("👀 SAMPLE REVIEWS")
        logger.info("=" * 60)
        
        for bank in df['bank_name'].unique():
            bank_reviews = df[df['bank_name'] == bank].head(num_samples)
            logger.info(f"\n🏦 {bank}:")
            
            for idx, review in bank_reviews.iterrows():
                logger.info(f"\n   Rating: {'⭐' * review['rating']}")
                logger.info(f"   Review: {review['review_text'][:100]}...")
                logger.info(f"   Date: {review['review_date']}")
                logger.info("   " + "-" * 40)


def main():
    """Main execution function"""
    scraper = PlayStoreScraper()
    df = scraper.scrape_all_banks()
    
    if not df.empty:
        scraper.display_sample_data(df)
    
    return df


if __name__ == "__main__":
    reviews_df = main()