
"""
Google Play Store Review Scraper
Task 1: Data Collection

This script scrapes user reviews from Google Play Store for three Ethiopian banks.
Target: 400+ reviews per bank (1200 total minimum)
"""

import sys
import os
# Add parent directory to path to allow importing modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google_play_scraper import app, Sort, reviews_all, reviews
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm
from config.banks_config import APP_IDS, BANK_NAMES, SCRAPING_CONFIG, DATA_PATHS


class PlayStoreScraper:
    """Scraper class for Google Play Store reviews"""

    def __init__(self):
        # Load configuration variables from the config file
        self.app_ids = APP_IDS
        self.bank_names = BANK_NAMES
        self.reviews_per_bank = SCRAPING_CONFIG['reviews_per_bank']
        self.lang = SCRAPING_CONFIG['lang']
        self.country = SCRAPING_CONFIG['country']
        self.max_retries = SCRAPING_CONFIG['max_retries']

    def get_app_info(self, app_id):
        """
        Get basic information about the app (rating, total reviews, etc.)
        """
        try:
            # Fetch app details from Google Play Store
            result = app(app_id, lang=self.lang, country=self.country)
            return {
                'app_id': app_id,
                'title': result.get('title', 'N/A'),
                'score': result.get('score', 0),
                'ratings': result.get('ratings', 0),
                'reviews': result.get('reviews', 0),
                'installs': result.get('installs', 'N/A')
            }
        except Exception as e:
            print(f"Error getting app info for {app_id}: {str(e)}")
            return None

    def scrape_reviews(self, app_id, count=400):
        """
        Scrape reviews for a specific app.
        Attempts to fetch 'count' number of reviews, sorted by newest first.
        Includes a retry mechanism for stability.
        """
        print(f"\nScraping reviews for {app_id}...")

        # Retry loop to handle potential network errors or API issues
        for attempt in range(self.max_retries):
            try:
                # Use the google_play_scraper 'reviews' function
                result, _ = reviews(
                    app_id,
                    lang=self.lang,
                    country=self.country,
                    sort=Sort.NEWEST,      # Get the most recent reviews
                    count=count,           # Number of reviews to fetch
                    filter_score_with=None # Fetch all ratings (1-5 stars)
                )

                print(f"Successfully scraped {len(result)} reviews")
                return result

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                # Wait before retrying if it's not the last attempt
                if attempt < self.max_retries - 1:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"Failed to scrape reviews after {self.max_retries} attempts")
                    return []

        return []

    def process_reviews(self, reviews_data, bank_code):
        """
        Process raw review data from the scraper into a clean dictionary format.
        Extracts only the relevant fields we need for analysis.
        """
        processed = []

        for review in reviews_data:
            processed.append({
                'review_id': review.get('reviewId', ''),
                'review_text': review.get('content', ''),
                'rating': review.get('score', 0),
                'review_date': review.get('at', datetime.now()),
                'user_name': review.get('userName', 'Anonymous'),
                'thumbs_up': review.get('thumbsUpCount', 0),
                'reply_content': review.get('replyContent', None),
                'bank_code': bank_code,
                'bank_name': self.bank_names[bank_code],
                'app_id': review.get('reviewCreatedVersion', 'N/A'),
                'source': 'Google Play'
            })

        return processed

    def scrape_all_banks(self):
        """
        Main orchestration method:
        1. Iterates through all configured banks
        2. Fetches app metadata
        3. Scrapes reviews for each bank
        4. Combines all data into a single DataFrame
        5. Saves the raw data to CSV
        """
        all_reviews = []
        app_info_list = []

        print("=" * 60)
        print("Starting Google Play Store Review Scraper")
        print("=" * 60)

        # --- Phase 1: Fetch App Info ---
        print("\n[1/2] Fetching app information...")
        for bank_code, app_id in self.app_ids.items():
            print(f"\n{bank_code}: {self.bank_names[bank_code]}")
            print(f"App ID: {app_id}")

            info = self.get_app_info(app_id)
            if info:
                info['bank_code'] = bank_code
                info['bank_name'] = self.bank_names[bank_code]
                app_info_list.append(info)
                print(f"Current Rating: {info['score']}")
                print(f"Total Ratings: {info['ratings']}")
                print(f"Total Reviews: {info['reviews']}")

        # Save the gathered app info to a CSV file
        if app_info_list:
            app_info_df = pd.DataFrame(app_info_list)
            os.makedirs(DATA_PATHS['raw'], exist_ok=True)
            app_info_df.to_csv(f"{DATA_PATHS['raw']}/app_info.csv", index=False)
            print(f"\nApp information saved to {DATA_PATHS['raw']}/app_info.csv")

        # --- Phase 2: Scrape Reviews ---
        print("\n[2/2] Scraping reviews...")
        # Use tqdm to show a progress bar for the banks
        for bank_code, app_id in tqdm(self.app_ids.items(), desc="Banks"):
            # Fetch the reviews
            reviews_data = self.scrape_reviews(app_id, self.reviews_per_bank)

            if reviews_data:
                # Process and format the data
                processed = self.process_reviews(reviews_data, bank_code)
                all_reviews.extend(processed)
                print(f"Collected {len(processed)} reviews for {self.bank_names[bank_code]}")
            else:
                print(f"WARNING: No reviews collected for {self.bank_names[bank_code]}")

            # Small delay between banks to be polite to the server
            time.sleep(2)

        # --- Phase 3: Save Data ---
        if all_reviews:
            df = pd.DataFrame(all_reviews)

            # Save raw data to CSV
            os.makedirs(DATA_PATHS['raw'], exist_ok=True)
            df.to_csv(DATA_PATHS['raw_reviews'], index=False)

            print("\n" + "=" * 60)
            print("Scraping Complete!")
            print("=" * 60)
            print(f"\nTotal reviews collected: {len(df)}")
            
            # Print stats per bank
            print(f"Reviews per bank:")
            for bank_code in self.bank_names.keys():
                count = len(df[df['bank_code'] == bank_code])
                print(f"  {self.bank_names[bank_code]}: {count}")

            print(f"\nData saved to: {DATA_PATHS['raw_reviews']}")

            return df
        else:
            print("\nERROR: No reviews were collected!")
            return pd.DataFrame()

    def display_sample_reviews(self, df, n=3):
        """
        Display sample reviews from each bank to verify data quality.
        """
        print("\n" + "=" * 60)
        print("Sample Reviews")
        print("=" * 60)

        for bank_code in self.bank_names.keys():
            bank_df = df[df['bank_code'] == bank_code]
            if not bank_df.empty:
                print(f"\n{self.bank_names[bank_code]}:")
                print("-" * 60)
                samples = bank_df.head(n)
                for idx, row in samples.iterrows():
                    print(f"\nRating: {'⭐' * row['rating']}")
                    print(f"Review: {row['review_text'][:200]}...")
                    print(f"Date: {row['review_date']}")


def main():
    """Main execution function"""

    # Initialize scraper
    scraper = PlayStoreScraper()

    # Scrape all reviews
    df = scraper.scrape_all_banks()

    # Display samples if data was collected
    if not df.empty:
        scraper.display_sample_reviews(df)

    return df


if __name__ == "__main__":
    reviews_df = main()
# """
# Data Collection Module for scraping bank app reviews from Google Play Store
# """

# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import pandas as pd
# from google_play_scraper import app, reviews, Sort
# import time
# from tqdm import tqdm
# import logging
# import sys
# import os
# from datetime import datetime

# # Set up logging without emojis for Windows compatibility
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('data_collection.log', encoding='utf-8'),
#         logging.StreamHandler(sys.stdout)  # Use stdout for better encoding
#     ]
# )
# logger = logging.getLogger(__name__)

# # Try multiple possible app IDs for Ethiopian banks
# ETHIOPIAN_BANKS = {
#     "cbe": {
#         "app_ids": [
#             "com.cbe.mobilebanking",  # Primary
#             "com.cbe.ethiopia",       # Alternative
#             "com.cbe.digital",        # Alternative  
#             "com.combank.ethiopia"    # Alternative
#         ],
#         "name": "Commercial Bank of Ethiopia"
#     },
#     "awash": {
#         "app_ids": [
#             "com.awash.bank",         # Primary
#             "com.awash.mobile",       # Alternative
#             "com.awash.digital",      # Alternative
#             "com.awash.ethiopia"      # Alternative
#         ],
#         "name": "Awash Bank"
#     },
#     "dashen": {
#         "app_ids": [
#             "com.dashen.sc",          # Primary
#             "com.dashen.bank",        # Alternative
#             "com.dashen.mobile",      # Alternative
#             "com.dashen.ethiopia"     # Alternative
#         ],
#         "name": "Dashen Bank"
#     }
# }

# class EthiopianBankReviewScraper:
#     """Scrape reviews from Google Play Store for Ethiopian banks"""
    
#     def __init__(self, banks_config: dict = ETHIOPIAN_BANKS):
#         self.banks_config = banks_config
#         self.reviews_data = []
#         self.working_app_ids = {}  # Store working app IDs
    
#     def find_working_app_id(self, bank_key: str) -> str:
#         """
#         Try multiple app IDs to find one that works
#         """
#         bank_config = self.banks_config[bank_key]
        
#         logger.info(f"Searching for working app ID for {bank_config['name']}...")
        
#         for app_id in bank_config["app_ids"]:
#             try:
#                 logger.info(f"  Trying: {app_id}")
#                 app_info = app(app_id, lang="en", country="us")  # Try US store
#                 logger.info(f"  SUCCESS - Found: {app_info.get('title', 'Unknown')}")
#                 logger.info(f"    Rating: {app_info.get('score', 'N/A')}")
#                 logger.info(f"    Installs: {app_info.get('installs', 'N/A')}")
#                 self.working_app_ids[bank_key] = app_id
#                 return app_id
#             except Exception as e:
#                 logger.info(f"  FAILED - {app_id}: {str(e)}")
#                 continue
        
#         logger.warning(f"  No working app ID found for {bank_config['name']}")
#         return None
    
#     def scrape_bank_reviews(self, bank_key: str, count: int = 100) -> list:
#         """
#         Scrape reviews for a specific bank
#         """
#         if bank_key not in self.working_app_ids:
#             logger.warning(f"No working app ID for {self.banks_config[bank_key]['name']}")
#             return []
        
#         app_id = self.working_app_ids[bank_key]
#         bank_name = self.banks_config[bank_key]["name"]
        
#         try:
#             logger.info(f"Scraping reviews for {bank_name}...")
            
#             scraped_reviews = []
#             continuation_token = None
            
#             while len(scraped_reviews) < count:
#                 try:
#                     batch_count = min(100, count - len(scraped_reviews))
                    
#                     result, continuation_token = reviews(
#                         app_id,
#                         lang="en",
#                         country="us",  # Try US store
#                         sort=Sort.NEWEST,
#                         count=batch_count,
#                         continuation_token=continuation_token
#                     )
                    
#                     if not result:
#                         logger.info(f"No more reviews available for {bank_name}")
#                         break
                    
#                     # Process reviews
#                     for review in result:
#                         review_data = {
#                             'review_id': review['reviewId'],
#                             'content': review['content'],
#                             'rating': review['score'],
#                             'thumbs_up_count': review['thumbsUpCount'],
#                             'review_created_version': review.get('reviewCreatedVersion', ''),
#                             'date': review['at'].strftime('%Y-%m-%d'),
#                             'reply_content': review.get('replyContent', ''),
#                             'replied_at': review.get('repliedAt', ''),
#                             'bank': bank_name,
#                             'bank_key': bank_key,
#                             'source': 'Google Play Store'
#                         }
#                         scraped_reviews.append(review_data)
                    
#                     logger.info(f"Collected {len(scraped_reviews)} reviews for {bank_name}")
                    
#                     if continuation_token is None:
#                         break
                    
#                     # Rate limiting
#                     time.sleep(1)
                    
#                 except Exception as e:
#                     logger.warning(f"Error in batch scraping: {str(e)}")
#                     break
            
#             logger.info(f"Successfully collected {len(scraped_reviews)} reviews for {bank_name}")
#             return scraped_reviews
            
#         except Exception as e:
#             logger.error(f"Error scraping reviews for {bank_name}: {str(e)}")
#             return []
    
#     def scrape_all_banks(self, reviews_per_bank: int = 100) -> pd.DataFrame:
#         """
#         Scrape reviews for all configured banks
#         """
#         logger.info("Starting review scraping for all Ethiopian banks...")
        
#         all_reviews = []
#         successful_banks = 0
        
#         for bank_key in tqdm(self.banks_config.keys(), desc="Scraping banks"):
#             # Find working app ID first
#             working_app_id = self.find_working_app_id(bank_key)
            
#             if working_app_id:
#                 bank_reviews = self.scrape_bank_reviews(bank_key, reviews_per_bank)
                
#                 if bank_reviews:
#                     all_reviews.extend(bank_reviews)
#                     successful_banks += 1
#                     logger.info(f"SUCCESS - {self.banks_config[bank_key]['name']}: {len(bank_reviews)} reviews")
#                 else:
#                     logger.warning(f"WARNING - No reviews collected for {self.banks_config[bank_key]['name']}")
#             else:
#                 logger.warning(f"SKIPPING - No working app found for {self.banks_config[bank_key]['name']}")
            
#             # Rate limiting between banks
#             time.sleep(2)
        
#         # Create DataFrame
#         if all_reviews:
#             df = pd.DataFrame(all_reviews)
#             logger.info(f"TOTAL - Collected {len(df)} reviews from {successful_banks} banks")
            
#             # Log distribution
#             bank_counts = df['bank'].value_counts()
#             for bank, count in bank_counts.items():
#                 logger.info(f"  - {bank}: {count} reviews")
                
#             return df
#         else:
#             logger.error("FAILED - No reviews were collected from any bank!")
#             return pd.DataFrame()
    
#     def save_raw_data(self, df: pd.DataFrame, filename: str = "raw_reviews.csv"):
#         """Save raw scraped data to CSV"""
#         try:
#             # Create directories if they don't exist
#             os.makedirs('data/raw', exist_ok=True)
            
#             filepath = f"data/raw/{filename}"
#             df.to_csv(filepath, index=False, encoding='utf-8')
#             logger.info(f"SAVED - Raw data saved to {filepath}")
            
#         except Exception as e:
#             logger.error(f"ERROR saving raw data: {str(e)}")

# def main():
#     """Main execution function"""
#     logger.info("=" * 60)
#     logger.info("ETHIOPIAN BANK REVIEWS DATA COLLECTION")
#     logger.info("=" * 60)
    
#     # Initialize scraper
#     scraper = EthiopianBankReviewScraper()
    
#     # Start with small batch for testing
#     reviews_df = scraper.scrape_all_banks(reviews_per_bank=50)
    
#     if not reviews_df.empty:
#         # Save raw data
#         scraper.save_raw_data(reviews_df)
        
#         # Display summary
#         logger.info("\nDATA COLLECTION SUMMARY:")
#         logger.info(f"   Total Reviews: {len(reviews_df)}")
#         logger.info(f"   Banks Covered: {reviews_df['bank'].nunique()}")
#         logger.info(f"   Date Range: {reviews_df['date'].min()} to {reviews_df['date'].max()}")
        
#         logger.info("SUCCESS - Data collection completed!")
#         return reviews_df
#     else:
#         logger.error("FAILED - No reviews collected")
#         return pd.DataFrame()

# if __name__ == "__main__":
#     reviews_df = main()