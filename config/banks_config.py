# config/banks_config.py
"""
Configuration file for Bank Reviews Analysis Project
No environment variables required
"""
import os

# Google Play Store App IDs for Ethiopian Banks
APP_IDS = {
    'CBE': 'com.combanketh.mobilebanking',  # Commercial Bank of Ethiopia
    'Awash': 'com.sc.awashpay',             # Awash Bank
    'Amharabank': 'com.amharabank.Aba_mobile_banking'  # Amhara Bank
}

# Bank Names Mapping
BANK_NAMES = {
    'CBE': 'Commercial Bank of Ethiopia',
    'Awash': 'Awash Bank', 
    'Amharabank': 'Amhara Bank'
}

# Scraping Configuration
SCRAPING_CONFIG = {
    'reviews_per_bank': 400,  # Target 400+ reviews per bank
    'max_retries': 3,
    'lang': 'en',
    'country': 'et'  # Ethiopia
}

# File Paths
DATA_PATHS = {
    'raw': 'data/raw_data',
    'processed': 'data/processed_data',
    'raw_reviews': 'data/raw_data/reviews_raw.csv',
    'processed_reviews': 'data/processed_data/reviews_processed.csv',
    'sentiment_results': 'data/processed_data/reviews_with_sentiment.csv',
    'final_results': 'data/processed_data/reviews_final.csv',
    'themes_results': 'data/processed_data/thematic_analysis.csv',
    'visualizations': 'data/processed_data/visualizations'
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'min_review_length': 5,
    'theme_keywords': {
        'login_issues': ['login', 'sign in', 'password', 'account access', 'authentication', 'verification'],
        'transaction_issues': ['transfer', 'transaction', 'payment', 'send money', 'slow', 'failed', 'pending'],
        'app_performance': ['crash', 'slow', 'freeze', 'bug', 'error', 'not working', 'hang'],
        'ui_ux': ['interface', 'design', 'layout', 'user friendly', 'easy to use', 'complicated', 'navigation'],
        'customer_service': ['support', 'help', 'service', 'response', 'contact', 'assistance'],
        'features': ['feature', 'function', 'option', 'missing', 'should have', 'add', 'include'],
        'security': ['security', 'safe', 'hack', 'privacy', 'protection'],
        'notification': ['notification', 'alert', 'message', 'reminder']
    },
    'sentiment_threshold': 0.6
}