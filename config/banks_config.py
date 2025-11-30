"""
Configuration for Ethiopian banks to analyze
"""

BANKS_CONFIG = {
    "cbe": {
        "app_id": "com.cbe.telebirr",
        "name": "Commercial Bank of Ethiopia",
        "country": "et",
        "language": "en"
    },
    "awash": {
        "app_id": "com.awash.bank",
        "name": "Awash Bank",
        "country": "et", 
        "language": "en"
    },
    "dashen": {
        "app_id": "com.dashen.sc",
        "name": "Dashen Bank",
        "country": "et",
        "language": "en"
    }
}

# Analysis parameters
SENTIMENT_THRESHOLDS = {
    'positive': 0.6,
    'negative': 0.4,
    'neutral': (0.4, 0.6)
}

THEME_CATEGORIES = [
    'User Interface & Experience',
    'Transaction Performance', 
    'Account Access & Security',
    'Customer Support',
    'App Reliability & Bugs',
    'Feature Requests'
]