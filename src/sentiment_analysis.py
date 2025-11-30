"""
Sentiment Analysis Module using multiple NLP approaches
"""

import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
import logging
from typing import Dict, Tuple, List
from tqdm import tqdm
from config.banks_config import SENTIMENT_THRESHOLDS

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Perform sentiment analysis using multiple models"""
    
    def __init__(self, method: str = "distilbert"):
        self.method = method
        self.models = {}
        self.setup_models()
    
    def setup_models(self):
        """Initialize sentiment analysis models"""
        try:
            if self.method == "distilbert":
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
                    truncation=True,
                    max_length=512
                )
            elif self.method == "vader":
                self.sentiment_pipeline = SentimentIntensityAnalyzer()
            elif self.method == "textblob":
                # TextBlob doesn't need initialization
                pass
                
            logger.info(f"Initialized {self.method} sentiment analyzer")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {str(e)}")
            raise
    
    def analyze_sentiment_distilbert(self, text: str) -> Dict:
        """Analyze sentiment using DistilBERT"""
        try:
            result = self.sentiment_pipeline(text[:1000])[0]  # Truncate for model limits
            score = result['score']
            label = result['label'].lower()
            
            # Convert to our threshold system
            if label == 'positive':
                sentiment_score = score
            else:  # negative
                sentiment_score = 1 - score
                
            return {
                'sentiment_label': self._classify_sentiment(sentiment_score),
                'sentiment_score': sentiment_score,
                'confidence': score
            }
        except Exception as e:
            logger.warning(f"Error in DistilBERT analysis: {str(e)}")
            return {'sentiment_label': 'neutral', 'sentiment_score': 0.5, 'confidence': 0.0}
    
    def analyze_sentiment_vader(self, text: str) -> Dict:
        """Analyze sentiment using VADER"""
        try:
            scores = self.sentiment_pipeline.polarity_scores(text)
            compound_score = (scores['compound'] + 1) / 2  # Convert to 0-1 scale
            
            return {
                'sentiment_label': self._classify_sentiment(compound_score),
                'sentiment_score': compound_score,
                'confidence': abs(scores['compound'])
            }
        except Exception as e:
            logger.warning(f"Error in VADER analysis: {str(e)}")
            return {'sentiment_label': 'neutral', 'sentiment_score': 0.5, 'confidence': 0.0}
    
    def analyze_sentiment_textblob(self, text: str) -> Dict:
        """Analyze sentiment using TextBlob"""
        try:
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            sentiment_score = (polarity + 1) / 2  # Convert to 0-1 scale
            
            return {
                'sentiment_label': self._classify_sentiment(sentiment_score),
                'sentiment_score': sentiment_score,
                'confidence': abs(polarity)
            }
        except Exception as e:
            logger.warning(f"Error in TextBlob analysis: {str(e)}")
            return {'sentiment_label': 'neutral', 'sentiment_score': 0.5, 'confidence': 0.0}
    
    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment based on thresholds"""
        if score >= SENTIMENT_THRESHOLDS['positive']:
            return 'positive'
        elif score <= SENTIMENT_THRESHOLDS['negative']:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_text(self, text: str) -> Dict:
        """Analyze sentiment for a single text"""
        if self.method == "distilbert":
            return self.analyze_sentiment_distilbert(text)
        elif self.method == "vader":
            return self.analyze_sentiment_vader(text)
        elif self.method == "textblob":
            return self.analyze_sentiment_textblob(text)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'review') -> pd.DataFrame:
        """Analyze sentiment for entire DataFrame"""
        logger.info(f"Starting sentiment analysis using {self.method}")
        
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing sentiments"):
            text = row[text_column]
            sentiment_result = self.analyze_text(str(text))
            
            result_row = {
                'review_id': row['review_id'],
                'review_text': text,
                'sentiment_label': sentiment_result['sentiment_label'],
                'sentiment_score': sentiment_result['sentiment_score'],
                'confidence': sentiment_result['confidence'],
                'rating': row['rating'],
                'bank': row['bank'],
                'date': row['date']
            }
            results.append(result_row)
        
        results_df = pd.DataFrame(results)
        logger.info(f"Sentiment analysis completed for {len(results_df)} reviews")
        
        return results_df
    
    def calculate_sentiment_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate sentiment metrics by bank and rating"""
        metrics = {}
        
        # Overall sentiment distribution
        overall_sentiment = df['sentiment_label'].value_counts(normalize=True).to_dict()
        metrics['overall_sentiment_distribution'] = overall_sentiment
        
        # Sentiment by bank
        bank_sentiment = df.groupby(['bank', 'sentiment_label']).size().unstack(fill_value=0)
        bank_sentiment_pct = bank_sentiment.div(bank_sentiment.sum(axis=1), axis=0)
        metrics['sentiment_by_bank'] = bank_sentiment_pct.to_dict()
        
        # Sentiment by rating
        rating_sentiment = df.groupby(['rating', 'sentiment_label']).size().unstack(fill_value=0)
        rating_sentiment_pct = rating_sentiment.div(rating_sentiment.sum(axis=1), axis=0)
        metrics['sentiment_by_rating'] = rating_sentiment_pct.to_dict()
        
        # Average sentiment scores
        avg_sentiment_by_bank = df.groupby('bank')['sentiment_score'].mean().to_dict()
        metrics['average_sentiment_by_bank'] = avg_sentiment_by_bank
        
        logger.info("Sentiment metrics calculated")
        return metrics

def compare_sentiment_methods(df: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
    """Compare different sentiment analysis methods on a sample"""
    sample_df = df.sample(min(sample_size, len(df)))
    
    methods = ['distilbert', 'vader', 'textblob']
    comparison_results = []
    
    for method in methods:
        analyzer = SentimentAnalyzer(method=method)
        results = analyzer.analyze_dataframe(sample_df)
        results['method'] = method
        comparison_results.append(results)
    
    comparison_df = pd.concat(comparison_results, ignore_index=True)
    return comparison_df

if __name__ == "__main__":
    # Load processed data
    processed_df = pd.read_csv("data/processed/processed_reviews.csv")
    
    # Analyze with DistilBERT (primary method)
    analyzer = SentimentAnalyzer(method="distilbert")
    sentiment_results = analyzer.analyze_dataframe(processed_df)
    
    # Calculate metrics
    metrics = analyzer.calculate_sentiment_metrics(sentiment_results)
    
    # Save results
    sentiment_results.to_csv("data/results/sentiment_analysis.csv", index=False)
    
    logger.info(f"Sentiment analysis completed. Results saved.")
    logger.info(f"Coverage: {len(sentiment_results)/len(processed_df)*100:.1f}%")