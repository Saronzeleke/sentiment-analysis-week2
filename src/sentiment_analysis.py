"""
Sentiment and Thematic Analysis
Task 2: Sentiment Analysis and Theme Identification

This script performs:
- Sentiment analysis using distilBERT
- Keyword extraction using TF-IDF
- Theme clustering and identification
- Comparison of multiple sentiment analysis methods
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from tqdm import tqdm
from config.banks_config import DATA_PATHS, ANALYSIS_CONFIG

# Optional: VADER (will be used if installed)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


class SentimentAnalyzer:
    """Sentiment analysis and thematic analysis class"""

    def __init__(self):
        self.df = None
        self.sentiment_pipeline = None
        self.nlp = None
        self.themes = {}

    def load_data(self):
        """Load processed reviews data"""
        print("Loading processed data...")
        try:
            self.df = pd.read_csv(DATA_PATHS['processed_reviews'])
            print(f"Loaded {len(self.df)} reviews for analysis")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load data: {str(e)}")
            return False

    def initialize_models(self):
        """Initialize NLP models"""
        print("Initializing NLP models...")
        
        # Initialize sentiment analysis pipeline
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
            )
            print("✓ distilBERT sentiment model loaded")
        except Exception as e:
            print(f"WARNING: Failed to load distilBERT: {str(e)}")
            self.sentiment_pipeline = None

        # Initialize spaCy for text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✓ spaCy model loaded")
        except Exception as e:
            print(f"WARNING: Failed to load spaCy: {str(e)}")
            self.nlp = None

    def analyze_sentiment(self):
        """Perform sentiment analysis on reviews"""
        print("\n[1/4] Performing sentiment analysis...")

        sentiments = []
        scores = []

        for text in tqdm(self.df['review_text'], desc="Analyzing sentiment"):
            try:
                if self.sentiment_pipeline:
                    result = self.sentiment_pipeline(text[:512])[0]
                    sentiment = result['label']
                    score = result['score']
                else:
                    sentiment, score = self._basic_sentiment_analysis(text)
                
                sentiments.append(sentiment)
                scores.append(score)
                
            except Exception as e:
                print(f"Error in sentiment analysis: {str(e)}")
                sentiments.append('NEUTRAL')
                scores.append(0.5)

        self.df['sentiment_label'] = sentiments
        self.df['sentiment_score'] = scores

        self.df['sentiment'] = self.df['sentiment_label'].map({
            'POSITIVE': 'positive',
            'NEGATIVE': 'negative',
            'LABEL_0': 'negative',
            'LABEL_1': 'positive'
        }).fillna(self.df['sentiment_label'].str.lower())

        print("Sentiment distribution:")
        print(self.df['sentiment'].value_counts())

    def _basic_sentiment_analysis(self, text):
        """Basic sentiment analysis as fallback"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'easy', 'nice', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'slow', 'crash', 'error', 'problem']

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return 'POSITIVE', 0.7
        elif negative_count > positive_count:
            return 'NEGATIVE', 0.7
        else:
            return 'NEUTRAL', 0.5

    def extract_keywords(self):
        """Extract keywords using TF-IDF"""
        print("\n[2/4] Extracting keywords...")

        for bank in self.df['bank_name'].unique():
            print(f"\nAnalyzing keywords for {bank}...")
            bank_reviews = self.df[self.df['bank_name'] == bank]
            
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(bank_reviews['review_text'])
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
                top_indices = tfidf_scores.argsort()[-20:][::-1]
                top_keywords = [feature_names[i] for i in top_indices]
                
                print(f"Top keywords for {bank}: {top_keywords[:10]}")
                
                self.themes[bank] = {
                    'keywords': top_keywords,
                    'reviews': bank_reviews
                }
                
            except Exception as e:
                print(f"Error extracting keywords for {bank}: {str(e)}")

    def identify_themes(self):
        """Identify themes from keywords"""
        print("\n[3/4] Identifying themes...")

        theme_mapping = ANALYSIS_CONFIG['theme_keywords']
        
        for bank, bank_data in self.themes.items():
            theme_counts = {theme: 0 for theme in theme_mapping.keys()}
            theme_examples = {theme: [] for theme in theme_mapping.keys()}
            
            for _, review in bank_data['reviews'].iterrows():
                review_text = review['review_text'].lower()
                for theme, keywords in theme_mapping.items():
                    for keyword in keywords:
                        if keyword in review_text:
                            theme_counts[theme] += 1
                            if len(theme_examples[theme]) < 3:
                                theme_examples[theme].append(review_text[:100] + "...")
                            break
            
            top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            print(f"Top themes for {bank}:")
            for theme, count in top_themes:
                if count > 0:
                    print(f"  {theme}: {count} occurrences")
                    if theme_examples[theme]:
                        print(f"    Example: {theme_examples[theme][0]}")
            
            self.themes[bank]['top_themes'] = top_themes
            self.themes[bank]['theme_examples'] = theme_examples

    def assign_themes_to_reviews(self):
        """Assign themes to individual reviews"""
        print("\n[4/4] Assigning themes to reviews...")

        theme_mapping = ANALYSIS_CONFIG['theme_keywords']
        assigned_themes = []

        for _, review in self.df.iterrows():
            review_text = review['review_text'].lower()
            review_themes = []

            for theme, keywords in theme_mapping.items():
                for keyword in keywords:
                    if keyword in review_text:
                        review_themes.append(theme)
                        break

            assigned_themes.append(', '.join(review_themes) if review_themes else 'Other')

        self.df['assigned_themes'] = assigned_themes

        print("Theme assignment completed:")
        print(self.df['assigned_themes'].value_counts().head(10))

    def generate_insights(self):
        """Generate actionable insights for each bank"""
        print("\n" + "=" * 60)
        print("ACTIONABLE INSIGHTS")
        print("=" * 60)

        for bank in self.df['bank_name'].unique():
            bank_data = self.df[self.df['bank_name'] == bank]
            total_reviews = len(bank_data)
            
            print(f"\n📊 {bank} - Analysis Summary")
            print(f"   Total Reviews: {total_reviews}")
            
            sentiment_counts = bank_data['sentiment'].value_counts()
            positive_pct = (sentiment_counts.get('positive', 0) / total_reviews) * 100
            negative_pct = (sentiment_counts.get('negative', 0) / total_reviews) * 100
            
            print(f"   Positive Sentiment: {positive_pct:.1f}%")
            print(f"   Negative Sentiment: {negative_pct:.1f}%")
            
            avg_rating = bank_data['rating'].mean()
            print(f"   Average Rating: {avg_rating:.1f}/5")
            
            if bank in self.themes:
                print(f"\n   🎯 Key Themes (Drivers & Pain Points):")
                for theme, count in self.themes[bank]['top_themes'][:4]:
                    if count > 0:
                        theme_pct = (count / total_reviews) * 100
                        theme_name = theme.replace('_', ' ').title()
                        print(f"      • {theme_name}: {count} reviews ({theme_pct:.1f}%)")

    def save_results(self):
        """Save analysis results"""
        print("\nSaving analysis results...")

        try:
            os.makedirs(os.path.dirname(DATA_PATHS['sentiment_results']), exist_ok=True)
            
            self.df.to_csv(DATA_PATHS['sentiment_results'], index=False)
            print(f"✓ Sentiment analysis saved to: {DATA_PATHS['sentiment_results']}")

            theme_summary = []
            for bank, bank_data in self.themes.items():
                for theme, count in bank_data.get('top_themes', []):
                    theme_summary.append({
                        'bank': bank,
                        'theme': theme,
                        'occurrences': count,
                        'percentage': (count / len(self.df[self.df['bank_name'] == bank])) * 100
                    })

            theme_df = pd.DataFrame(theme_summary)
            theme_df.to_csv(DATA_PATHS['themes_results'], index=False)
            print(f"✓ Thematic analysis saved to: {DATA_PATHS['themes_results']}")

            return True

        except Exception as e:
            print(f"ERROR: Failed to save results: {str(e)}")
            return False

    def analyze(self):
        """Run complete analysis pipeline"""
        print("=" * 60)
        print("STARTING SENTIMENT & THEMATIC ANALYSIS")
        print("=" * 60)

        if not self.load_data():
            return False

        self.initialize_models()
        self.analyze_sentiment()
        self.extract_keywords()
        self.identify_themes()
        self.assign_themes_to_reviews()
        self.generate_insights()

        if self.save_results():
            print("\n✓ Analysis completed successfully!")
            return True
        else:
            print("\n✗ Analysis completed with errors!")
            return False

    def calculate_sentiment_metrics(self, df=None):
        """Calculate sentiment metrics"""
        df = df if df is not None else self.df
        if df is None:
            return {}
        
        total = len(df)
        counts = df['sentiment'].value_counts()
        return {
            'total_reviews': total,
            'positive_pct': counts.get('positive', 0) / total * 100,
            'negative_pct': counts.get('negative', 0) / total * 100,
            'neutral_pct': counts.get('neutral', 0) / total * 100,
        }


def _basic_sentiment_analysis(text):
    """Standalone basic sentiment analysis for comparison"""
    positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'easy', 'nice', 'perfect']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'slow', 'crash', 'error', 'problem']
    text_lower = text.lower()
    pos = sum(1 for w in positive_words if w in text_lower)
    neg = sum(1 for w in negative_words if w in text_lower)
    if pos > neg:
        return 'positive', 0.7
    elif neg > pos:
        return 'negative', 0.7
    else:
        return 'neutral', 0.5


def compare_sentiment_methods(df, sample_size=200):
    """
    Compare multiple sentiment analysis methods on a sample of reviews.
    
    Returns a DataFrame with columns:
    - review_id
    - review_text
    - method (e.g., 'distilBERT', 'Rule-Based', 'VADER')
    - sentiment_label
    - confidence_score
    """
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42).copy()
    results = []

    # 1. distilBERT
    try:
        bert_pipe = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        for _, row in sample_df.iterrows():
            text = str(row['review_text'])[:512]
            if not text.strip():
                continue
            result = bert_pipe(text)[0]
            results.append({
                'review_id': row.get('review_id', ''),
                'review_text': text,
                'method': 'distilBERT',
                'sentiment_label': result['label'].lower().replace('label_0', 'negative').replace('label_1', 'positive'),
                'confidence_score': result['score']
            })
    except Exception as e:
        print(f"⚠️ distilBERT comparison failed: {e}")

    # 2. Rule-Based
    for _, row in sample_df.iterrows():
        text = str(row['review_text'])
        if not text.strip():
            continue
        label, score = _basic_sentiment_analysis(text)
        results.append({
            'review_id': row.get('review_id', ''),
            'review_text': text,
            'method': 'Rule-Based',
            'sentiment_label': label,
            'confidence_score': score
        })

    # 3. VADER (if available)
    if VADER_AVAILABLE:
        vader = SentimentIntensityAnalyzer()
        for _, row in sample_df.iterrows():
            text = str(row['review_text'])
            if not text.strip():
                continue
            scores = vader.polarity_scores(text)
            compound = scores['compound']
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            results.append({
                'review_id': row.get('review_id', ''),
                'review_text': text,
                'method': 'VADER',
                'sentiment_label': label,
                'confidence_score': abs(compound)
            })
    else:
        print("⚠️ VADER not installed. Skipping.")

    return pd.DataFrame(results)


def main():
    """Main execution function"""
    analyzer = SentimentAnalyzer()
    success = analyzer.analyze()
    return success


if __name__ == "__main__":
    main()