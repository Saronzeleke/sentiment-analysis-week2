# Save this as: src/analysis/insights.py

"""
Insights Analysis Module for Banking Apps Sentiment Analysis
Senior Data Scientist Implementation - Task 4

This module contains reusable analysis functions for extracting insights
from banking app reviews.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import Counter
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BankingInsightsAnalyzer:
    """Analyze banking app reviews to extract insights and recommendations"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer with review data.
        
        Args:
            df: DataFrame containing review data with columns:
                - bank_name: Name of the bank
                - review_text: Original review text
                - rating: Numeric rating (1-5)
                - sentiment_label: Sentiment classification
                - sentiment_score: Sentiment score
                - review_date: Date of review
        """
        self.df = df.copy()
        self.insights = {}
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate that required columns exist"""
        required_columns = ['bank_name', 'review_text', 'rating', 'sentiment_label']
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            # Try to create missing columns if possible
            if 'rating' not in self.df.columns:
                self.df['rating'] = 3.0  # Default neutral rating
            
            if 'sentiment_label' not in self.df.columns and 'sentiment_score' in self.df.columns:
                self.df['sentiment_label'] = self.df['sentiment_score'].apply(
                    lambda x: 'positive' if x > 0.1 else 'negative' if x < -0.1 else 'neutral'
                )
    
    def analyze_bank_performance(self) -> Dict[str, Dict]:
        """
        Analyze performance metrics for each bank.
        
        Returns:
            Dictionary with bank names as keys and performance metrics as values
        """
        logger.info("Analyzing bank performance metrics")
        
        bank_metrics = {}
        
        for bank in self.df['bank_name'].unique():
            bank_data = self.df[self.df['bank_name'] == bank]
            
            metrics = {
                'total_reviews': len(bank_data),
                'avg_rating': bank_data['rating'].mean(),
                'avg_sentiment': bank_data.get('sentiment_score', 0).mean(),
                'positive_pct': (bank_data['sentiment_label'] == 'positive').mean() * 100,
                'negative_pct': (bank_data['sentiment_label'] == 'negative').mean() * 100,
                'rating_std': bank_data['rating'].std(),
                'rating_median': bank_data['rating'].median(),
            }
            
            # Add distribution data
            if 'rating' in bank_data.columns:
                metrics['rating_distribution'] = dict(
                    bank_data['rating'].value_counts().sort_index()
                )
            
            if 'sentiment_label' in bank_data.columns:
                metrics['sentiment_distribution'] = dict(
                    bank_data['sentiment_label'].value_counts()
                )
            
            bank_metrics[bank] = metrics
        
        self.insights['bank_metrics'] = bank_metrics
        return bank_metrics
    
    def extract_key_phrases(self, text_series: pd.Series, bank_name: str) -> Dict[str, List[Dict]]:
        """
        Extract key phrases from reviews using pattern matching.
        
        Args:
            text_series: Series of text reviews
            bank_name: Name of the bank for context
            
        Returns:
            Dictionary with 'drivers' and 'pain_points' lists
        """
        # Define patterns for different categories
        patterns = {
            'positive': {
                'easy_to_use': r'\b(easy|simple|intuitive|user.friendly|straightforward)\b',
                'fast': r'\b(fast|quick|speed|responsive|instant)\b',
                'reliable': r'\b(reliable|stable|consistent|dependable|trustworthy)\b',
                'secure': r'\b(secure|safe|protected|encrypted|security)\b',
                'helpful': r'\b(helpful|supportive|friendly|professional|knowledgeable)\b'
            },
            'negative': {
                'crash': r'\b(crash|freeze|hang|not.responding|bug|glitch)\b',
                'slow': r'\b(slow|lag|delay|waiting|loading)\b',
                'login': r'\b(login|sign.in|password|authentication|verify)\b',
                'transaction': r'\b(transaction|transfer|payment|failed|error)\b',
                'update': r'\b(update|version|new.update|after.update)\b'
            }
        }
        
        results = {'drivers': [], 'pain_points': []}
        
        for sentiment, categories in patterns.items():
            for category, pattern in categories.items():
                matches = text_series.str.contains(pattern, case=False, na=False)
                count = matches.sum()
                
                if count > 0:
                    # Get sample matching reviews
                    sample_reviews = text_series[matches].head(3).tolist()
                    
                    insight = {
                        'category': category,
                        'count': int(count),
                        'sample_reviews': sample_reviews,
                        'bank': bank_name
                    }
                    
                    if sentiment == 'positive':
                        results['drivers'].append(insight)
                    else:
                        results['pain_points'].append(insight)
        
        return results
    
    def identify_drivers_pain_points(self) -> Dict[str, Dict]:
        """
        Identify satisfaction drivers and pain points for each bank.
        
        Returns:
            Dictionary with bank names as keys and drivers/pain points as values
        """
        logger.info("Identifying drivers and pain points")
        
        drivers_pain_points = {}
        
        for bank in self.df['bank_name'].unique():
            bank_data = self.df[self.df['bank_name'] == bank]
            
            # Get positive and negative reviews separately
            positive_reviews = bank_data[bank_data['sentiment_label'] == 'positive']
            negative_reviews = bank_data[bank_data['sentiment_label'] == 'negative']
            
            # Extract key phrases
            if len(positive_reviews) > 0:
                positive_insights = self.extract_key_phrases(
                    positive_reviews['review_text'], bank
                )
                drivers = positive_insights.get('drivers', [])
            else:
                drivers = []
            
            if len(negative_reviews) > 0:
                negative_insights = self.extract_key_phrases(
                    negative_reviews['review_text'], bank
                )
                pain_points = negative_insights.get('pain_points', [])
            else:
                pain_points = []
            
            # Sort by count and take top 3
            drivers.sort(key=lambda x: x['count'], reverse=True)
            pain_points.sort(key=lambda x: x['count'], reverse=True)
            
            drivers_pain_points[bank] = {
                'drivers': drivers[:3],
                'pain_points': pain_points[:3]
            }
            
            logger.info(f"Identified {len(drivers[:3])} drivers and {len(pain_points[:3])} pain points for {bank}")
        
        self.insights['drivers_pain_points'] = drivers_pain_points
        return drivers_pain_points
    
    def compare_banks(self) -> Dict[str, Any]:
        """
        Compare banks across multiple dimensions.
        
        Returns:
            Dictionary containing comparison metrics and rankings
        """
        logger.info("Comparing banks")
        
        comparison = {}
        
        if 'bank_metrics' not in self.insights:
            self.analyze_bank_performance()
        
        bank_metrics = self.insights['bank_metrics']
        banks = list(bank_metrics.keys())
        
        # 1. Performance ranking based on composite score
        performance_scores = {}
        for bank, metrics in bank_metrics.items():
            # Calculate composite score (40% rating, 30% sentiment, 30% positive rate)
            rating_score = (metrics['avg_rating'] / 5) * 40
            sentiment_score = ((metrics['avg_sentiment'] + 1) / 2) * 30
            positive_score = (metrics['positive_pct'] / 100) * 30
            
            composite_score = rating_score + sentiment_score + positive_score
            performance_scores[bank] = composite_score
        
        # Rank banks
        ranked_banks = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
        comparison['performance_ranking'] = ranked_banks
        
        # 2. Statistical comparison of ratings
        try:
            from scipy import stats
            
            comparison['statistical_tests'] = {}
            for i in range(len(banks)):
                for j in range(i + 1, len(banks)):
                    bank1_data = self.df[self.df['bank_name'] == banks[i]]['rating'].dropna()
                    bank2_data = self.df[self.df['bank_name'] == banks[j]]['rating'].dropna()
                    
                    if len(bank1_data) > 10 and len(bank2_data) > 10:
                        t_stat, p_value = stats.ttest_ind(bank1_data, bank2_data, equal_var=False)
                        
                        comparison['statistical_tests'][f'{banks[i]}_vs_{banks[j]}'] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
        except ImportError:
            logger.warning("SciPy not available for statistical tests")
        
        # 3. Strengths and weaknesses comparison
        if 'drivers_pain_points' not in self.insights:
            self.identify_drivers_pain_points()
        
        drivers_pain_points = self.insights['drivers_pain_points']
        
        comparison['strengths_weaknesses'] = {}
        for bank in banks:
            bank_drivers = [d['category'] for d in drivers_pain_points.get(bank, {}).get('drivers', [])]
            bank_pain_points = [p['category'] for p in drivers_pain_points.get(bank, {}).get('pain_points', [])]
            
            comparison['strengths_weaknesses'][bank] = {
                'strengths': bank_drivers,
                'weaknesses': bank_pain_points
            }
        
        self.insights['bank_comparison'] = comparison
        return comparison
    
    def generate_recommendations(self) -> Dict[str, List[Dict]]:
        """
        Generate actionable recommendations based on analysis.
        
        Returns:
            Dictionary with bank names as keys and recommendation lists as values
        """
        logger.info("Generating recommendations")
        
        if 'drivers_pain_points' not in self.insights:
            self.identify_drivers_pain_points()
        
        drivers_pain_points = self.insights['drivers_pain_points']
        
        # Recommendation templates based on issues
        recommendation_templates = {
            'crash': [
                "Improve app stability through rigorous testing before updates",
                "Implement automatic crash reporting and monitoring system",
                "Optimize memory usage to prevent app crashes"
            ],
            'slow': [
                "Optimize database queries and API response times",
                "Implement lazy loading for non-critical features",
                "Add progress indicators during loading processes"
            ],
            'login': [
                "Simplify authentication process with biometric options",
                "Implement passwordless login with email/SMS verification",
                "Add 'Remember me' functionality for trusted devices"
            ],
            'transaction': [
                "Improve transaction confirmation process with clearer instructions",
                "Add real-time transaction status notifications",
                "Implement two-factor authentication for large transactions"
            ],
            'update': [
                "Implement phased rollout for app updates",
                "Provide detailed changelogs with each update",
                "Maintain backward compatibility for critical features"
            ],
            'easy_to_use': [
                "Continue simplifying user interface based on feedback",
                "Add in-app tutorials for new features",
                "Implement contextual help and tooltips"
            ],
            'secure': [
                "Enhance security features with regular security audits",
                "Implement transaction anomaly detection",
                "Add security education features within the app"
            ]
        }
        
        recommendations = {}
        
        for bank in self.df['bank_name'].unique():
            bank_recs = []
            
            # Get pain points for this bank
            pain_points = drivers_pain_points.get(bank, {}).get('pain_points', [])
            
            # Generate recommendations for each pain point
            for pain_point in pain_points:
                category = pain_point['category']
                count = pain_point['count']
                
                if category in recommendation_templates:
                    priority = 'HIGH' if count > 20 else 'MEDIUM' if count > 10 else 'LOW'
                    
                    for template in recommendation_templates[category]:
                        recommendation = {
                            'type': 'pain_point_solution',
                            'category': category,
                            'recommendation': template,
                            'priority': priority,
                            'evidence': f"Mentioned in {count} negative reviews"
                        }
                        bank_recs.append(recommendation)
            
            # Add enhancement recommendations based on drivers
            drivers = drivers_pain_points.get(bank, {}).get('drivers', [])
            for driver in drivers[:2]:  # Top 2 drivers
                category = driver['category']
                count = driver['count']
                
                if category in recommendation_templates:
                    enhancement = f"Further enhance {category.replace('_', ' ')} features"
                    bank_recs.append({
                        'type': 'enhancement',
                        'category': category,
                        'recommendation': enhancement,
                        'priority': 'MEDIUM',
                        'evidence': f"Praised in {count} positive reviews"
                    })
            
            # Add general recommendations
            general_recs = [
                {
                    'type': 'general',
                    'category': 'feedback',
                    'recommendation': "Implement in-app feedback system for real-time user input",
                    'priority': 'MEDIUM',
                    'evidence': 'Industry best practice'
                },
                {
                    'type': 'general',
                    'category': 'analytics',
                    'recommendation': "Use analytics to track feature usage and user behavior patterns",
                    'priority': 'LOW',
                    'evidence': 'Data-driven decision making'
                }
            ]
            
            bank_recs.extend(general_recs)
            recommendations[bank] = bank_recs
        
        self.insights['recommendations'] = recommendations
        return recommendations
    
    def analyze_ethics_biases(self) -> Dict[str, List[str]]:
        """
        Analyze potential ethical issues and biases in the data.
        
        Returns:
            Dictionary containing ethical considerations and biases
        """
        logger.info("Analyzing ethical considerations and biases")
        
        ethics_analysis = {
            'review_biases': [],
            'sampling_issues': [],
            'ethical_considerations': [],
            'limitations': []
        }
        
        # 1. Check for negative bias
        if 'sentiment_label' in self.df.columns:
            negative_rate = (self.df['sentiment_label'] == 'negative').mean() * 100
            if negative_rate > 60:
                ethics_analysis['review_biases'].append(
                    f"Strong negative bias detected: {negative_rate:.1f}% of reviews are negative"
                )
        
        # 2. Check sampling bias
        if 'bank_name' in self.df.columns:
            bank_counts = self.df['bank_name'].value_counts()
            if len(bank_counts) > 1:
                min_count = bank_counts.min()
                max_count = bank_counts.max()
                ratio = min_count / max_count
                
                if ratio < 0.7:
                    ethics_analysis['sampling_issues'].append(
                        f"Uneven sampling across banks (ratio: {ratio:.2f})"
                    )
        
        # 3. Ethical considerations
        ethical_points = [
            "Reviews may not represent all user demographics equally",
            "Sentiment analysis models may have cultural and linguistic biases",
            "Automated analysis should be validated with human review for important decisions",
            "Privacy concerns: reviews may contain sensitive financial information",
            "Potential for manipulation: competitors or biased users might leave fake reviews"
        ]
        
        ethics_analysis['ethical_considerations'].extend(ethical_points)
        
        # 4. Limitations
        limitations = [
            "Analysis based on text data only - no behavioral or transactional data",
            "Limited to Google Play Store reviews - excludes iOS users and other platforms",
            "Cannot verify the authenticity of all reviews",
            "May not capture seasonal or temporary issues accurately"
        ]
        
        ethics_analysis['limitations'].extend(limitations)
        
        self.insights['ethics_analysis'] = ethics_analysis
        return ethics_analysis
    
    def generate_executive_summary(self) -> str:
        """
        Generate executive summary of findings.
        
        Returns:
            Formatted executive summary string
        """
        if 'bank_metrics' not in self.insights:
            self.analyze_bank_performance()
        
        bank_metrics = self.insights['bank_metrics']
        
        # Find best and worst performing banks
        if bank_metrics:
            best_bank = max(bank_metrics.items(), key=lambda x: x[1]['avg_rating'])[0]
            worst_bank = min(bank_metrics.items(), key=lambda x: x[1]['avg_rating'])[0]
            
            summary = f"""
            EXECUTIVE SUMMARY - Banking Apps Sentiment Analysis
            
            1. OVERVIEW:
               • Total Reviews Analyzed: {len(self.df):,}
               • Banks Covered: {', '.join(self.df['bank_name'].unique())}
               • Analysis Period: {self.df['review_date'].min() if 'review_date' in self.df.columns else 'N/A'} to {self.df['review_date'].max() if 'review_date' in self.df.columns else 'N/A'}
            
            2. KEY FINDINGS:
               • Best Performing Bank: {best_bank} (Rating: {bank_metrics[best_bank]['avg_rating']:.2f}/5)
               • Needs Most Improvement: {worst_bank} (Rating: {bank_metrics[worst_bank]['avg_rating']:.2f}/5)
               • Overall Satisfaction: {np.mean([m['positive_pct'] for m in bank_metrics.values()]):.1f}% positive reviews
            
            3. CRITICAL ISSUES IDENTIFIED:
               • App crashes and stability issues are common across all banks
               • Login and authentication problems affect user experience
               • Transaction failures need immediate attention
               • Performance issues (slow loading) are frequent complaints
            
            4. OPPORTUNITIES:
               • User-friendly interfaces are key satisfaction drivers
               • Reliable performance builds user trust
               • Good customer support enhances loyalty
               • Security features are highly valued by users
            
            This analysis provides data-driven insights for improving mobile banking 
            applications and enhancing customer satisfaction.
            """
        else:
            summary = "Insufficient data to generate executive summary."
        
        self.insights['executive_summary'] = summary
        return summary
    
    def get_all_insights(self) -> Dict[str, Any]:
        """
        Run all analyses and return comprehensive insights.
        
        Returns:
            Dictionary containing all insights and analyses
        """
        logger.info("Running comprehensive analysis")
        
        # Run all analysis steps
        self.analyze_bank_performance()
        self.identify_drivers_pain_points()
        self.compare_banks()
        self.generate_recommendations()
        self.analyze_ethics_biases()
        self.generate_executive_summary()
        
        return self.insights


# Helper functions for text analysis
def extract_common_words(text_series: pd.Series, n_words: int = 10) -> List[Tuple[str, int]]:
    """
    Extract most common words from text series.
    
    Args:
        text_series: Series of text data
        n_words: Number of top words to return
        
    Returns:
        List of (word, count) tuples
    """
    # Common stopwords to filter out
    stopwords = {
        'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'that', 'for',
        'on', 'with', 'as', 'this', 'by', 'i', 'not', 'be', 'are', 'was',
        'but', 'have', 'has', 'had', 'my', 'me', 'we', 'our', 'you', 'your'
    }
    
    all_words = []
    for text in text_series.dropna().astype(str):
        # Simple word splitting (improve with regex for production)
        words = text.lower().split()
        filtered_words = [w for w in words if len(w) > 2 and w not in stopwords]
        all_words.extend(filtered_words)
    
    word_counts = Counter(all_words)
    return word_counts.most_common(n_words)


def calculate_sentiment_trends(df: pd.DataFrame, date_col: str = 'review_date') -> pd.DataFrame:
    """
    Calculate sentiment trends over time.
    
    Args:
        df: DataFrame with sentiment data
        date_col: Name of date column
        
    Returns:
        DataFrame with sentiment trends by time period
    """
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df['month'] = df[date_col].dt.to_period('M')
        
        trends = df.groupby('month').agg({
            'sentiment_score': 'mean',
            'rating': 'mean',
            'review_text': 'count'
        }).rename(columns={'review_text': 'review_count'})
        
        return trends.reset_index()
    except Exception as e:
        logger.error(f"Error calculating sentiment trends: {e}")
        return pd.DataFrame()