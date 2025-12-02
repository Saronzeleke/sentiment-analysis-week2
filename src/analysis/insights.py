import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from collections import Counter
import re
from src.database.database_connection import DatabaseConnection

logger = logging.getLogger(__name__)

class BankingInsightsAnalyzer:
    """Analyze banking app reviews to extract insights and recommendations"""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
        self.insights = {
            'cbe': {'drivers': [], 'pain_points': [], 'recommendations': []},
            'boa': {'drivers': [], 'pain_points': [], 'recommendations': []},
            'dashen': {'drivers': [], 'pain_points': [], 'recommendations': []}
        }
    
    def load_review_data(self) -> pd.DataFrame:
        """Load review data from database"""
        query = """
        SELECT 
            b.bank_name,
            r.review_text,
            r.cleaned_text,
            r.rating,
            r.sentiment_label,
            r.sentiment_score,
            r.keywords,
            r.theme,
            r.review_date
        FROM reviews r
        JOIN banks b ON r.bank_id = b.bank_id
        WHERE r.review_text IS NOT NULL
        AND LENGTH(r.review_text) > 10
        """
        
        try:
            results = self.db.execute_query(query)
            df = pd.DataFrame(results)
            logger.info(f"Loaded {len(df)} reviews for analysis")
            return df
        except Exception as e:
            logger.error(f"Failed to load review data: {e}")
            return pd.DataFrame()
    
    def extract_key_phrases(self, text_series: pd.Series, bank_name: str) -> Dict[str, List[str]]:
        """Extract key phrases from reviews using pattern matching"""
        
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
                    
                    if sentiment == 'positive':
                        results['drivers'].append({
                            'category': category,
                            'count': count,
                            'sample_reviews': sample_reviews,
                            'bank': bank_name
                        })
                    else:
                        results['pain_points'].append({
                            'category': category,
                            'count': count,
                            'sample_reviews': sample_reviews,
                            'bank': bank_name
                        })
        
        return results
    
    def analyze_sentiment_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment trends over time"""
        df['review_date'] = pd.to_datetime(df['review_date'])
        df['month'] = df['review_date'].dt.to_period('M')
        
        trends = df.groupby(['bank_name', 'month', 'sentiment_label']).size().unstack(fill_value=0)
        trends['total'] = trends.sum(axis=1)
        trends['positive_rate'] = trends.get('positive', 0) / trends['total'] * 100
        trends['negative_rate'] = trends.get('negative', 0) / trends['total'] * 100
        
        return trends.reset_index()
    
    def compare_banks(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Compare banks across multiple metrics"""
        
        comparison = {}
        
        for bank in df['bank_name'].unique():
            bank_df = df[df['bank_name'] == bank]
            
            comparison[bank] = {
                'total_reviews': len(bank_df),
                'avg_rating': bank_df['rating'].mean(),
                'avg_sentiment': bank_df['sentiment_score'].mean(),
                'positive_pct': (bank_df['sentiment_label'] == 'positive').mean() * 100,
                'negative_pct': (bank_df['sentiment_label'] == 'negative').mean() * 100,
                'rating_distribution': bank_df['rating'].value_counts().sort_index().to_dict(),
                'common_themes': bank_df['theme'].value_counts().head(5).to_dict()
            }
        
        return comparison
    
    def identify_drivers_pain_points(self, df: pd.DataFrame) -> Dict:
        """Identify satisfaction drivers and pain points for each bank"""
        
        drivers_pain_points = {}
        
        for bank in df['bank_name'].unique():
            bank_df = df[df['bank_name'] == bank]
            
            # Get positive and negative reviews separately
            positive_reviews = bank_df[bank_df['sentiment_label'] == 'positive']
            negative_reviews = bank_df[bank_df['sentiment_label'] == 'negative']
            
            # Extract key phrases
            positive_insights = self.extract_key_phrases(
                positive_reviews['review_text'], bank
            ) if len(positive_reviews) > 0 else {'drivers': []}
            
            negative_insights = self.extract_key_phrases(
                negative_reviews['review_text'], bank
            ) if len(negative_reviews) > 0 else {'pain_points': []}
            
            # Aggregate results
            drivers = sorted(
                positive_insights.get('drivers', []),
                key=lambda x: x['count'],
                reverse=True
            )[:3]  # Top 3 drivers
            
            pain_points = sorted(
                negative_insights.get('pain_points', []),
                key=lambda x: x['count'],
                reverse=True
            )[:3]  # Top 3 pain points
            
            drivers_pain_points[bank] = {
                'drivers': drivers,
                'pain_points': pain_points
            }
            
            logger.info(f"Identified {len(drivers)} drivers and {len(pain_points)} pain points for {bank}")
        
        return drivers_pain_points
    
    def generate_recommendations(self, drivers_pain_points: Dict) -> Dict:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = {}
        
        # Mapping of issues to recommendations
        solution_mapping = {
            'crash': [
                "Improve app stability through rigorous testing",
                "Implement better error handling and crash reporting",
                "Optimize memory usage and resource management"
            ],
            'slow': [
                "Optimize database queries and API response times",
                "Implement lazy loading for non-critical features",
                "Add progress indicators during loading"
            ],
            'login': [
                "Simplify authentication process",
                "Add biometric login options (fingerprint, face ID)",
                "Implement passwordless authentication"
            ],
            'transaction': [
                "Improve transaction confirmation process",
                "Add real-time transaction status updates",
                "Implement two-factor authentication for large transactions"
            ],
            'update': [
                "Implement phased rollout for updates",
                "Provide detailed changelogs with each update",
                "Maintain backward compatibility for critical features"
            ],
            'easy_to_use': [
                "Continue simplifying the user interface",
                "Add in-app tutorials for new users",
                "Implement contextual help and tooltips"
            ],
            'secure': [
                "Enhance security features with regular audits",
                "Implement transaction anomaly detection",
                "Add security education for users"
            ]
        }
        
        for bank, insights in drivers_pain_points.items():
            bank_recs = []
            
            # Recommendations based on pain points
            for pain_point in insights['pain_points']:
                category = pain_point['category']
                if category in solution_mapping:
                    for solution in solution_mapping[category]:
                        bank_recs.append({
                            'type': 'improvement',
                            'category': category,
                            'recommendation': solution,
                            'priority': 'high' if pain_point['count'] > 20 else 'medium'
                        })
            
            # Enhancements based on drivers
            for driver in insights['drivers']:
                category = driver['category']
                if category in solution_mapping:
                    bank_recs.append({
                        'type': 'enhancement',
                        'category': category,
                        'recommendation': f"Further enhance {category.replace('_', ' ')} features",
                        'priority': 'medium'
                    })
            
            # Generic recommendations
            bank_recs.extend([
                {
                    'type': 'general',
                    'category': 'engagement',
                    'recommendation': "Implement in-app feedback system",
                    'priority': 'medium'
                },
                {
                    'type': 'general',
                    'category': 'analytics',
                    'recommendation': "Use analytics to track feature usage and user behavior",
                    'priority': 'low'
                }
            ])
            
            recommendations[bank] = bank_recs
        
        return recommendations
    
    def analyze_ethics_biases(self, df: pd.DataFrame) -> Dict:
        """Analyze potential biases and ethical considerations"""
        
        analysis = {
            'review_biases': [],
            'sampling_issues': [],
            'ethical_considerations': []
        }
        
        # Check for negative bias
        negative_rate = (df['sentiment_label'] == 'negative').mean() * 100
        if negative_rate > 60:
            analysis['review_biases'].append(
                f"Negative review bias detected ({negative_rate:.1f}% negative reviews). "
                "Users are more likely to leave reviews when they have negative experiences."
            )
        
        # Check rating distribution
        rating_stats = df['rating'].describe()
        if rating_stats['std'] > 1.5:
            analysis['review_biases'].append(
                "High variance in ratings indicates polarized user opinions"
            )
        
        # Check for sampling bias
        bank_counts = df['bank_name'].value_counts()
        min_max_ratio = bank_counts.min() / bank_counts.max()
        
        if min_max_ratio < 0.7:
            analysis['sampling_issues'].append(
                f"Uneven sample sizes across banks (ratio: {min_max_ratio:.2f})"
            )
        
        # Ethical considerations
        analysis['ethical_considerations'].extend([
            "Reviews may not represent all user demographics equally",
            "Sentiment analysis may have cultural and linguistic biases",
            "Automated analysis should be validated with human review",
            "Privacy concerns: reviews contain personal financial experiences"
        ])
        
        return analysis
    
    def generate_full_report(self) -> Dict:
        """Generate comprehensive insights report"""
        
        # Load data
        df = self.load_review_data()
        
        if df.empty:
            logger.error("No data available for analysis")
            return {}
        
        # Perform analyses
        sentiment_trends = self.analyze_sentiment_trends(df)
        bank_comparison = self.compare_banks(df)
        drivers_pain_points = self.identify_drivers_pain_points(df)
        recommendations = self.generate_recommendations(drivers_pain_points)
        ethics_analysis = self.analyze_ethics_biases(df)
        
        # Compile report
        report = {
            'executive_summary': self._generate_executive_summary(bank_comparison),
            'bank_comparison': bank_comparison,
            'drivers_pain_points': drivers_pain_points,
            'recommendations': recommendations,
            'sentiment_trends': sentiment_trends,
            'ethics_analysis': ethics_analysis,
            'key_findings': self._extract_key_findings(
                bank_comparison, drivers_pain_points, recommendations
            )
        }
        
        return report
    
    def _generate_executive_summary(self, comparison: Dict) -> str:
        """Generate executive summary"""
        
        best_bank = max(
            comparison.items(),
            key=lambda x: x[1]['avg_rating'] * 0.5 + x[1]['positive_pct'] * 0.5
        )[0]
        
        summary = f"""
        EXECUTIVE SUMMARY
        
        Based on the analysis of {sum(c['total_reviews'] for c in comparison.values())} reviews:
        
        1. Overall Performance: {best_bank} shows the strongest performance in terms of 
           user ratings and positive sentiment.
        
        2. Key Metrics:
           - Average rating across all banks: {np.mean([c['avg_rating'] for c in comparison.values()]):.2f}/5
           - Average positive sentiment: {np.mean([c['positive_pct'] for c in comparison.values()]):.1f}%
        
        3. Critical Issues Identified:
           - App stability and crashes are common pain points
           - Login and authentication issues affect user experience
           - Transaction failures need immediate attention
        
        4. Success Factors:
           - User-friendly interfaces drive satisfaction
           - Reliable performance builds trust
           - Good customer support enhances loyalty
        
        This report provides actionable insights for improving mobile banking applications.
        """
        
        return summary
    
    def _extract_key_findings(self, comparison: Dict, drivers: Dict, recommendations: Dict) -> List[str]:
        """Extract key findings from analysis"""
        
        findings = []
        
        # Performance findings
        for bank, stats in comparison.items():
            if stats['avg_rating'] >= 4.0:
                findings.append(f"{bank}: Strong performance with {stats['avg_rating']:.1f}/5 rating")
            elif stats['avg_rating'] <= 2.5:
                findings.append(f"{bank}: Needs improvement with {stats['avg_rating']:.1f}/5 rating")
        
        # Driver findings
        for bank, insights in drivers.items():
            if insights['drivers']:
                top_driver = insights['drivers'][0]['category'].replace('_', ' ')
                findings.append(f"{bank}: Top satisfaction driver is '{top_driver}'")
            
            if insights['pain_points']:
                top_pain = insights['pain_points'][0]['category'].replace('_', ' ')
                findings.append(f"{bank}: Top pain point is '{top_pain}'")
        
        return findings