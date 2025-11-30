
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import nltk
from nltk import bigrams, trigrams
from collections import Counter
import re
from typing import Dict, List, Tuple
import logging
#from config.banks_config import THEME_CATEGORIES

logger = logging.getLogger(__name__)

class ThematicAnalyzer:
    """Perform thematic analysis on review data"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.theme_keywords = self._initialize_theme_keywords()
    
    def _initialize_theme_keywords(self) -> Dict:
        """Initialize keyword mappings for theme categories"""
        return {
            'User Interface & Experience': [
                'interface', 'design', 'layout', 'navigation', 'button', 'screen',
                'look', 'appearance', 'theme', 'color', 'font', 'size', 'display'
            ],
            'Transaction Performance': [
                'transaction', 'transfer', 'payment', 'slow', 'fast', 'speed',
                'processing', 'complete', 'failed', 'successful', 'instant',
                'delay', 'waiting', 'time', 'quick'
            ],
            'Account Access & Security': [
                'login', 'password', 'pin', 'security', 'access', 'account',
                'verification', 'authentication', 'biometric', 'fingerprint',
                'face', 'lock', 'unlock', 'secure', 'hack', 'privacy'
            ],
            'Customer Support': [
                'support', 'help', 'service', 'assistance', 'response',
                'contact', 'call', 'email', 'chat', 'representative',
                'complaint', 'issue', 'problem', 'resolve', 'staff'
            ],
            'App Reliability & Bugs': [
                'crash', 'bug', 'error', 'glitch', 'freeze', 'hang',
                'not working', 'broken', 'issue', 'problem', 'fix',
                'update', 'version', 'stable', 'reliable'
            ],
            'Feature Requests': [
                'should', 'could', 'would', 'please', 'add', 'feature',
                'missing', 'need', 'want', 'suggest', 'recommend',
                'improvement', 'enhancement', 'option', 'setting'
            ]
        }
    
    def extract_keywords_tfidf(self, texts: List[str], max_features: int = 100) -> List[str]:
        """Extract keywords using TF-IDF"""
        try:
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),  
                min_df=2  
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            avg_tfidf_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
            
            # Create keyword-score pairs and sort
            keyword_scores = list(zip(feature_names, avg_tfidf_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            top_keywords = [kw for kw, score in keyword_scores[:50]]
            logger.info(f"Extracted {len(top_keywords)} keywords using TF-IDF")
            
            return top_keywords
            
        except Exception as e:
            logger.error(f"Error in TF-IDF keyword extraction: {str(e)}")
            return []
    
    def extract_phrases_spacy(self, texts: List[str]) -> List[str]:
        """Extract meaningful phrases using spaCy"""
        phrases = []
        
        for text in texts:
            doc = self.nlp(str(text))
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  
                    phrases.append(chunk.text.lower())
            
            # Extract verb phrases and other patterns
            for token in doc:
                if token.pos_ in ['VERB', 'ADJ'] and token.is_alpha:
                    phrases.append(token.lemma_.lower())
        
        # Count and return most common phrases
        phrase_counts = Counter(phrases)
        common_phrases = [phrase for phrase, count in phrase_counts.most_common(100) if count > 1]
        
        logger.info(f"Extracted {len(common_phrases)} phrases using spaCy")
        return common_phrases
    
    def map_keywords_to_themes(self, keywords: List[str]) -> Dict[str, List[str]]:
        """Map extracted keywords to predefined themes"""
        theme_mappings = {theme: [] for theme in self.theme_keywords.keys()}

        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            matched = False
            
            for theme, theme_words in self.theme_keywords.items():
                for theme_word in theme_words:
                    if theme_word in keyword_lower or keyword_lower in theme_word:
                        theme_mappings[theme].append(keyword)
                        matched = True
                        break
                if matched:
                    break
            
            # If no match found, assign to most relevant category based on content
            if not matched:
                if any(word in keyword_lower for word in ['ui', 'design', 'interface', 'button']):
                    theme_mappings['User Interface & Experience'].append(keyword)
                elif any(word in keyword_lower for word in ['transfer', 'payment', 'transaction', 'slow', 'fast']):
                    theme_mappings['Transaction Performance'].append(keyword)
                elif any(word in keyword_lower for word in ['login', 'password', 'security', 'access']):
                    theme_mappings['Account Access & Security'].append(keyword)
                elif any(word in keyword_lower for word in ['support', 'help', 'service', 'response']):
                    theme_mappings['Customer Support'].append(keyword)
                elif any(word in keyword_lower for word in ['crash', 'error', 'bug', 'not working']):
                    theme_mappings['App Reliability & Bugs'].append(keyword)
                elif any(word in keyword_lower for word in ['should', 'could', 'add', 'feature']):
                    theme_mappings['Feature Requests'].append(keyword)
        
        # Remove empty themes and limit keywords per theme
        final_themes = {}
        for theme, theme_keywords in theme_mappings.items():
            if theme_keywords:
                final_themes[theme] = list(set(theme_keywords))[:10]  # Limit to top 10 keywords
        
        logger.info(f"Mapped keywords to {len(final_themes)} themes")
        return final_themes
    
    def analyze_bank_themes(self, bank_df: pd.DataFrame, bank_name: str) -> Dict:
        """Perform thematic analysis for a specific bank"""
        logger.info(f"Analyzing themes for {bank_name}")
        
        # Combine all reviews
        reviews_text = bank_df['review_text'].dropna().tolist()
        
        if not reviews_text:
            logger.warning(f"No reviews found for {bank_name}")
            return {}
        
        # Extract keywords using multiple methods
        tfidf_keywords = self.extract_keywords_tfidf(reviews_text)
        spacy_phrases = self.extract_phrases_spacy(reviews_text)
        
        # Combine and deduplicate keywords
        all_keywords = list(set(tfidf_keywords + spacy_phrases))
        
        # Map to themes
        theme_mappings = self.map_keywords_to_themes(all_keywords)
        
        # Get example reviews for each theme
        theme_examples = self.get_theme_examples(bank_df, theme_mappings)
        
        analysis_result = {
            'bank': bank_name,
            'total_reviews_analyzed': len(reviews_text),
            'themes_identified': theme_mappings,
            'theme_examples': theme_examples,
            'top_keywords': all_keywords[:20]  
        }
        
        return analysis_result
    
    def get_theme_examples(self, bank_df: pd.DataFrame, theme_mappings: Dict) -> Dict:
        """Get example reviews for each theme"""
        theme_examples = {}
        
        for theme, keywords in theme_mappings.items():
            if not keywords:
                continue
                
            # Find reviews that contain theme keywords
            matching_reviews = []
            
            for _, review in bank_df.iterrows():
                review_text = str(review['review_text']).lower()
                
                # Check if review contains any of the theme keywords
                for keyword in keywords[:5]:  # Use top 5 keywords
                    if keyword.lower() in review_text:
                        matching_reviews.append({
                            'review_text': review['review_text'],
                            'sentiment': review['sentiment_label'],
                            'rating': review['rating']
                        })
                        break
                
                # Limit number of examples
                if len(matching_reviews) >= 3:
                    break
            
            theme_examples[theme] = matching_reviews
        
        return theme_examples
    
    def run_analysis(self, sentiment_df: pd.DataFrame) -> Dict:
        """Run thematic analysis for all banks"""
        logger.info("Starting thematic analysis for all banks")
        
        results = {}
        
        for bank_name in sentiment_df['bank_name'].unique():
            bank_df = sentiment_df[sentiment_df['bank_name'] == bank_name]
            bank_analysis = self.analyze_bank_themes(bank_df, bank_name)
            results[bank_name] = bank_analysis
        
        logger.info("Thematic analysis completed for all banks")
        return results

def save_thematic_results(results: Dict, filename: str = "thematic_analysis.json"):
    """Save thematic analysis results"""
    import json
    
    try:
        with open(f"data/results/{filename}", 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Thematic results saved to data/results/{filename}")
    except Exception as e:
        logger.error(f"Error saving thematic results: {str(e)}")

if __name__ == "__main__":
    # Load sentiment analysis results
    sentiment_df = pd.read_csv("data/results/sentiment_analysis.csv")
    
    # Perform thematic analysis
    analyzer = ThematicAnalyzer()
    thematic_results = analyzer.run_analysis(sentiment_df)
    
    # Save results
    save_thematic_results(thematic_results)
    
    # Print summary
    for bank, analysis in thematic_results.items():
        print(f"\n{bank}:")
        print(f"  Themes identified: {len(analysis.get('themes_identified', {}))}")
        for theme, keywords in analysis.get('themes_identified', {}).items():
            print(f"  - {theme}: {', '.join(keywords[:3])}...")