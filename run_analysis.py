#!/usr/bin/env python3
"""
Enhanced analysis runner with progress tracking and real-time updates
Command-line interface for the Ethiopian Bank Reviews Analysis project
"""

import argparse
import logging
import sys
import time
from tqdm import tqdm
import pandas as pd
import json
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection import ReviewScraper
from preprocessing import DataPreprocessor
from sentiment_analysis import SentimentAnalyzer
from thematic_analysis import ThematicAnalyzer
from visualization import ReviewVisualizer
from utils import setup_logging, validate_data_quality, save_metrics

class AnalysisRunner:
    """Enhanced analysis runner with progress tracking and CLI options"""
    
    def __init__(self, args):
        self.args = args
        setup_logging(level=logging.INFO if not args.verbose else logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        
    def run(self):
        """Run the analysis based on command line arguments"""
        self.logger.info("🚀 ETHIOPIAN BANK REVIEWS ANALYSIS - ENHANCED RUNNER")
        self.logger.info("=" * 70)
        
        try:
            if self.args.phase == 'all':
                return self._run_complete_analysis()
            elif self.args.phase == 'data':
                return self._run_data_phase()
            elif self.args.phase == 'sentiment':
                return self._run_sentiment_phase()
            elif self.args.phase == 'thematic':
                return self._run_thematic_phase()
            elif self.args.phase == 'visualize':
                return self._run_visualization_phase()
            else:
                self.logger.error(f"Unknown phase: {self.args.phase}")
                return False
                
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return False
        finally:
            self._print_execution_time()
    
    def _run_complete_analysis(self):
        """Run complete analysis pipeline with enhanced progress tracking"""
        self.logger.info("Running COMPLETE analysis pipeline...")
        
        phases = [
            ("Data Collection & Preprocessing", self._run_data_phase),
            ("Sentiment Analysis", self._run_sentiment_phase), 
            ("Thematic Analysis", self._run_thematic_phase),
            ("Visualization & Reporting", self._run_visualization_phase)
        ]
        
        with tqdm(total=len(phases), desc="Overall Progress", position=0) as overall_pbar:
            for phase_name, phase_func in phases:
                overall_pbar.set_description(f"Processing: {phase_name}")
                
                # Create nested progress bar for phase details
                phase_success = phase_func(show_progress=True)
                
                if not phase_success:
                    self.logger.error(f"Phase failed: {phase_name}")
                    return False
                
                overall_pbar.update(1)
                overall_pbar.set_postfix({"Status": "Completed"})
        
        self.logger.info("🎉 COMPLETE ANALYSIS PIPELINE FINISHED SUCCESSFULLY!")
        return True
    
    def _run_data_phase(self, show_progress=False):
        """Run data collection and preprocessing phase"""
        if show_progress:
            pbar = tqdm(total=5, desc="Data Phase", position=1, leave=False)
        
        try:
            # Step 1: Initialize scraper
            if show_progress:
                pbar.set_description("Initializing scraper")
            scraper = ReviewScraper()
            if show_progress:
                pbar.update(1)
            
            # Step 2: Scrape data
            if show_progress:
                pbar.set_description("Scraping reviews from Google Play")
            raw_reviews = scraper.scrape_all_banks(reviews_per_bank=self.args.reviews_per_bank)
            if show_progress:
                pbar.update(1)
                pbar.set_postfix({"Reviews": len(raw_reviews)})
            
            # Step 3: Preprocess data
            if show_progress:
                pbar.set_description("Preprocessing and cleaning data")
            preprocessor = DataPreprocessor()
            processed_df, quality_metrics = preprocessor.run_pipeline('data/raw/raw_reviews.csv')
            if show_progress:
                pbar.update(1)
                pbar.set_postfix({"Clean Reviews": len(processed_df)})
            
            # Step 4: Validate quality
            if show_progress:
                pbar.set_description("Validating data quality")
            meets_requirements = validate_data_quality(processed_df)
            if show_progress:
                pbar.update(1)
                pbar.set_postfix({"Quality OK": meets_requirements})
            
            # Step 5: Save data
            if show_progress:
                pbar.set_description("Saving processed data")
            os.makedirs('data/processed', exist_ok=True)
            processed_df.to_csv('data/processed/processed_reviews.csv', index=False)
            save_metrics(quality_metrics, 'data_quality_metrics.json')
            if show_progress:
                pbar.update(1)
            
            self.logger.info(f"✅ Data phase completed: {len(processed_df)} clean reviews")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data phase failed: {str(e)}")
            return False
        finally:
            if show_progress:
                pbar.close()
    
    def _run_sentiment_phase(self, show_progress=False):
        """Run sentiment analysis phase"""
        if show_progress:
            pbar = tqdm(total=4, desc="Sentiment Analysis", position=1, leave=False)
        
        try:
            # Step 1: Load data
            if show_progress:
                pbar.set_description("Loading processed data")
            processed_df = pd.read_csv('data/processed/processed_reviews.csv')
            if show_progress:
                pbar.update(1)
            
            # Step 2: Initialize analyzer
            if show_progress:
                pbar.set_description("Initializing sentiment analyzer")
            sentiment_analyzer = SentimentAnalyzer(method=self.args.sentiment_method)
            if show_progress:
                pbar.update(1)
            
            # Step 3: Analyze sentiment
            if show_progress:
                pbar.set_description("Analyzing review sentiments")
            sentiment_results = sentiment_analyzer.analyze_dataframe(processed_df)
            if show_progress:
                pbar.update(1)
                pbar.set_postfix({"Analyzed": len(sentiment_results)})
            
            # Step 4: Save results
            if show_progress:
                pbar.set_description("Saving sentiment results")
            os.makedirs('data/results', exist_ok=True)
            sentiment_results.to_csv('data/results/sentiment_analysis.csv', index=False)
            
            # Calculate and save metrics
            sentiment_metrics = sentiment_analyzer.calculate_sentiment_metrics(sentiment_results)
            save_metrics(sentiment_metrics, 'sentiment_metrics.json')
            if show_progress:
                pbar.update(1)
            
            self.logger.info(f"✅ Sentiment analysis completed: {len(sentiment_results)} reviews analyzed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Sentiment analysis failed: {str(e)}")
            return False
        finally:
            if show_progress:
                pbar.close()
    
    def _run_thematic_phase(self, show_progress=False):
        """Run thematic analysis phase"""
        if show_progress:
            pbar = tqdm(total=3, desc="Thematic Analysis", position=1, leave=False)
        
        try:
            # Step 1: Load sentiment results
            if show_progress:
                pbar.set_description("Loading sentiment results")
            sentiment_results = pd.read_csv('data/results/sentiment_analysis.csv')
            if show_progress:
                pbar.update(1)
            
            # Step 2: Analyze themes
            if show_progress:
                pbar.set_description("Extracting themes and keywords")
            thematic_analyzer = ThematicAnalyzer()
            thematic_results = thematic_analyzer.run_analysis(sentiment_results)
            if show_progress:
                pbar.update(1)
            
            # Step 3: Save results
            if show_progress:
                pbar.set_description("Saving thematic results")
            save_metrics(thematic_results, 'thematic_analysis.json')
            
            # Generate insights report
            insights_report = self._generate_quick_insights(thematic_results)
            save_metrics(insights_report, 'quick_insights.json')
            if show_progress:
                pbar.update(1)
                pbar.set_postfix({"Themes": sum(len(a.get('themes_identified', {})) for a in thematic_results.values())})
            
            self.logger.info("✅ Thematic analysis completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Thematic analysis failed: {str(e)}")
            return False
        finally:
            if show_progress:
                pbar.close()
    
    def _run_visualization_phase(self, show_progress=False):
        """Run visualization and reporting phase"""
        if show_progress:
            pbar = tqdm(total=3, desc="Visualization", position=1, leave=False)
        
        try:
            # Step 1: Load data
            if show_progress:
                pbar.set_description("Loading analysis results")
            sentiment_results = pd.read_csv('data/results/sentiment_analysis.csv')
            
            with open('data/results/thematic_analysis.json', 'r') as f:
                thematic_results = json.load(f)
            if show_progress:
                pbar.update(1)
            
            # Step 2: Create visualizations
            if show_progress:
                pbar.set_description("Generating visualizations")
            visualizer = ReviewVisualizer()
            
            # Create dashboards and charts
            sentiment_dashboard = visualizer.create_sentiment_dashboard(sentiment_results)
            sentiment_dashboard.write_html('data/results/sentiment_dashboard.html')
            
            theme_wordclouds = visualizer.create_theme_wordclouds(thematic_results)
            theme_wordclouds.savefig('data/results/theme_wordclouds.png', dpi=300, bbox_inches='tight')
            
            comparison_heatmap = visualizer.create_theme_comparison_heatmap(thematic_results)
            comparison_heatmap.savefig('data/results/theme_comparison.png', dpi=300, bbox_inches='tight')
            if show_progress:
                pbar.update(1)
            
            # Step 3: Generate reports
            if show_progress:
                pbar.set_description("Generating reports")
            self._generate_quick_report(sentiment_results, thematic_results)
            if show_progress:
                pbar.update(1)
            
            self.logger.info("✅ Visualization and reporting completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Visualization phase failed: {str(e)}")
            return False
        finally:
            if show_progress:
                pbar.close()
    
    def _generate_quick_insights(self, thematic_results):
        """Generate quick insights from thematic analysis"""
        insights = {
            'generation_time': time.time(),
            'summary': {}
        }
        
        for bank, analysis in thematic_results.items():
            themes = analysis.get('themes_identified', {})
            insights['summary'][bank] = {
                'theme_count': len(themes),
                'primary_themes': list(themes.keys())[:3],
                'total_keywords': sum(len(keywords) for keywords in themes.values())
            }
        
        return insights
    
    def _generate_quick_report(self, sentiment_results, thematic_results):
        """Generate a quick text report"""
        report_path = 'data/results/quick_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("QUICK ANALYSIS REPORT - ETHIOPIAN BANK REVIEWS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("SENTIMENT ANALYSIS SUMMARY:\n")
            f.write(f"Total Reviews: {len(sentiment_results)}\n")
            f.write(f"Positive: {(sentiment_results['sentiment_label'] == 'positive').mean()*100:.1f}%\n")
            f.write(f"Negative: {(sentiment_results['sentiment_label'] == 'negative').mean()*100:.1f}%\n")
            f.write(f"Neutral: {(sentiment_results['sentiment_label'] == 'neutral').mean()*100:.1f}%\n\n")
            
            f.write("THEMATIC ANALYSIS SUMMARY:\n")
            for bank, analysis in thematic_results.items():
                themes = analysis.get('themes_identified', {})
                f.write(f"{bank}: {len(themes)} themes identified\n")
                for theme in list(themes.keys())[:2]:  # Top 2 themes
                    f.write(f"  - {theme}\n")
                f.write("\n")
            
            f.write(f"\nReport generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def _print_execution_time(self):
        """Print total execution time"""
        execution_time = time.time() - self.start_time
        minutes, seconds = divmod(execution_time, 60)
        self.logger.info(f"⏱️  Total execution time: {int(minutes)}m {seconds:.1f}s")

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Ethiopian Bank Reviews Analysis Runner')
    
    parser.add_argument('--phase', choices=['all', 'data', 'sentiment', 'thematic', 'visualize'],
                       default='all', help='Analysis phase to run (default: all)')
    
    parser.add_argument('--reviews-per-bank', type=int, default=400,
                       help='Number of reviews to scrape per bank (default: 400)')
    
    parser.add_argument('--sentiment-method', choices=['distilbert', 'vader', 'textblob'],
                       default='distilbert', help='Sentiment analysis method (default: distilbert)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True) 
    os.makedirs('data/results', exist_ok=True)
    
    # Run analysis
    runner = AnalysisRunner(args)
    success = runner.run()
    
    if success:
        print(f"\n✨ Analysis completed successfully!")
        print(f"📁 Check results in: data/results/")
        return 0
    else:
        print(f"\n❌ Analysis failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)