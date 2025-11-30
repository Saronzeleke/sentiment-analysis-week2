"""
Main execution script for Ethiopian Bank Reviews Analysis
Complete project orchestration and reporting
"""

import logging
import pandas as pd
import json
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection import ReviewScraper
from preprocessing import DataPreprocessor
from sentiment_analysis import SentimentAnalyzer
from thematic_analysis import ThematicAnalyzer
from visualization import ReviewVisualizer
from utils import setup_logging, validate_data_quality, save_metrics

class EthiopianBankAnalysis:
    """Main class to orchestrate the complete analysis pipeline"""
    
    def __init__(self):
        setup_logging()
        self.logger = logging.getLogger(__name__)
        self.results = {
            'project_start_time': datetime.now().isoformat(),
            'execution_phases': {},
            'key_metrics': {},
            'recommendations': {}
        }
    
    def run_complete_analysis(self):
        """Execute complete analysis pipeline"""
        try:
            self.logger.info("🚀 STARTING ETHIOPIAN BANK REVIEWS ANALYSIS PROJECT")
            self.logger.info("=" * 70)
            
            # Phase 1: Data Collection & Preprocessing
            phase1_success = self._run_data_collection_phase()
            if not phase1_success:
                self.logger.error("❌ Data collection phase failed")
                return False
            
            # Phase 2: Sentiment Analysis
            phase2_success = self._run_sentiment_analysis_phase()
            if not phase2_success:
                self.logger.error("❌ Sentiment analysis phase failed")
                return False
            
            # Phase 3: Thematic Analysis
            phase3_success = self._run_thematic_analysis_phase()
            if not phase3_success:
                self.logger.error("❌ Thematic analysis phase failed")
                return False
            
            # Phase 4: Visualization & Reporting
            phase4_success = self._run_visualization_phase()
            if not phase4_success:
                self.logger.error("❌ Visualization phase failed")
                return False
            
            # Generate final report
            self._generate_final_report()
            
            self.logger.info("🎉 PROJECT COMPLETED SUCCESSFULLY!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Project execution failed: {str(e)}")
            raise
    
    def _run_data_collection_phase(self):
        """Execute Phase 1: Data Collection & Preprocessing"""
        self.logger.info("\n📊 PHASE 1: DATA COLLECTION & PREPROCESSING")
        self.logger.info("-" * 50)
        
        try:
            # Step 1: Data Collection
            self.logger.info("Step 1: Scraping reviews from Google Play Store...")
            scraper = ReviewScraper()
            raw_reviews = scraper.scrape_all_banks(reviews_per_bank=400)
            
            self.logger.info(f"✓ Raw data collected: {len(raw_reviews)} reviews")
            
            # Save raw data
            os.makedirs('data/raw', exist_ok=True)
            raw_reviews.to_csv('data/raw/raw_reviews.csv', index=False)
            
            # Step 2: Data Preprocessing
            self.logger.info("Step 2: Preprocessing and cleaning data...")
            preprocessor = DataPreprocessor()
            processed_df, quality_metrics = preprocessor.run_pipeline('data/raw/raw_reviews.csv')
            
            self.logger.info(f"✓ Data preprocessing completed: {len(processed_df)} clean reviews")
            
            # Step 3: Data Quality Validation
            self.logger.info("Step 3: Validating data quality...")
            meets_requirements = validate_data_quality(processed_df)
            
            # Save processed data
            os.makedirs('data/processed', exist_ok=True)
            processed_df.to_csv('data/processed/processed_reviews.csv', index=False)
            
            # Store phase results
            self.results['execution_phases']['data_collection'] = {
                'status': 'completed',
                'raw_reviews_count': len(raw_reviews),
                'processed_reviews_count': len(processed_df),
                'data_quality_metrics': quality_metrics,
                'meets_requirements': meets_requirements,
                'completion_time': datetime.now().isoformat()
            }
            
            self.results['key_metrics']['data_quality'] = {
                'total_reviews': len(processed_df),
                'missing_data_percentage': quality_metrics.get('missing_content_pct', 0),
                'reviews_per_bank': quality_metrics.get('reviews_per_bank', {}),
                'quality_check_passed': meets_requirements
            }
            
            self.logger.info("✅ PHASE 1 COMPLETED SUCCESSFULLY")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data collection phase failed: {str(e)}")
            self.results['execution_phases']['data_collection'] = {
                'status': 'failed',
                'error': str(e),
                'completion_time': datetime.now().isoformat()
            }
            return False
    
    def _run_sentiment_analysis_phase(self):
        """Execute Phase 2: Sentiment Analysis"""
        self.logger.info("\n😊 PHASE 2: SENTIMENT ANALYSIS")
        self.logger.info("-" * 50)
        
        try:
            # Load processed data
            self.logger.info("Step 1: Loading processed data...")
            processed_df = pd.read_csv('data/processed/processed_reviews.csv')
            
            # Step 2: Initialize sentiment analyzer
            self.logger.info("Step 2: Initializing sentiment analyzer (DistilBERT)...")
            sentiment_analyzer = SentimentAnalyzer(method="distilbert")
            
            # Step 3: Perform sentiment analysis
            self.logger.info("Step 3: Analyzing sentiment for all reviews...")
            sentiment_results = sentiment_analyzer.analyze_dataframe(processed_df)
            
            self.logger.info(f"✓ Sentiment analysis completed: {len(sentiment_results)} reviews analyzed")
            
            # Step 4: Calculate sentiment metrics
            self.logger.info("Step 4: Calculating sentiment metrics...")
            sentiment_metrics = sentiment_analyzer.calculate_sentiment_metrics(sentiment_results)
            
            # Step 5: Save results
            self.logger.info("Step 5: Saving sentiment analysis results...")
            os.makedirs('data/results', exist_ok=True)
            sentiment_results.to_csv('data/results/sentiment_analysis.csv', index=False)
            save_metrics(sentiment_metrics, 'sentiment_metrics.json')
            
            # Store phase results
            self.results['execution_phases']['sentiment_analysis'] = {
                'status': 'completed',
                'reviews_analyzed': len(sentiment_results),
                'coverage_percentage': (len(sentiment_results) / len(processed_df)) * 100,
                'primary_method': 'distilbert',
                'completion_time': datetime.now().isoformat()
            }
            
            self.results['key_metrics']['sentiment_analysis'] = {
                'overall_positive_rate': sentiment_metrics.get('overall_sentiment_distribution', {}).get('positive', 0),
                'overall_negative_rate': sentiment_metrics.get('overall_sentiment_distribution', {}).get('negative', 0),
                'average_sentiment_by_bank': sentiment_metrics.get('average_sentiment_by_bank', {}),
                'analysis_coverage': (len(sentiment_results) / len(processed_df)) * 100
            }
            
            self.logger.info("✅ PHASE 2 COMPLETED SUCCESSFULLY")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Sentiment analysis phase failed: {str(e)}")
            self.results['execution_phases']['sentiment_analysis'] = {
                'status': 'failed',
                'error': str(e),
                'completion_time': datetime.now().isoformat()
            }
            return False
    
    def _run_thematic_analysis_phase(self):
        """Execute Phase 3: Thematic Analysis"""
        self.logger.info("\n🔍 PHASE 3: THEMATIC ANALYSIS")
        self.logger.info("-" * 50)
        
        try:
            # Load sentiment results
            self.logger.info("Step 1: Loading sentiment analysis results...")
            sentiment_results = pd.read_csv('data/results/sentiment_analysis.csv')
            
            # Step 2: Initialize thematic analyzer
            self.logger.info("Step 2: Initializing thematic analyzer...")
            thematic_analyzer = ThematicAnalyzer()
            
            # Step 3: Perform thematic analysis
            self.logger.info("Step 3: Extracting themes and keywords...")
            thematic_results = thematic_analyzer.run_analysis(sentiment_results)
            
            # Step 4: Generate actionable insights
            self.logger.info("Step 4: Generating actionable insights...")
            insights_report = self._generate_insights_report(thematic_results)
            
            # Step 5: Save results
            self.logger.info("Step 5: Saving thematic analysis results...")
            save_metrics(thematic_results, 'thematic_analysis.json')
            save_metrics(insights_report, 'actionable_insights.json')
            
            # Store phase results
            total_themes = sum(len(analysis.get('themes_identified', {})) 
                             for analysis in thematic_results.values())
            
            self.results['execution_phases']['thematic_analysis'] = {
                'status': 'completed',
                'total_themes_identified': total_themes,
                'themes_per_bank': {bank: len(analysis.get('themes_identified', {})) 
                                  for bank, analysis in thematic_results.items()},
                'completion_time': datetime.now().isoformat()
            }
            
            self.results['key_metrics']['thematic_analysis'] = {
                'total_themes': total_themes,
                'average_themes_per_bank': total_themes / len(thematic_results) if thematic_results else 0,
                'banks_analyzed': list(thematic_results.keys())
            }
            
            self.results['recommendations'] = insights_report.get('recommendations', {})
            
            self.logger.info("✅ PHASE 3 COMPLETED SUCCESSFULLY")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Thematic analysis phase failed: {str(e)}")
            self.results['execution_phases']['thematic_analysis'] = {
                'status': 'failed',
                'error': str(e),
                'completion_time': datetime.now().isoformat()
            }
            return False
    
    def _run_visualization_phase(self):
        """Execute Phase 4: Visualization & Reporting"""
        self.logger.info("\n📈 PHASE 4: VISUALIZATION & REPORTING")
        self.logger.info("-" * 50)
        
        try:
            # Load necessary data
            self.logger.info("Step 1: Loading analysis results...")
            sentiment_results = pd.read_csv('data/results/sentiment_analysis.csv')
            
            with open('data/results/thematic_analysis.json', 'r') as f:
                thematic_results = json.load(f)
            
            # Step 2: Create visualizations
            self.logger.info("Step 2: Creating visualizations...")
            visualizer = ReviewVisualizer()
            
            # Create sentiment dashboard
            self.logger.info("   - Creating sentiment dashboard...")
            sentiment_dashboard = visualizer.create_sentiment_dashboard(sentiment_results)
            sentiment_dashboard.write_html('data/results/sentiment_dashboard.html')
            
            # Create theme visualizations
            self.logger.info("   - Creating theme visualizations...")
            theme_wordclouds = visualizer.create_theme_wordclouds(thematic_results)
            theme_wordclouds.savefig('data/results/theme_wordclouds.png', dpi=300, bbox_inches='tight')
            
            # Create comparison heatmap
            self.logger.info("   - Creating comparison heatmap...")
            comparison_heatmap = visualizer.create_theme_comparison_heatmap(thematic_results)
            comparison_heatmap.savefig('data/results/theme_comparison.png', dpi=300, bbox_inches='tight')
            
            # Step 3: Generate executive summary
            self.logger.info("Step 3: Generating executive summary...")
            self._generate_executive_summary()
            
            # Store phase results
            self.results['execution_phases']['visualization'] = {
                'status': 'completed',
                'visualizations_created': [
                    'sentiment_dashboard.html',
                    'theme_wordclouds.png', 
                    'theme_comparison.png'
                ],
                'completion_time': datetime.now().isoformat()
            }
            
            self.logger.info("✅ PHASE 4 COMPLETED SUCCESSFULLY")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Visualization phase failed: {str(e)}")
            self.results['execution_phases']['visualization'] = {
                'status': 'failed',
                'error': str(e),
                'completion_time': datetime.now().isoformat()
            }
            return False
    
    def _generate_insights_report(self, thematic_results):
        """Generate actionable insights from thematic analysis"""
        insights_report = {
            'generation_date': datetime.now().isoformat(),
            'banks_analyzed': list(thematic_results.keys()),
            'key_findings': {},
            'recommendations': {}
        }
        
        for bank, analysis in thematic_results.items():
            themes = analysis.get('themes_identified', {})
            
            # Key findings
            insights_report['key_findings'][bank] = {
                'primary_themes': list(themes.keys()),
                'theme_count': len(themes),
                'strongest_areas': [],
                'improvement_areas': []
            }
            
            # Recommendations
            recommendations = []
            for theme, keywords in themes.items():
                # Generate specific recommendations based on themes
                if any(word in theme.lower() for word in ['login', 'access', 'password']):
                    recommendations.append({
                        'area': 'Account Access & Security',
                        'issue': theme,
                        'recommendation': 'Improve authentication flow, add biometric options, simplify password reset process',
                        'priority': 'High'
                    })
                elif any(word in theme.lower() for word in ['transaction', 'transfer', 'payment', 'slow']):
                    recommendations.append({
                        'area': 'Transaction Performance', 
                        'issue': theme,
                        'recommendation': 'Optimize processing speed, add transaction status tracking, improve error handling',
                        'priority': 'High'
                    })
                elif any(word in theme.lower() for word in ['ui', 'interface', 'design', 'navigation']):
                    recommendations.append({
                        'area': 'User Interface',
                        'issue': theme,
                        'recommendation': 'Redesign for better usability, conduct user testing, improve visual hierarchy',
                        'priority': 'Medium'
                    })
                elif any(word in theme.lower() for word in ['support', 'help', 'service', 'response']):
                    recommendations.append({
                        'area': 'Customer Support',
                        'issue': theme,
                        'recommendation': 'Enhance in-app support, reduce response times, add chat functionality',
                        'priority': 'Medium'
                    })
                elif any(word in theme.lower() for word in ['crash', 'bug', 'error', 'freeze']):
                    recommendations.append({
                        'area': 'App Stability',
                        'issue': theme,
                        'recommendation': 'Prioritize stability fixes, improve error handling, enhance testing',
                        'priority': 'Critical'
                    })
            
            insights_report['recommendations'][bank] = recommendations[:5]  # Top 5 recommendations
        
        return insights_report
    
    def _generate_executive_summary(self):
        """Generate executive summary report"""
        executive_summary = {
            'project_title': 'Ethiopian Bank Mobile App Reviews Analysis',
            'execution_date': self.results['project_start_time'],
            'overview': {
                'total_reviews_analyzed': self.results['key_metrics']['data_quality']['total_reviews'],
                'banks_covered': list(self.results['key_metrics']['thematic_analysis']['banks_analyzed']),
                'data_quality_score': f"{100 - self.results['key_metrics']['data_quality']['missing_data_percentage']:.1f}%",
                'analysis_coverage': f"{self.results['key_metrics']['sentiment_analysis']['analysis_coverage']:.1f}%"
            },
            'key_insights': {
                'overall_sentiment': {
                    'positive': f"{self.results['key_metrics']['sentiment_analysis']['overall_positive_rate']*100:.1f}%",
                    'negative': f"{self.results['key_metrics']['sentiment_analysis']['overall_negative_rate']*100:.1f}%"
                },
                'thematic_analysis': {
                    'total_themes_identified': self.results['key_metrics']['thematic_analysis']['total_themes'],
                    'average_themes_per_bank': self.results['key_metrics']['thematic_analysis']['average_themes_per_bank']
                }
            },
            'top_recommendations': self.results['recommendations']
        }
        
        save_metrics(executive_summary, 'executive_summary.json')
        
        # Also create a text version for quick reading
        with open('data/results/executive_summary.txt', 'w') as f:
            f.write("EXECUTIVE SUMMARY - ETHIOPIAN BANK MOBILE APP ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("PROJECT OVERVIEW:\n")
            f.write(f"- Total Reviews Analyzed: {executive_summary['overview']['total_reviews_analyzed']}\n")
            f.write(f"- Banks Covered: {', '.join(executive_summary['overview']['banks_covered'])}\n")
            f.write(f"- Data Quality Score: {executive_summary['overview']['data_quality_score']}\n")
            f.write(f"- Analysis Coverage: {executive_summary['overview']['analysis_coverage']}\n\n")
            
            f.write("KEY INSIGHTS:\n")
            f.write(f"- Overall Positive Sentiment: {executive_summary['key_insights']['overall_sentiment']['positive']}\n")
            f.write(f"- Overall Negative Sentiment: {executive_summary['key_insights']['overall_sentiment']['negative']}\n")
            f.write(f"- Total Themes Identified: {executive_summary['key_insights']['thematic_analysis']['total_themes']}\n")
            f.write(f"- Average Themes per Bank: {executive_summary['key_insights']['thematic_analysis']['average_themes_per_bank']:.1f}\n\n")
            
            f.write("TOP RECOMMENDATIONS:\n")
            for bank, recs in executive_summary['top_recommendations'].items():
                f.write(f"\n{bank.upper()}:\n")
                for i, rec in enumerate(recs[:3], 1):  # Top 3 per bank
                    f.write(f"  {i}. [{rec['priority']}] {rec['area']}: {rec['recommendation']}\n")
    
    def _generate_final_report(self):
        """Generate comprehensive final report"""
        self.results['project_end_time'] = datetime.now().isoformat()
        self.results['total_duration'] = str(
            datetime.fromisoformat(self.results['project_end_time']) - 
            datetime.fromisoformat(self.results['project_start_time'])
        )
        
        # Calculate success rate
        completed_phases = [phase for phase in self.results['execution_phases'].values() 
                          if phase.get('status') == 'completed']
        self.results['success_rate'] = f"{len(completed_phases)}/4 phases completed"
        
        # Save final results
        save_metrics(self.results, 'final_project_results.json')
        
        # Print final summary
        self._print_final_summary()
    
    def _print_final_summary(self):
        """Print final project summary to console"""
        print("\n" + "=" * 70)
        print("🎉 PROJECT EXECUTION SUMMARY")
        print("=" * 70)
        
        print(f"📅 Execution Period: {self.results['project_start_time']} to {self.results['project_end_time']}")
        print(f"⏱️  Total Duration: {self.results['total_duration']}")
        print(f"✅ Success Rate: {self.results['success_rate']}")
        
        print(f"\n📊 KEY METRICS:")
        data_metrics = self.results['key_metrics']['data_quality']
        sentiment_metrics = self.results['key_metrics']['sentiment_analysis']
        thematic_metrics = self.results['key_metrics']['thematic_analysis']
        
        print(f"   • Reviews Collected: {data_metrics['total_reviews']}")
        print(f"   • Data Quality: {100 - data_metrics['missing_data_percentage']:.1f}%")
        print(f"   • Sentiment Coverage: {sentiment_metrics['analysis_coverage']:.1f}%")
        print(f"   • Positive Reviews: {sentiment_metrics['overall_positive_rate']*100:.1f}%")
        print(f"   • Themes Identified: {thematic_metrics['total_themes']}")
        
        print(f"\n🏦 BANKS ANALYZED: {', '.join(thematic_metrics['banks_analyzed'])}")
        
        print(f"\n📁 RESULTS SAVED IN:")
        print("   • data/raw/ - Raw scraped data")
        print("   • data/processed/ - Cleaned and processed data") 
        print("   • data/results/ - Analysis results and visualizations")
        
        print(f"\n🔍 ACCESS RESULTS:")
        print("   • Run notebooks/02_sentiment_analysis.ipynb for detailed sentiment analysis")
        print("   • Run notebooks/03_thematic_analysis.ipynb for detailed thematic analysis")
        print("   • Open data/results/sentiment_dashboard.html for interactive dashboard")
        print("   • Check data/results/executive_summary.txt for quick insights")

def main():
    """Main execution function"""
    analysis = EthiopianBankAnalysis()
    success = analysis.run_complete_analysis()
    
    if success:
        print("\n✨ PROJECT COMPLETED! Check the results in data/results/ directory")
        return 0
    else:
        print("\n❌ PROJECT FAILED! Check logs for details")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)