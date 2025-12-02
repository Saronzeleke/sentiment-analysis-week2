import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from typing import Dict, List, Optional
import logging
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BankingVisualizations:
    """Generate visualizations for banking app insights"""
    
    def __init__(self, style: str = 'seaborn'):
        plt.style.use(style)
        self.colors = {
            'cbe': '#1f77b4',  # Blue
            'boa': '#ff7f0e',  # Orange
            'dashen': '#2ca02c',  # Green
            'positive': '#4caf50',  # Green
            'negative': '#f44336',  # Red
            'neutral': '#ffc107'  # Yellow
        }
    
    def create_sentiment_distribution(self, df: pd.DataFrame) -> Figure:
        """Create sentiment distribution plot"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Sentiment Distribution by Bank', fontsize=16, fontweight='bold')
        
        banks = df['bank_name'].unique()
        
        for idx, bank in enumerate(banks):
            ax = axes[idx]
            bank_data = df[df['bank_name'] == bank]
            
            sentiment_counts = bank_data['sentiment_label'].value_counts()
            colors = [self.colors.get(sent, '#757575') for sent in sentiment_counts.index]
            
            wedges, texts, autotexts = ax.pie(
                sentiment_counts.values,
                labels=sentiment_counts.index,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90
            )
            
            # Improve text appearance
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title(f'{bank}\n(n={len(bank_data)})', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_rating_comparison(self, df: pd.DataFrame) -> Figure:
        """Create rating comparison plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Box plot for ratings
        bank_order = df.groupby('bank_name')['rating'].median().sort_values(ascending=False).index
        
        sns.boxplot(
            data=df,
            x='bank_name',
            y='rating',
            order=bank_order,
            palette=[self.colors.get(b.lower(), '#757575') for b in bank_order],
            ax=ax1
        )
        
        ax1.set_title('Rating Distribution by Bank', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Bank')
        ax1.set_ylabel('Rating (1-5)')
        ax1.grid(True, alpha=0.3)
        
        # Bar plot for average ratings
        avg_ratings = df.groupby('bank_name')['rating'].mean().sort_values(ascending=False)
        
        bars = ax2.bar(
            avg_ratings.index,
            avg_ratings.values,
            color=[self.colors.get(b.lower(), '#757575') for b in avg_ratings.index]
        )
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.05,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontweight='bold'
            )
        
        ax2.set_title('Average Ratings by Bank', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Bank')
        ax2.set_ylabel('Average Rating')
        ax2.set_ylim(0, 5.5)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def create_sentiment_trend(self, trends_df: pd.DataFrame) -> Figure:
        """Create sentiment trend over time"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert period to datetime for plotting
        trends_df['date'] = trends_df['month'].dt.to_timestamp()
        
        for bank in trends_df['bank_name'].unique():
            bank_data = trends_df[trends_df['bank_name'] == bank].sort_values('date')
            
            ax.plot(
                bank_data['date'],
                bank_data['positive_rate'],
                label=f'{bank} (Positive)',
                marker='o',
                color=self.colors.get(bank.lower(), '#757575'),
                linewidth=2
            )
            
            ax.plot(
                bank_data['date'],
                bank_data['negative_rate'],
                label=f'{bank} (Negative)',
                marker='s',
                color=self.colors.get(bank.lower(), '#757575'),
                linestyle='--',
                linewidth=2,
                alpha=0.7
            )
        
        ax.set_title('Sentiment Trends Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Percentage (%)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def create_word_cloud(self, df: pd.DataFrame, bank_name: str, sentiment: str = None) -> Figure:
        """Create word cloud for specific bank and sentiment"""
        # Filter data
        mask = df['bank_name'] == bank_name
        if sentiment:
            mask &= df['sentiment_label'] == sentiment
        
        text_data = df[mask]['review_text'].dropna()
        
        if len(text_data) == 0:
            logger.warning(f"No data for {bank_name} with sentiment {sentiment}")
            return None
        
        # Combine all text
        text = ' '.join(text_data.astype(str))
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            contour_width=3,
            contour_color='steelblue',
            colormap='viridis'
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        title = f'Word Cloud: {bank_name}'
        if sentiment:
            title += f' ({sentiment} reviews)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_theme_analysis(self, df: pd.DataFrame) -> Figure:
        """Create theme analysis visualization"""
        # Count themes by bank and sentiment
        theme_counts = df.groupby(['bank_name', 'theme', 'sentiment_label']).size().unstack(fill_value=0)
        
        # Get top themes for each bank
        top_themes = {}
        for bank in df['bank_name'].unique():
            bank_themes = theme_counts.xs(bank)
            top_themes[bank] = bank_themes.sum(axis=1).nlargest(5).index.tolist()
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        for idx, (bank, themes) in enumerate(top_themes.items()):
            ax = axes[idx]
            
            # Get data for this bank's top themes
            bank_data = theme_counts.xs(bank).loc[themes]
            
            # Create stacked bar chart
            x = np.arange(len(themes))
            width = 0.6
            
            bottom = np.zeros(len(themes))
            for sentiment in ['positive', 'neutral', 'negative']:
                if sentiment in bank_data.columns:
                    values = bank_data[sentiment].values
                    ax.bar(x, values, width, bottom=bottom, 
                          label=sentiment.capitalize(),
                          color=self.colors[sentiment])
                    bottom += values
            
            ax.set_title(f'{bank}: Top Themes Analysis', fontsize=12, fontweight='bold')
            ax.set_xlabel('Theme')
            ax.set_ylabel('Number of Reviews')
            ax.set_xticks(x)
            ax.set_xticklabels(themes, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Theme Analysis by Bank and Sentiment', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_drivers_pain_points(self, drivers_data: Dict) -> Figure:
        """Create visualization for drivers and pain points"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Satisfaction Drivers and Pain Points Analysis', 
                    fontsize=16, fontweight='bold')
        
        banks = list(drivers_data.keys())
        
        for idx, bank in enumerate(banks):
            # Drivers
            ax_driver = axes[0, idx]
            drivers = drivers_data[bank]['drivers'][:5]  # Top 5 drivers
            
            if drivers:
                driver_categories = [d['category'].replace('_', ' ').title() for d in drivers]
                driver_counts = [d['count'] for d in drivers]
                
                bars = ax_driver.barh(driver_categories, driver_counts, 
                                     color=self.colors.get(bank.lower(), '#757575'))
                
                # Add count labels
                for bar in bars:
                    width = bar.get_width()
                    ax_driver.text(width + 1, bar.get_y() + bar.get_height()/2,
                                 f'{int(width)}', va='center')
                
                ax_driver.set_title(f'{bank}: Top Drivers', fontweight='bold')
                ax_driver.set_xlabel('Number of Mentions')
                ax_driver.grid(True, alpha=0.3, axis='x')
            
            # Pain points
            ax_pain = axes[1, idx]
            pain_points = drivers_data[bank]['pain_points'][:5]  # Top 5 pain points
            
            if pain_points:
                pain_categories = [p['category'].replace('_', ' ').title() for p in pain_points]
                pain_counts = [p['count'] for p in pain_points]
                
                bars = ax_pain.barh(pain_categories, pain_counts, 
                                   color='#f44336')  # Red for pain points
                
                # Add count labels
                for bar in bars:
                    width = bar.get_width()
                    ax_pain.text(width + 1, bar.get_y() + bar.get_height()/2,
                               f'{int(width)}', va='center')
                
                ax_pain.set_title(f'{bank}: Top Pain Points', fontweight='bold')
                ax_pain.set_xlabel('Number of Mentions')
                ax_pain.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def save_all_visualizations(self, df: pd.DataFrame, trends_df: pd.DataFrame, 
                               drivers_data: Dict, output_dir: str = './reports/plots'):
        """Generate and save all visualizations"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        visualizations = [
            ('sentiment_distribution.png', self.create_sentiment_distribution(df)),
            ('rating_comparison.png', self.create_rating_comparison(df)),
            ('sentiment_trends.png', self.create_sentiment_trend(trends_df)),
            ('theme_analysis.png', self.create_theme_analysis(df)),
            ('drivers_pain_points.png', self.create_drivers_pain_points(drivers_data))
        ]
        
        # Add word clouds for each bank
        for bank in df['bank_name'].unique():
            for sentiment in ['positive', 'negative']:
                fig = self.create_word_cloud(df, bank, sentiment)
                if fig:
                    filename = f'wordcloud_{bank.lower()}_{sentiment}.png'
                    filepath = os.path.join(output_dir, filename)
                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"Saved {filename}")
        
        # Save main visualizations
        for filename, fig in visualizations:
            if fig:
                filepath = os.path.join(output_dir, filename)
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved {filename}")
        
        logger.info(f"All visualizations saved to {output_dir}")