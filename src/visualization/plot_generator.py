import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from matplotlib.figure import Figure
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging to show warnings/errors in console
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class BankingVisualizations:
    """Generate visualizations for banking app insights"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizations with style settings.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.colors = {
            'CBE': '#1f77b4',      # Blue
            'BOA': '#ff7f0e',      # Orange
            'Dashen': '#2ca02c',   # Green
            'Commercial Bank of Ethiopia': '#1f77b4',
            'Bank of Abyssinia': '#ff7f0e',
            'Amhara Bank': '#9467bd',  # Purple
            'Awash Bank': '#8c564b',   # Brown
            'positive': '#4caf50',     # Green
            'negative': '#f44336',     # Red
            'neutral': '#ffc107'       # Yellow
        }
        
    def create_sentiment_distribution(self, df: pd.DataFrame) -> Figure:
        """
        Create sentiment distribution plot.
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            Matplotlib Figure object
        """
        if 'bank_name' not in df.columns or 'sentiment_label' not in df.columns:
            logger.error("Missing required columns for sentiment distribution")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Sentiment Distribution by Bank', fontsize=16, fontweight='bold')
        
        banks = df['bank_name'].unique()[:3]  # Limit to 3 banks
        
        for idx, bank in enumerate(banks):
            ax = axes[idx]
            bank_data = df[df['bank_name'] == bank]
            
            sentiment_counts = bank_data['sentiment_label'].value_counts()
            colors = [self.colors.get(sent.lower(), '#757575') for sent in sentiment_counts.index]
            
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
        """
        Create rating comparison plot.
        
        Args:
            df: DataFrame with rating data
            
        Returns:
            Matplotlib Figure object
        """
        if 'bank_name' not in df.columns or 'rating' not in df.columns:
            logger.error("Missing required columns for rating comparison")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Box plot for ratings
        bank_order = df.groupby('bank_name')['rating'].median().sort_values(ascending=False).index
        
        sns.boxplot(
            data=df,
            x='bank_name',
            y='rating',
            order=bank_order,
            palette=[self.colors.get(b, '#757575') for b in bank_order],
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
            color=[self.colors.get(b, '#757575') for b in avg_ratings.index]
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
    
    def create_drivers_pain_points_chart(self, drivers_data: Dict[str, Dict]) -> Figure:
        """
        Create visualization for drivers and pain points.
        
        Args:
            drivers_data: Dictionary containing drivers and pain points
            
        Returns:
            Matplotlib Figure object
        """
        if not drivers_data:
            logger.error("No drivers data provided")
            return None
        
        banks = list(drivers_data.keys())
        n_banks = len(banks)
        
        fig, axes = plt.subplots(2, n_banks if n_banks > 1 else 1, figsize=(5 * max(n_banks, 1), 8))
        fig.suptitle('Satisfaction Drivers and Pain Points Analysis', 
                    fontsize=16, fontweight='bold')
        
        if n_banks == 1:
            axes = np.array([axes]) if n_banks == 1 and len(axes) == 2 else axes.reshape(2, 1)
        
        for idx, bank in enumerate(banks):
            # Drivers subplot
            ax1 = axes[0] if n_banks == 1 else axes[0, idx]
            drivers = drivers_data[bank].get('drivers', [])
            
            if drivers:
                categories = [d['category'].replace('_', ' ').title() for d in drivers]
                counts = [d['count'] for d in drivers]
                
                bars = ax1.barh(categories, counts, color=self.colors.get(bank, '#757575'))
                ax1.set_title(f'{bank}: Top Drivers', fontweight='bold')
                ax1.set_xlabel('Number of Mentions')
                ax1.grid(True, alpha=0.3, axis='x')
                
                # Add count labels
                for bar in bars:
                    width = bar.get_width()
                    ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                            f'{int(width)}', va='center')
            else:
                ax1.text(0.5, 0.5, 'No drivers identified', 
                        ha='center', va='center', fontsize=12)
                ax1.axis('off')
            
            # Pain points subplot
            ax2 = axes[1] if n_banks == 1 else axes[1, idx]
            pain_points = drivers_data[bank].get('pain_points', [])
            
            if pain_points:
                categories = [p['category'].replace('_', ' ').title() for p in pain_points]
                counts = [p['count'] for p in pain_points]
                
                bars = ax2.barh(categories, counts, color='#f44336')  # Red for pain points
                ax2.set_title(f'{bank}: Top Pain Points', fontweight='bold')
                ax2.set_xlabel('Number of Mentions')
                ax2.grid(True, alpha=0.3, axis='x')
                
                # Add count labels
                for bar in bars:
                    width = bar.get_width()
                    ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                            f'{int(width)}', va='center')
            else:
                ax2.text(0.5, 0.5, 'No pain points identified', 
                        ha='center', va='center', fontsize=12)
                ax2.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_sentiment_trend_chart(self, df: pd.DataFrame) -> Figure:
        """
        Create sentiment trend over time chart.
        
        Args:
            df: DataFrame with date and sentiment data
            
        Returns:
            Matplotlib Figure object
        """
        if 'review_date' not in df.columns or 'sentiment_score' not in df.columns:
            logger.error("Missing required columns for sentiment trend")
            return None
        
        try:
            df['review_date'] = pd.to_datetime(df['review_date'])
            df['month'] = df['review_date'].dt.to_period('M').dt.to_timestamp()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if 'bank_name' in df.columns:
                # Plot sentiment trends for each bank
                for bank in df['bank_name'].unique()[:3]:  # Limit to 3 banks
                    bank_data = df[df['bank_name'] == bank]
                    monthly_sentiment = bank_data.groupby('month')['sentiment_score'].mean()
                    
                    ax.plot(monthly_sentiment.index, monthly_sentiment.values,
                           marker='o', linewidth=2, label=bank,
                           color=self.colors.get(bank, '#757575'))
            else:
                # Overall sentiment trend
                monthly_sentiment = df.groupby('month')['sentiment_score'].mean()
                ax.plot(monthly_sentiment.index, monthly_sentiment.values,
                       marker='o', linewidth=2, color='blue')
            
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Neutral')
            ax.set_title('Sentiment Trends Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Average Sentiment Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating sentiment trend chart: {e}")
            return None
    
    def create_word_cloud(self, df: pd.DataFrame, bank_name: str, 
                         sentiment: Optional[str] = None) -> Figure:
        """
        Create word cloud for specific bank and sentiment.
        
        Args:
            df: DataFrame with review text
            bank_name: Name of the bank
            sentiment: Optional sentiment filter (expects 'positive', 'negative', 'neutral')
            
        Returns:
            Matplotlib Figure object
        """
        try:
            from wordcloud import WordCloud
        except ImportError:
            logger.error("WordCloud library not installed")
            return None
        
        if 'review_text' not in df.columns:
            logger.error("Missing review_text column")
            return None
        
        # Filter data
        mask = df['bank_name'] == bank_name
        if sentiment:
            # Ensure sentiment_label is lowercase for matching
            mask &= df['sentiment_label'] == sentiment
        
        text_data = df[mask]['review_text'].dropna()
        
        if len(text_data) == 0:
            logger.warning(f"No data for {bank_name} with sentiment '{sentiment}'")
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
    
    def create_recommendations_chart(self, recommendations: Dict[str, List[Dict]]) -> Figure:
        """
        Create visualization of recommendations by priority.
        
        Args:
            recommendations: Dictionary of recommendations by bank
            
        Returns:
            Matplotlib Figure object
        """
        if not recommendations:
            logger.error("No recommendations data provided")
            return None
        
        # Prepare data for visualization
        rec_data = []
        for bank, recs in recommendations.items():
            for rec in recs:
                rec_data.append({
                    'Bank': bank,
                    'Category': rec.get('category', 'general'),
                    'Priority': rec.get('priority', 'MEDIUM'),
                    'Type': rec.get('type', 'general')
                })
        
        if not rec_data:
            return None
        
        rec_df = pd.DataFrame(rec_data)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Recommendations Analysis', fontsize=16, fontweight='bold')
        
        # 1. Recommendations by priority and bank
        ax1 = axes[0]
        priority_counts = rec_df.groupby(['Bank', 'Priority']).size().unstack(fill_value=0)
        
        x = np.arange(len(priority_counts.index))
        width = 0.25
        
        priorities = ['HIGH', 'MEDIUM', 'LOW']
        colors = ['#f44336', '#ff9800', '#4caf50']  # Red, Orange, Green
        
        for i, priority in enumerate(priorities):
            if priority in priority_counts.columns:
                offset = width * i
                ax1.bar(x + offset, priority_counts[priority], width, 
                        label=priority, color=colors[i])
        
        ax1.set_title('Recommendations by Priority Level', fontweight='bold')
        ax1.set_xlabel('Bank')
        ax1.set_ylabel('Number of Recommendations')
        ax1.set_xticks(x + width, priority_counts.index)
        ax1.legend(title='Priority')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Recommendations by category
        ax2 = axes[1]
        category_counts = rec_df['Category'].value_counts().head(10)
        
        bars = ax2.barh(category_counts.index, category_counts.values)
        ax2.set_title('Top Recommendation Categories', fontweight='bold')
        ax2.set_xlabel('Number of Recommendations')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{int(width)}', va='center')
        
        plt.tight_layout()
        return fig
    
    def save_visualization(self, fig: Figure, filename: str,
                          output_dir: str = r'C:\Users\admin\sentiment-analysis-week2\src\reports') -> bool:
        """
        Save visualization to file.
        """
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            filepath = Path(output_dir) / filename
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved visualization: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save visualization {filename}: {e}")
            return False
    
    def generate_all_visualizations(self, df: pd.DataFrame, insights: Dict[str, Any],
                                   output_dir: str = r'C:\Users\admin\sentiment-analysis-week2\src\reports') -> Dict[str, bool]:
        """
        Generate all visualizations and save to files.
        """
        logger.info(f"Generating all visualizations to {output_dir}")
        results = {}

        # 1. Sentiment distribution
        fig1 = self.create_sentiment_distribution(df)
        if fig1:
            results['sentiment_distribution'] = self.save_visualization(fig1, 'sentiment_distribution.png', output_dir)

        # 2. Rating comparison
        fig2 = self.create_rating_comparison(df)
        if fig2:
            results['rating_comparison'] = self.save_visualization(fig2, 'rating_comparison.png', output_dir)

        # 3. Drivers and pain points
        drivers_data = insights.get('drivers_pain_points', {})
        fig3 = self.create_drivers_pain_points_chart(drivers_data)
        if fig3:
            results['drivers_pain_points'] = self.save_visualization(fig3, 'drivers_pain_points.png', output_dir)

        # 4. Sentiment trend
        fig4 = self.create_sentiment_trend_chart(df)
        if fig4:
            results['sentiment_trend'] = self.save_visualization(fig4, 'sentiment_trend.png', output_dir)

        # 5. Recommendations chart
        recommendations = insights.get('recommendations', {})
        fig5 = self.create_recommendations_chart(recommendations)
        if fig5:
            results['recommendations'] = self.save_visualization(fig5, 'recommendations.png', output_dir)

        # 6. Word clouds for each bank
        if 'bank_name' in df.columns:
            for bank in df['bank_name'].unique()[:3]:
                # Positive
                fig_pos = self.create_word_cloud(df, bank, 'positive')
                if fig_pos:
                    results[f'wordcloud_{bank}_positive'] = self.save_visualization(fig_pos, f'wordcloud_{bank}_positive.png', output_dir)
                # Negative
                fig_neg = self.create_word_cloud(df, bank, 'negative')
                if fig_neg:
                    results[f'wordcloud_{bank}_negative'] = self.save_visualization(fig_neg, f'wordcloud_{bank}_negative.png', output_dir)
                # Neutral (optional)
                fig_neu = self.create_word_cloud(df, bank, 'neutral')
                if fig_neu:
                    results[f'wordcloud_{bank}_neutral'] = self.save_visualization(fig_neu, f'wordcloud_{bank}_neutral.png', output_dir)

        return results


# Standalone visualization functions
def plot_rating_vs_sentiment(df: pd.DataFrame, bank_name: Optional[str] = None) -> Figure:
    """
    Plot rating vs sentiment correlation.
    
    Args:
        df: DataFrame with rating and sentiment data
        bank_name: Optional bank filter
        
    Returns:
        Matplotlib Figure object
    """
    if 'rating' not in df.columns or 'sentiment_score' not in df.columns:
        logger.error("Missing required columns for rating vs sentiment plot")
        return None
    
    # Filter by bank if specified
    if bank_name and 'bank_name' in df.columns:
        plot_data = df[df['bank_name'] == bank_name]
        title = f'Rating vs Sentiment: {bank_name}'
    else:
        plot_data = df
        title = 'Rating vs Sentiment Correlation'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(plot_data['rating'], plot_data['sentiment_score'], 
                        alpha=0.5, s=20)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Sentiment Score')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    try:
        z = np.polyfit(plot_data['rating'], plot_data['sentiment_score'], 1)
        p = np.poly1d(z)
        ax.plot(plot_data['rating'].sort_values(), p(plot_data['rating'].sort_values()), 
                "r--", alpha=0.8, label=f'Trend: y = {z[0]:.3f}x + {z[1]:.3f}')
        ax.legend()
    except Exception:
        pass
    
    plt.tight_layout()
    return fig


def plot_monthly_review_volume(df: pd.DataFrame) -> Figure:
    """
    Plot monthly review volume over time.
    
    Args:
        df: DataFrame with date data
        
    Returns:
        Matplotlib Figure object
    """
    if 'review_date' not in df.columns:
        logger.error("Missing review_date column")
        return None
    
    try:
        df['review_date'] = pd.to_datetime(df['review_date'])
        df['month'] = df['review_date'].dt.to_period('M').dt.to_timestamp()
        
        monthly_counts = df.groupby('month').size()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(monthly_counts.index, monthly_counts.values, 
                marker='o', linewidth=2)
        ax.set_title('Monthly Review Volume', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Reviews')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Error creating monthly review volume plot: {e}")
        return None


if __name__ == "__main__":
    # Load your data
    df = pd.read_csv(r"C:\Users\admin\sentiment-analysis-week2\data\processed_data\reviews_with_sentiment.csv")
    
    # 🔑 CRITICAL FIX: Normalize sentiment_label to lowercase to match filtering logic
    if 'sentiment_label' in df.columns:
        df['sentiment_label'] = df['sentiment_label'].str.lower()
    else:
        logger.error("Column 'sentiment_label' not found in data.")
        exit(1)
    
    # Provide insights (can be empty, but structure must exist)
    insights = {
        'drivers_pain_points': {},
        'recommendations': {}
    }
    
    viz = BankingVisualizations()
    results = viz.generate_all_visualizations(df, insights)
    print("\n✅ Visualization Results:")
    for name, success in results.items():
        print(f"  - {name}: {'Saved' if success else 'Failed'}")