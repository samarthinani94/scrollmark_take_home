import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Visualizer:
    """
    Handles the creation and saving of all plots.
    """
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Set a consistent, clean style for all plots
        sns.set_theme(style="whitegrid", palette="viridis")
        plt.rcParams['figure.dpi'] = 100

    def _save_plot(self, filename):
        """Helper function to save plots and return the filename."""
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"Chart saved: {filepath}")
        return filename

    def plot_avg_comments_by_day(self, data):
        """Plots the average number of comments per post for each day of the week."""
        plt.figure(figsize=(10, 6))
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        data.reindex(day_order).plot(kind='bar', color=sns.color_palette("viridis", 7))
        plt.title('Average Comments per Post by Day of the Week', fontsize=16, fontweight='bold')
        plt.xlabel('Day of the Week', fontsize=12)
        plt.ylabel('Average Comments per Post', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        return self._save_plot("avg_comments_by_day.png")

    def plot_avg_posts_per_day_by_hour(self, data):
        """Plots the average number of posts per day for each hour."""
        plt.figure(figsize=(12, 6))
        data.plot(kind='bar', color=sns.color_palette("magma", 24))
        plt.title('Average Posts per Day by Hour of Day (PST)', fontsize=16, fontweight='bold')
        plt.xlabel('Hour of Day (24-hour format)', fontsize=12)
        plt.ylabel('Average Number of Posts', fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        return self._save_plot("avg_posts_per_day_by_hour.png")

    def plot_avg_positive_neutral_comments_by_hour(self, data):
        """Plots the average number of positive+neutral comments per post for each hour of the day."""
        plt.figure(figsize=(12, 6))
        data.sort_index().plot(kind='bar', color=sns.color_palette("plasma", 24))
        plt.title('Average Positive+Neutral Comments per Post by Hour (PST)', fontsize=16, fontweight='bold')
        plt.xlabel('Hour of Day (24-hour format)', fontsize=12)
        plt.ylabel('Average Positive+Neutral Comments per Post', fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        return self._save_plot("avg_positive_neutral_comments_by_hour.png")

    def plot_emoji_sentiment(self, sentiments):
        """Plots the distribution of emoji sentiments."""
        sentiment_counts = {k: len(v) for k, v in sentiments.items()}
        df = pd.DataFrame([sentiment_counts])
        
        plt.figure(figsize=(8, 6))
        df.plot(kind='bar', stacked=True, color={'positive': '#2ECC71', 'neutral': '#F1C40F', 'negative': '#E74C3C'})
        plt.title('Emoji Sentiment Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Sentiment', fontsize=12)
        plt.ylabel('Number of Unique Emojis', fontsize=12)
        plt.xticks([]) # Remove x-axis labels
        plt.legend(title='Sentiment')
        return self._save_plot("emoji_sentiment_distribution.png")


