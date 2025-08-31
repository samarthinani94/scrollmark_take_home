import pandas as pd
from collections import Counter
from nltk.util import ngrams

class TrendAnalyzer:
    """
    Analyzes trends from the cleaned data, including n-grams and engagement times.
    """
    def __init__(self, df):
        self.df = df.copy()
        # Convert UTC timestamp to PST for actionable insights
        self.df['timestamp_pst'] = self.df['timestamp'].dt.tz_convert('America/Los_Angeles')

    def _get_all_words(self, source='comments'):
        """Helper to get a flat list of words from either comments or captions."""
        column = 'cleaned_comment' if source == 'comments' else 'cleaned_caption'
        return " ".join(self.df[column].dropna()).split()

    def get_top_n_words(self, source='comments', n=25):
        """Gets the top N most common words."""
        words = self._get_all_words(source)
        return Counter(words).most_common(n)

    def get_top_n_ngrams(self, source='comments', n=25, ngram_size=2):
        """Gets the top N most common n-grams (bigrams, trigrams)."""
        words = self._get_all_words(source)
        n_grams = ngrams(words, ngram_size)
        ngram_counts = Counter(n_grams)
        # Format for readability
        formatted_counts = [(" ".join(gram), count) for gram, count in ngram_counts.most_common(n)]
        return formatted_counts

    def analyze_engagement_by_day(self):
        """Calculates average comments per post for each day of the week."""
        df_copy = self.df.copy()
        df_copy['day_of_week'] = df_copy['timestamp_pst'].dt.dayofweek
        
        # Group by day and unique post (media_id), count comments
        comments_per_post_day = df_copy.groupby(['day_of_week', 'media_id']).size()
        
        # Calculate the average number of comments per post for each day
        avg_comments_per_day = comments_per_post_day.groupby('day_of_week').mean()
        return avg_comments_per_day

    def analyze_posts_by_hour(self):
        """Calculates the total number of brand posts for each hour of the day."""
        df_copy = self.df.copy()
        df_copy['hour'] = df_copy['timestamp_pst'].dt.hour
        # Count unique posts per hour
        posts_per_hour = df_copy.groupby('hour')['media_id'].nunique()
        return posts_per_hour

    def analyze_engagement_by_hour(self):
        """Calculates average comments per post for each hour of the day."""
        df_copy = self.df.copy()
        df_copy['hour'] = df_copy['timestamp_pst'].dt.hour
        comments_per_post_hour = df_copy.groupby(['hour', 'media_id']).size()
        avg_comments_per_hour = comments_per_post_hour.groupby('hour').mean()
        return avg_comments_per_hour

