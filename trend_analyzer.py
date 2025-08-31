import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class TrendAnalyzer:
    """
    Performs trend analysis on the cleaned data, including n-grams and time-based metrics.
    """
    def __init__(self, df):
        self.df = df.copy()
        # Ensure timestamp is in PST
        self.df['timestamp_pst'] = self.df['timestamp'].dt.tz_convert('America/Los_Angeles')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self._precalculate_engagement_metrics()
        # Create a dataframe with one row per unique post for per-post metric analysis
        self.posts_df = self.df.drop_duplicates(subset=['media_id']).copy()
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))


    def _precalculate_engagement_metrics(self):
        """
        Pre-calculates sentiment and other per-post metrics to improve efficiency.
        """
        self.df['sentiment_score'] = self.df['comment_text'].apply(
            lambda x: self.sentiment_analyzer.polarity_scores(str(x))['compound']
        )
        self.df['is_positive_or_neutral'] = (self.df['sentiment_score'] >= -0.05).astype(int)

        post_metrics = self.df.groupby('media_id').agg(
            total_comments=('comment_text', 'count'),
            positive_neutral_comments=('is_positive_or_neutral', 'sum')
        ).reset_index()

        self.df = pd.merge(self.df, post_metrics, on='media_id', how='left')

    def get_top_ngrams(self, text_source='comments', n=2, top_k=25):
        return self.get_top_ngrams_for_series(
            self.df[self._get_column_for_source(text_source)], n=n, top_k=top_k
        )
        
    def _get_column_for_source(self, text_source):
        """Helper to get the correct column name."""
        return {'comments': 'cleaned_comment', 'captions': 'cleaned_caption'}[text_source]

    def get_top_ngrams_for_series(self, text_series, n, top_k):
        """Helper to get n-grams for a specific pandas Series."""
        domain_stop_words = {'treehut', 'user_tagged', 'tree', 'hut'}
        combined_stop_words = self.stop_words.union(domain_stop_words)
        
        tokens = [
            token for text in text_series.dropna()
            for token in text.split() 
            if token not in combined_stop_words
        ]
        
        n_grams = ngrams(tokens, n)
        return Counter(n_grams).most_common(top_k)

    def get_avg_positive_neutral_comments_per_post_by_day(self):
        """Calculates the average number of positive+neutral comments per post for each day."""
        daily_stats = self.posts_df.groupby(self.posts_df['timestamp_pst'].dt.day_name())
        return daily_stats['positive_neutral_comments'].mean()

    def get_avg_posts_per_day_by_hour(self):
        """Calculates the average number of posts published per day, for each hour."""
        posts_df = self.df.drop_duplicates(subset=['media_id']).copy()
        posts_df['date'] = posts_df['timestamp_pst'].dt.date
        posts_df['hour'] = posts_df['timestamp_pst'].dt.hour
        
        hourly_counts = posts_df.groupby(['date', 'hour']).size().reset_index(name='post_count')
        avg_hourly = hourly_counts.groupby('hour')['post_count'].mean()
        return avg_hourly

    def get_avg_positive_neutral_comments_per_post_by_hour(self):
        """Calculates the average number of positive+neutral comments per post for each hour."""
        hourly_stats = self.posts_df.groupby(self.posts_df['timestamp_pst'].dt.hour)
        return hourly_stats['positive_neutral_comments'].mean()

    def calculate_engagement_by_caption_feature(self, topic_ids, topic_info):
        """Calculates engagement metrics correlated with caption topics and bigrams."""
        self.df['topic_id'] = topic_ids

        # 1. Engagement by Topic
        topic_engagement = self.df.groupby('topic_id').agg(
            avg_comments_per_post=('total_comments', 'mean')
        ).reset_index()

        topic_info_renamed = topic_info.rename(columns={'Topic': 'topic_id', 'Name': 'caption_topic'})
        topic_engagement = pd.merge(topic_engagement, topic_info_renamed[['topic_id', 'caption_topic']], on='topic_id')
        topic_engagement = topic_engagement[topic_engagement['topic_id'] != -1]
        topic_engagement = topic_engagement[['caption_topic', 'avg_comments_per_post']]
        topic_engagement.columns = ['Caption Topic', 'Avg Comments/Post']
        
        # 2. Engagement by Caption Bigram
        top_caption_bigrams = self.get_top_ngrams('captions', n=2, top_k=25)
        bigram_engagement_data = []

        for bigram, _ in top_caption_bigrams:
            bigram_str = " ".join(bigram)
            relevant_posts = self.df[self.df['cleaned_caption'].str.contains(bigram_str, na=False)]
            
            if not relevant_posts.empty:
                unique_posts = relevant_posts.drop_duplicates(subset=['media_id'])
                avg_comments = unique_posts['total_comments'].mean()
                bigram_engagement_data.append((bigram_str, avg_comments))

        bigram_df = pd.DataFrame(bigram_engagement_data, columns=['Caption Bigram', 'Avg Comments/Post'])
        
        bigram_df['Avg Comments/Post'] = bigram_df['Avg Comments/Post'].round(2)
        topic_engagement['Avg Comments/Post'] = topic_engagement['Avg Comments/Post'].round(2)

        return {'topics': topic_engagement, 'bigrams': bigram_df}
        
    def create_topic_trigram_table(self, topic_ids, topic_info):
        """Creates a table correlating caption topics with top user comment trigrams."""
        self.df['topic_id'] = topic_ids
        df_topics = self.df[self.df['topic_id'] != -1].copy()

        topic_info_renamed = topic_info.rename(columns={'Topic': 'topic_id', 'Name': 'caption_topic'})
        df_topics = pd.merge(df_topics, topic_info_renamed[['topic_id', 'caption_topic']], on='topic_id')
        
        results = []
        for topic_name, group in df_topics.groupby('caption_topic'):
            avg_comments = group.drop_duplicates(subset=['media_id'])['total_comments'].mean()
            top_trigrams = self.get_top_ngrams_for_series(group['cleaned_comment'], n=3, top_k=5)
            
            trigrams_str = ", ".join([f"`{' '.join(gram)}`" for gram, count in top_trigrams]) if top_trigrams else "N/A"
            
            results.append({
                'Topic': topic_name,
                'Top 5 Comment Trigrams': trigrams_str,
                'Avg Comments/Post': avg_comments
            })
            
        result_df = pd.DataFrame(results)
        result_df['Avg Comments/Post'] = result_df['Avg Comments/Post'].round(2)
        return result_df

