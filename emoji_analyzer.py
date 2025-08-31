import pandas as pd
import emoji
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class EmojiAnalyzer:
    """
    Handles all emoji extraction and sentiment analysis.
    """
    def __init__(self, df):
        self.df = df.copy()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def get_top_n_emojis(self, n=25):
        """Extracts and counts all emojis from the comment text."""
        all_emojis = []
        for text in self.df['comment_text'].dropna():
            all_emojis.extend([e['emoji'] for e in emoji.emoji_list(text)])
        return Counter(all_emojis).most_common(n)

    def analyze_comment_sentiment(self):
        """
        Analyzes the sentiment of each comment and returns the breakdown.
        """
        sentiments = self.df['comment_text'].dropna().apply(
            lambda text: self.sentiment_analyzer.polarity_scores(text)['compound']
        )
        
        breakdown = {
            'Positive': (sentiments > 0.05).sum(),
            'Neutral': ((sentiments >= -0.05) & (sentiments <= 0.05)).sum(),
            'Negative': (sentiments < -0.05).sum()
        }
        return breakdown

    def correlate_engagement_with_captions(self, caption_topics, caption_bigrams):
        """
        Creates tables correlating caption features (topics, bigrams) with engagement.
        """
        # --- Bigram Correlation ---
        bigram_list = [gram for gram, count in caption_bigrams]
        bigram_engagement_data = []

        for bigram in bigram_list:
            # Find all posts whose captions contain this bigram
            relevant_posts = self.df[self.df['cleaned_caption'].str.contains(bigram, na=False)]
            if not relevant_posts.empty:
                comments_per_post = relevant_posts.groupby('media_id').size().mean()
                
                # Calculate positive sentiment for these posts
                sentiments = relevant_posts['comment_text'].dropna().apply(lambda text: self.sentiment_analyzer.polarity_scores(text)['compound'])
                positive_comments = (sentiments > 0.05).sum()
                total_posts = relevant_posts['media_id'].nunique()
                positive_per_post = positive_comments / total_posts if total_posts > 0 else 0
                
                bigram_engagement_data.append({
                    "Caption Bigram": bigram,
                    "Avg Comments/Post": round(comments_per_post, 2),
                    "Avg Positive Comments/Post": round(positive_per_post, 2)
                })

        # --- Topic Correlation ---
        topic_engagement_data = []
        if caption_topics:
            for name, words in caption_topics:
                # Use the top word of the topic to find relevant posts
                # A more advanced approach would use topic-per-document distributions
                # but this is a fast and effective approximation.
                primary_word = words[0]
                relevant_posts = self.df[self.df['cleaned_caption'].str.contains(primary_word, na=False)]
                if not relevant_posts.empty:
                    comments_per_post = relevant_posts.groupby('media_id').size().mean()
                    sentiments = relevant_posts['comment_text'].dropna().apply(lambda text: self.sentiment_analyzer.polarity_scores(text)['compound'])
                    positive_comments = (sentiments > 0.05).sum()
                    total_posts = relevant_posts['media_id'].nunique()
                    positive_per_post = positive_comments / total_posts if total_posts > 0 else 0

                    topic_engagement_data.append({
                        "Caption Topic": name,
                        "Avg Comments/Post": round(comments_per_post, 2),
                        "Avg Positive Comments/Post": round(positive_per_post, 2)
                    })

        return {
            "bigrams": pd.DataFrame(bigram_engagement_data),
            "topics": pd.DataFrame(topic_engagement_data)
        }

