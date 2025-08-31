import pandas as pd
from collections import Counter
import demoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class EmojiAnalyzer:
    """
    Analyzes emojis from a series of text comments.
    """
    def __init__(self, text_series):
        demoji.download_codes()
        # Find all emojis across the entire series of comments using findall
        all_emojis = []
        for text in text_series.dropna():
            all_emojis.extend(demoji.findall(text).keys())
            
        self.emojis = all_emojis
        self.unique_emojis = sorted(list(set(all_emojis)))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def get_top_emojis(self, top_k=25):
        """
        Finds the most frequently used emojis.
        """
        return Counter(self.emojis).most_common(top_k)

    def analyze_emoji_sentiments(self):
        """
        Categorizes each unique emoji by its sentiment score.
        """
        sentiments = {'positive': [], 'negative': [], 'neutral': []}
        sentiment_map = {'positive': [], 'negative': [], 'neutral': []}

        for emoji in self.unique_emojis:
            score = self.sentiment_analyzer.polarity_scores(emoji)['compound']
            # Correctly get the text description using findall
            emoji_descriptions = demoji.findall(emoji)
            demojized = emoji_descriptions.get(emoji, 'unknown') # Use .get for safety
            
            if score > 0.05:
                sentiments['positive'].append(emoji)
                sentiment_map['positive'].append(f"{emoji} (:{demojized}:)")
            elif score < -0.05:
                sentiments['negative'].append(emoji)
                sentiment_map['negative'].append(f"{emoji} (:{demojized}:)")
            else:
                sentiments['neutral'].append(emoji)
                sentiment_map['neutral'].append(f"{emoji} (:{demojized}:)")
        
        return sentiments, sentiment_map

