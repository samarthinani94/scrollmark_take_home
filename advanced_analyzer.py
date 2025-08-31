import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

class AdvancedAnalyzer:
    """
    Performs advanced topic modeling using BERTopic.
    """
    def __init__(self, text_series):
        self.text_series = text_series.dropna()
        # Using a lightweight, efficient model for faster processing
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = BERTopic(embedding_model=embedding_model, verbose=False)

    def find_topics(self, max_topics=20):
        """
        Finds topics in the text series and returns them.
        """
        # BERTopic requires a list of strings
        docs = self.text_series.tolist()
        self.model.fit(docs)

        # Merge similar topics to reduce redundancy and meet the max_topics limit
        if len(self.model.get_topic_info()) > max_topics:
            self.model.reduce_topics(docs, nr_topics=max_topics)

        formatted_topics = []
        # Iterate through topics safely
        for topic_info in self.model.get_topic_info().to_dict('records'):
            topic_id = topic_info['Topic']
            if topic_id == -1: continue # Skip outliers

            # Get the human-readable name and top 10 representative words
            name = topic_info['Name']
            words = [word for word, _ in self.model.get_topic(topic_id)][:10]
            formatted_topics.append((name, words))
        
        return formatted_topics

