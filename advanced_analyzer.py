from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

class AdvancedAnalyzer:
    """
    Handles advanced analysis using BERTopic to discover topics from text.
    """
    def __init__(self, text_series):
        self.documents = text_series.dropna().tolist()
        self.model = None

    def find_topics(self, max_topics=20):
        """
        Performs BERTopic analysis to discover a controlled number of topics.
        """
        if not self.documents:
            print("Warning: No documents to analyze. Skipping BERTopic.")
            return None

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        stop_words = list(stopwords.words('english'))
        vectorizer_model = CountVectorizer(stop_words=stop_words)

        self.model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            verbose=False,
            min_topic_size=5,
            nr_topics=max_topics
        )
        
        topics, _ = self.model.fit_transform(self.documents)
        
        formatted_topics = []
        
        # Iterate over the actual topic IDs from the 'Topic' column to avoid index errors
        for topic_id in self.model.get_topic_freq()['Topic']:
            if topic_id == -1: continue # Skip outliers topic

            topic_info_df = self.model.get_topic_info(topic_id)
            
            # Safety check to ensure the topic info is not empty
            if topic_info_df.empty:
                continue

            name = topic_info_df.iloc[0]['Name']
            words = [word for word, _ in self.model.get_topic(topic_id)][:10]
            formatted_topics.append((name, words))
        
        return formatted_topics

