import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import emoji

class DataProcessor:
    """
    Handles loading, cleaning, and preparing the social media data from a local file or URL.
    """
    def __init__(self, source_path):
        self.source_path = source_path
        self._download_nltk_data()
        self.stop_words = set(stopwords.words('english'))

    def _download_nltk_data(self):
        """Downloads necessary NLTK data if not already present."""
        try:
            stopwords.words('english')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def _clean_caption_text(self, text):
        """Cleans caption text, preserving @mentions and removing stop words. Prints debug info."""
        from collections import Counter
        if not isinstance(text, str): return ""
        text = emoji.replace_emoji(text, replace='')
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^\w\s@]', '', text)
        tokens = word_tokenize(text)
        cleaned_tokens = [
            word for word in tokens 
            if (word.isalpha() or word.startswith('@')) and word not in self.stop_words
        ]
        return " ".join(cleaned_tokens)

    def _clean_comment_text(self, text):
        """Cleans comment text, standardizing @mentions and removing stop words. Prints debug info."""
        from collections import Counter
        if not isinstance(text, str): return ""
        text = emoji.replace_emoji(text, replace='')
        text = text.lower()
        text = re.sub(r'@\w+', '@user_tagged', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^\w\s@]', '', text)
        tokens = word_tokenize(text)
        cleaned_tokens = [
            word for word in tokens 
            if (word.isalpha() or word == '@user_tagged') and word not in self.stop_words
        ]
        return " ".join(cleaned_tokens)

    def process_data(self):
        """
        Main method to load and process the dataset from a file path or Google Drive URL.
        """
        filepath_or_url = self.source_path
        
        if "drive.google.com" in self.source_path:
            file_id = self.source_path.split('/d/')[1].split('/')[0]
            filepath_or_url = f'https://drive.google.com/uc?export=download&id={file_id}'
            print("Reading data directly from Google Drive URL...")
        else:
            print(f"Reading data from local file: {self.source_path}")

        try:
            df = pd.read_csv(filepath_or_url)
        except Exception as e:
            print(f"Error reading the data source: {e}")
            return pd.DataFrame()

        df.dropna(subset=['comment_text', 'media_caption'], inplace=True)

        df['cleaned_comment'] = df['comment_text'].apply(self._clean_comment_text)
        df['cleaned_caption'] = df['media_caption'].apply(self._clean_caption_text)

        # Filter out rows that are empty after cleaning
        df = df[df['cleaned_comment'] != '']
        df = df[df['cleaned_caption'] != '']

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df.dropna(subset=['timestamp'], inplace=True)
        
        return df

