from data_processor import DataProcessor
from trend_analyzer import TrendAnalyzer
from emoji_analyzer import EmojiAnalyzer
from advanced_analyzer import AdvancedAnalyzer
from visualizer import Visualizer
from report_generator import ReportGenerator

def main():
    """
    Main function to run the full data processing, analysis, and reporting pipeline.
    """
    # Define the data source - can be a local path or a Google Drive URL
    DATA_SOURCE = "https://drive.google.com/file/d/1o31ACWtmt-vjijb9QuP8Ohcr_hFlIPzS/view"

    # 1. Initialize all components
    processor = DataProcessor(DATA_SOURCE)
    visualizer = Visualizer()
    report = ReportGenerator()
    
    # 2. Process the Data
    cleaned_df = processor.process_data()
    if cleaned_df.empty:
        print("Data processing failed. Exiting pipeline.")
        return
    print("Data processing complete.")
    
    # 3. Initialize analyzers with the cleaned data
    trend_analyzer = TrendAnalyzer(cleaned_df)
    emoji_analyzer = EmojiAnalyzer(cleaned_df)
    
    # --- Perform All Analyses ---
    
    # Basic word and phrase analysis
    top_comment_words = trend_analyzer.get_top_n_words(source='comments', n=25)
    top_caption_words = trend_analyzer.get_top_n_words(source='captions', n=25)
    top_comment_bigrams = trend_analyzer.get_top_n_ngrams(source='comments', n=25, ngram_size=2)
    top_caption_bigrams = trend_analyzer.get_top_n_ngrams(source='captions', n=25, ngram_size=2)
    top_comment_trigrams = trend_analyzer.get_top_n_ngrams(source='comments', n=25, ngram_size=3)
    
    # Time-based analysis
    engagement_by_day = trend_analyzer.analyze_engagement_by_day()
    posts_by_hour = trend_analyzer.analyze_posts_by_hour()
    engagement_by_hour = trend_analyzer.analyze_engagement_by_hour()
    
    # Emoji and sentiment analysis
    top_emojis = emoji_analyzer.get_top_n_emojis(25)
    sentiment_breakdown = emoji_analyzer.analyze_comment_sentiment()
    
    # Advanced topic modeling
    print("Running BERTopic on user comments (this may take a moment)...")
    comment_analyzer = AdvancedAnalyzer(cleaned_df['cleaned_comment'])
    comment_topics = comment_analyzer.find_topics(max_topics=20)
    
    print("Running BERTopic on brand captions...")
    caption_analyzer = AdvancedAnalyzer(cleaned_df['cleaned_caption'])
    caption_topics = caption_analyzer.find_topics(max_topics=20)
    
    # Correlational analysis for tables
    engagement_tables = emoji_analyzer.correlate_engagement_with_captions(caption_topics, top_caption_bigrams)

    # --- Generate Outputs ---
    
    # Generate and save all visualizations
    visualizer.plot_engagement_by_day(engagement_by_day)
    visualizer.plot_posts_by_hour(posts_by_hour)
    visualizer.plot_engagement_by_hour(engagement_by_hour)
    
    # Build the final report
    report.add_section("Overall Comment Sentiment", f"- Positive: {sentiment_breakdown.get('Positive', 0)}\n- Neutral: {sentiment_breakdown.get('Neutral', 0)}\n- Negative: {sentiment_breakdown.get('Negative', 0)}")
    report.add_list("Top 25 Emojis in User Comments", top_emojis)
    report.add_list("Top 25 Words in User Comments", top_comment_words)
    report.add_list("Top 25 Two-Word Phrases in User Comments", top_comment_bigrams)
    report.add_list("Top 25 Three-Word Phrases in User Comments", top_comment_trigrams)
    report.add_topics("Discovered Topics in User Comments", comment_topics)
    report.add_topics("Discovered Topics in Brand Captions", caption_topics)
    
    # Add the powerful correlation tables
    report.add_dataframe("Brand Caption Topics vs. User Engagement", engagement_tables['topics'])
    report.add_dataframe("Brand Caption Bigrams vs. User Engagement", engagement_tables['bigrams'])

    report.generate_report()
    print("\nPipeline finished successfully!")

if __name__ == "__main__":
    main()

