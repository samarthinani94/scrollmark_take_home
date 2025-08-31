import pandas as pd
from data_processor import DataProcessor
from trend_analyzer import TrendAnalyzer
from emoji_analyzer import EmojiAnalyzer
from advanced_analyzer import AdvancedAnalyzer
from visualizer import Visualizer
from report_generator import ReportGenerator

def run_trend_analysis(analyzer, visualizer, report):
    """Handles all basic trend analysis."""
    print("Analyzing basic trends...")
    top_comment_bigrams = analyzer.get_top_ngrams('comments', n=2, top_k=25)
    top_comment_trigrams = analyzer.get_top_ngrams('comments', n=3, top_k=25)
    
    plot1_fn = visualizer.plot_avg_comments_by_day(analyzer.get_avg_positive_neutral_comments_per_post_by_day())
    plot2_fn = visualizer.plot_avg_posts_per_day_by_hour(analyzer.get_avg_posts_per_day_by_hour())
    plot3_fn = visualizer.plot_avg_positive_neutral_comments_by_hour(analyzer.get_avg_positive_neutral_comments_per_post_by_hour())

    report.add_ngrams("Top 25 Bigrams in User Comments", top_comment_bigrams)
    report.add_ngrams("Top 25 Trigrams in User Comments", top_comment_trigrams)
    report.add_chart("Average Positive+Neutral Comments per Post by Hour (PST)", plot3_fn)
    report.add_chart("Average Positive+Neutral Comments per Post by Day of Week", plot1_fn)
    report.add_chart("Average Posts per Day by Hour (PST)", plot2_fn)

def run_emoji_analysis(df, visualizer, report):
    """Handles all emoji-related analysis."""
    print("Analyzing emojis...")
    emoji_analyzer = EmojiAnalyzer(df['comment_text'])
    top_emojis = emoji_analyzer.get_top_emojis(top_k=25)
    emoji_sentiments, sentiment_map = emoji_analyzer.analyze_emoji_sentiments()
    
    plot_fn = visualizer.plot_emoji_sentiment(emoji_sentiments)
    report.add_chart("Emoji Sentiment Distribution", plot_fn)
    report.add_emoji_sentiment_map("Emoji Sentiment Classification Key", sentiment_map)
    report.add_top_emojis("Top 25 Most Used Emojis", top_emojis)

def run_topic_modeling(df, analyzer, report):
    """Handles all topic modeling analysis."""
    print("Running BERTopic on user comments (this may take a moment)...")
    comment_analyzer = AdvancedAnalyzer(df['cleaned_comment'])
    comment_topics = comment_analyzer.find_topics(max_topics=20)

    print("Running BERTopic on brand captions...")
    caption_analyzer = AdvancedAnalyzer(df['cleaned_caption'])
    caption_topics = caption_analyzer.find_topics(max_topics=20)

    if caption_topics:
        captions_list = df['cleaned_caption'].tolist()
        topics, _ = caption_analyzer.model.transform(captions_list)
        topic_ids = pd.Series(topics, index=df.index)

        topic_info = caption_analyzer.model.get_topic_info()
        
        # Create all engagement tables
        topic_trigram_table = analyzer.create_topic_trigram_table(topic_ids, topic_info)
        engagement_tables = analyzer.calculate_engagement_by_caption_feature(topic_ids, topic_info)
        
        # Add all sections to the report
        report.add_topics("Discovered Topics in User Comments", comment_topics)
        report.add_topics("Discovered Topics in Brand Captions", caption_topics)
        report.add_dataframe("Caption Topics vs. Top Comment Trigrams & Engagement", topic_trigram_table)
        report.add_dataframe("Brand Caption Topics vs. User Engagement", engagement_tables['topics'])

def main():
    """
    Main function to run the data processing and analysis pipeline.
    """
    file_source = 'https://drive.google.com/file/d/1o31ACWtmt-vjijb9QuP8Ohcr_hFlIPzS/view'
    processor = DataProcessor(file_source)
    cleaned_df = processor.process_data()
    print("Data processing complete.")
    print(f"Processed {len(cleaned_df)} comments.\n")

    analyzer = TrendAnalyzer(cleaned_df)
    visualizer = Visualizer()
    report = ReportGenerator()
    
    run_trend_analysis(analyzer, visualizer, report)
    run_emoji_analysis(cleaned_df, visualizer, report)
    run_topic_modeling(cleaned_df, analyzer, report)

    report.generate_report()
    print("\nPipeline finished successfully! Report and charts are in the 'report.md' file and 'outputs/' directory.")

if __name__ == "__main__":
    main()

