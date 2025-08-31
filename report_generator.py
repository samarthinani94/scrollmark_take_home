import os

class ReportGenerator:
    """
    Generates a final markdown report with all the analysis results.
    """
    def __init__(self, filename="report.md"):
        self.filename = filename
        self.content = ["# Tree Hut Instagram Comment Analysis Report"]

    def add_section(self, title, text_content):
        """Adds a new section to the report."""
        self.content.append(f"\n## {title}\n")
        self.content.append(text_content)

    def add_chart(self, title, chart_filename):
        """Adds a chart to the report using Markdown image syntax."""
        # Assumes charts are in the 'outputs' directory
        chart_path = os.path.join('outputs', chart_filename)
        self.add_section(title, f"![{title}]({chart_path})")

    def add_ngrams(self, title, ngrams):
        """Formats and adds a list of n-grams to the report."""
        if not ngrams:
            content = "No n-grams to display.\n"
        else:
            # Format n-gram tuples into readable strings
            formatted_ngrams = [f"- `{' '.join(gram)}`: {count}" for gram, count in ngrams]
            content = "\n".join(formatted_ngrams)
        self.add_section(title, content)

    def add_top_emojis(self, title, emojis):
        """Formats and adds a list of top emojis."""
        if not emojis:
            content = "No emojis to display.\n"
        else:
            formatted_emojis = [f"- {emoji}: {count}" for emoji, count in emojis]
            content = "\n".join(formatted_emojis)
        self.add_section(title, content)
        
    def add_topics(self, title, topics):
        """Formats and adds a list of topics to the report."""
        if not topics:
            content = "No topics were generated.\n"
        else:
            content = ""
            for topic_name, words in topics:
                content += f"**{topic_name}**: `{'`, `'.join(words)}`\n\n"
        self.add_section(title, content)

    def add_dataframe(self, title, df):
        """Adds a pandas DataFrame as a markdown table."""
        if df.empty:
            content = "No data to display.\n"
        else:
            content = df.to_markdown(index=False)
        self.add_section(title, content)

    def add_emoji_sentiment_map(self, title, sentiment_map):
        """Adds the emoji sentiment classification key to the report."""
        content = ""
        for sentiment, emojis in sentiment_map.items():
            if emojis:
                content += f"**{sentiment.capitalize()} Emojis**: {' '.join(emojis)}\n\n"
        self.add_section(title, content)

    def generate_report(self):
        """Writes the collected content to the markdown file."""
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.content))
        print(f"\nReport generated: {self.filename}")

