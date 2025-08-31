import pandas as pd

class ReportGenerator:
    """
    Generates a formatted text/markdown report from the analysis results.
    """
    def __init__(self, output_path="report.md"):
        self.output_path = output_path
        self.report_content = "# Social Media Trend Analysis Report for @treehut\n\n"

    def add_section(self, title, content):
        """Adds a new section to the report."""
        self.report_content += f"\n## {title}\n\n"
        self.report_content += content
        self.report_content += "\n"

    def add_list(self, title, data_list):
        """Adds a formatted list to the report."""
        content = ""
        if not data_list:
            content = "No data to display.\n"
        else:
            for item, count in data_list:
                content += f"- \"{item}\": {count}\n"
        self.add_section(title, content)
        
    def add_topics(self, title, topics):
        """Adds a formatted list for topics."""
        content = ""
        if not topics:
            content = "No topics discovered.\n"
        else:
            for name, words in topics:
                content += f"- **{name}**: {', '.join(words)}\n"
        self.add_section(title, content)

    def add_dataframe(self, title, df):
        """Adds a pandas DataFrame as a markdown table."""
        if df.empty:
            content = "No data to display.\n"
        else:
            content = df.to_markdown(index=False)
        self.add_section(title, content)

    def generate_report(self):
        """Writes the report content to the output file."""
        with open(self.output_path, 'w') as f:
            f.write(self.report_content)
        print(f"Report successfully generated at: {self.output_path}")
