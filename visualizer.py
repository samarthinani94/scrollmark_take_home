import matplotlib.pyplot as plt
import seaborn as sns
import os

class Visualizer:
    """
    Handles the creation and saving of all charts and visualizations.
    """
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Set a consistent, clean style for all plots
        sns.set_theme(style="whitegrid")

    def _save_plot(self, filename):
        """Helper function to save the current plot."""
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"Chart saved to {filepath}")

    def plot_engagement_by_day(self, data):
        """Plots average comments per post by day of the week."""
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=data.index, y=data.values, palette="viridis")
        ax.set_title('Average User Comments per Post by Day of the Week (PST)', fontsize=16)
        ax.set_xlabel('Day of the Week', fontsize=12)
        ax.set_ylabel('Average Comments per Post', fontsize=12)
        ax.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        self._save_plot("engagement_by_day.png")

    def plot_posts_by_hour(self, data):
        """Plots the number of brand posts by hour of the day."""
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=data.index, y=data.values, palette="plasma")
        ax.set_title('Total Brand Posts by Hour of the Day (PST)', fontsize=16)
        ax.set_xlabel('Hour of the Day (PST)', fontsize=12)
        ax.set_ylabel('Number of Posts', fontsize=12)
        self._save_plot("posts_by_hour.png")

    def plot_engagement_by_hour(self, data):
        """Plots average comments per post by hour of the day."""
        plt.figure(figsize=(12, 6))
        ax = sns.lineplot(x=data.index, y=data.values, marker='o', color='mediumseagreen')
        ax.set_title('Average User Comments per Post by Hour (PST)', fontsize=16)
        ax.set_xlabel('Hour of the Day (PST)', fontsize=12)
        ax.set_ylabel('Average Comments per Post', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot("engagement_by_hour.png")
