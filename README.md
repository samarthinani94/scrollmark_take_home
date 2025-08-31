pip install -r requirements.txt

# Tree Hut Instagram Comment Analysis

This project analyzes a corpus of ~18,000 Instagram comments from March 2025 for the brand **@treehut**. It processes the raw comment data, identifies key trends and topics using NLP and topic modeling, and generates a comprehensive report with actionable insights and visualizations for a social media manager.

---

## 🚀 How to Run This Project

### 1. Setup Environment
It is recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate

# Or activate it (Windows)
# .venv\Scripts\activate
```

### 2. Install Dependencies
Install all the required Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Run the Analysis
Execute the main script to start the data processing and analysis pipeline:

```bash
python3 main.py
```

### 4. Check the Outputs
Once the script finishes, you will find:

- A detailed `report.md` file containing all the analysis, tables, and embedded charts.
- An `outputs/` directory containing all the generated chart images as `.png` files.

---

## 🌟 Extension Proposal: Moving from Analysis to Action

This project has established a robust pipeline for analyzing social media comments. The following ranked features outline a plan to evolve this one-time report into a more powerful, intelligent, and real-time monitoring tool for the Tree Hut brand.

### 1️⃣ LLM-Powered Comment Classification
**What it is:**
> Instead of just identifying broad topics, we can use a Large Language Model (LLM) like Google's Gemini to classify each individual comment into a set of predefined, business-relevant categories.

**Why it's valuable:**
This moves beyond simple topic modeling to understand user intent. It would allow a social media manager to instantly filter and quantify comments that are:

- **Purchase Intent:** "I need to buy this now!"
- **Product Feedback (Positive/Negative):** "The vanilla scent is amazing," or "This scrub was too abrasive."
- **Availability Request:** "Please bring this to Canada/the UK!"
- **Customer Service Issue:** "My order hasn't arrived."
- **Community Praise:** "You're the best brand ever!"

This provides an immediate, triaged view of the most critical comments, allowing the team to quickly respond to service issues or capitalize on positive feedback.

**How to implement it:**
Leverage a powerful pre-trained LLM with a few-shot learning prompt. Define your categories and provide just a few examples for each. The model can then classify new, unseen comments with high accuracy without needing a large, custom-trained dataset.

---

### 2️⃣ Interactive Real-Time Trend Dashboard
**What it is:**
> A web-based, interactive dashboard (built with a tool like Streamlit or Dash) that visualizes the insights from this analysis and updates on a daily or weekly schedule.

**Why it's valuable:**
Empowers the social media manager to self-serve insights without needing to run any code. They could filter trends by a specific date range, explore the sentiment surrounding a new product launch in near-real-time, and track how key conversation topics are evolving week over week. It turns a static report into a dynamic monitoring tool.

**How to implement it:**
The existing Python scripts would form the back-end analytics engine. A front-end application would be built to call these scripts and display the charts and tables. The entire pipeline could be deployed on a cloud service and scheduled to run automatically.

---

### 3️⃣ Competitive Landscape Analysis
**What it is:**
> Applying this exact analysis pipeline to the comments on posts from 2-3 key competitors (e.g., @dove, @soldejaneiro).

**Why it's valuable:**
Provides crucial market context. Are competitors also seeing high demand in Canada? What are the most-praised features of their products? How does the sentiment of their community compare to Tree Hut's? This insight is vital for identifying market gaps, competitive advantages, and potential threats.

**How to implement it:**
This would require a data collection step, either through a social media API or a web scraper, to gather competitor comment data. Once collected, the data could be processed through our existing, modular pipeline to generate a comparative report.