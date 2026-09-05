# Data-Driven Social Engagement Initiative 📈

This repository contains my **final Data Science major project** for **Unlox**. The goal of this ecosystem is to replace creative guesswork with rigorous statistical analysis to identify, measure, and optimize content that fosters deep human connection. 

By moving beyond passive metrics, this dashboard decodes the science of "relatability".

## Key Features & Core Modules
* **Data Extraction Engine:** Uses `BeautifulSoup` to scrape mock engagement metrics (shares, saves, likes) directly from target URLs.
* **Virality Prediction:** Calculates a weighted 'Viral Coefficient' that prioritizes high-value actions (shares/saves) over passive actions (likes) to identify organic growth potential.
* **Audience Sentiment Analyzer:** An NLP pipeline utilizing `TextBlob` and specific linguistic triggers to categorize user comments strictly as "Relatable" or "Neutral" to validate problem awareness.
* **A/B Testing Framework:** Runs independent T-tests via `SciPy` to prove statistically significant differences between content variables (e.g., Short vs. Long format).
* **Engagement Optimization Recommender:** A `Scikit-learn` Decision Tree Classifier that acts as a prescriptive analytics engine, predicting the optimal posting format and hook style for future content.
* **Trend Forecasting:** Scrapes trending hashtags to predict rising "relatable struggles" before they hit the mainstream.
* **Automated Strategy Report:** Generates a downloadable `.txt` report summarizing top topics, statistical findings, and AI-driven recommendations.

## Technical Stack
* **Data Processing:** Python (Pandas, NumPy)
* **Web Scraping:** BeautifulSoup
* **Statistical Modeling & Machine Learning:** Scikit-learn, SciPy
* **NLP:** TextBlob
* **Frontend:** Streamlit

## How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install pandas numpy streamlit textblob scipy scikit-learn beautifulsoup4`
3. Launch the Streamlit server: `streamlit run social_engagement_dashboard.py`

---
*Developed by Harshil Chauhan*
