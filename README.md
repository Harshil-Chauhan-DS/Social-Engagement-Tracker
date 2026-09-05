# Data-Driven Social Engagement Initiative 📊
# Final Project of Unlox Data Science Course
A data science dashboard built to replace creative guesswork with actual metrics. This project analyzes raw social media data to figure out what makes content go viral and uses NLP to check if audiences actually find the content "relatable."

This is my final Data Science Major Project for Unlox / Graphic Era University (December 2026).

## What It Does
* **Virality Prediction:** Calculates a custom "Viral Score" by weighing shares and saves much heavier than passive likes.
* **Sentiment Analysis:** Uses TextBlob and keyword triggers to tag user comments as either "Relatable" or "Neutral".
* **A/B Testing:** Compares short vs. long-form content to see what actually drives growth.
* **Save-to-Share Ratio:** Maps out exactly when an audience feels understood.

## Tech Stack
* **Frontend:** Streamlit (chosen because it's way faster to deploy for a presentation than building a React app from scratch).
* **Data Processing:** Pandas & NumPy.
* **NLP:** TextBlob.
* **Database:** SQLite (kept it lightweight so it doesn't require setting up a separate PostgreSQL server just to test the code).

## How to Run It Locally
1. Clone this repo to your machine.
2. Open your terminal in the project folder and install the dependencies:
   ```pip install pandas numpy streamlit textblob```
   
* Launch the dashboard:
streamlit run social_engagement_dashboard.py
Upload your social media CSV file when the browser tab automatically opens.
Project Notes
The NLP script uses a hardcoded list of relatable trigger words (like "literally me" or "struggle") combined with TextBlob's baseline polarity. It's not a massive LLM, but it gets the job done perfectly for this scope.

Author: Harshil Chauhan
