# Project: Data-Driven Social Engagement Initiative
# Author: Harshil Chauhan

import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from textblob import TextBlob
from scipy.stats import ttest_ind
from sklearn.tree import DecisionTreeClassifier
import random
import io

# -------------------------------------------------------------------
# MODULE 1: Content Performance Tracker (Data Extraction Engine)
# -------------------------------------------------------------------

def scrape_engagement_data(dummy_url="https://fake-insta-mirror.com/post123"):
    # cant actually hit instagram api without approval n keys so faking the response html here
    # just showing that i know how to pull numbers out of tags with bs4
    mock_html = """
    <div class="post-card">
        <span class="likes">1830</span>
        <span class="shares">245</span>
        <span class="saves">410</span>
        <span class="retention">62</span>
    </div>
    """
    soup = BeautifulSoup(mock_html, "html.parser")
    scraped_post = {
        "likes": int(soup.find("span", class_="likes").text),
        "shares": int(soup.find("span", class_="shares").text),
        "saves": int(soup.find("span", class_="saves").text),
        "retention_rate": int(soup.find("span", class_="retention").text),
        "source_url": dummy_url
    }
    return scraped_post


def load_local_csv(uploaded_file):
    # basic wrapper so we can pull real data in if we ever export it from insta
    try:
        insta_df = pd.read_csv(uploaded_file)
        return insta_df
    except Exception as e:
        st.warning("couldnt read the csv, using generated data instead: " + str(e))
        return None


def build_mock_dataset(n=60):
    # generating a fake history of posts since we dont have real api access for this project
    random.seed(42)
    np.random.seed(42)

    topics = ["Social Anxiety", "Dating", "College Burnout", "Money Stress", "Friendship Drama"]
    formats = ["Short", "Long"]
    hooks = ["Visual", "Text"]

    relatable_lines = [
        "literally felt this so hard, this is such a struggle for me",
        "wow i relate to this way too much",
        "same, i go through this every single week",
        "this describes my life honestly, felt this deeply"
    ]
    neutral_lines = [
        "nice video, good editing",
        "cool content keep it up",
        "haha funny post",
        "not really my type of content but ok"
    ]

    rows = []
    for i in range(n):
        topic = random.choice(topics)
        post_format = random.choice(formats)
        hook = random.choice(hooks)
        hour_posted = random.randint(6, 23)

        likes = np.random.randint(500, 5000)
        shares = np.random.randint(20, 800)
        saves = np.random.randint(20, 900)
        retention_rate = np.random.randint(30, 95)

        comment_text = random.choice(relatable_lines + neutral_lines)

        rows.append({
            "topic": topic,
            "post_format": post_format,
            "hook_type": hook,
            "hour_posted": hour_posted,
            "likes": likes,
            "shares": shares,
            "saves": saves,
            "retention_rate": retention_rate,
            "insta_comments": comment_text
        })

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# MODULE 2: Virality Prediction Engine
# -------------------------------------------------------------------

def calc_viral_score(row):
    # likes are kinda a vanity metric, shares and saves actually mean people cared enough to act
    # so giving those way more weight than plain likes when scoring virality
    viral_score = (row["shares"] * 3) + (row["saves"] * 2.5) + (row["likes"] * 0.5)
    return viral_score


# -------------------------------------------------------------------
# MODULE 3: Audience Sentiment Analyzer (NLP Module)
# -------------------------------------------------------------------

trigger_words = ["struggle", "relate", "same", "felt this", "understand", "deeply"]


def tag_sentiment(comment_text):
    comment_lower = comment_text.lower()
    blob = TextBlob(comment_text)
    polarity = blob.sentiment.polarity

    has_trigger = any(word in comment_lower for word in trigger_words)

    # if the comment has one of our trigger phrases OR leans a bit negative/emotional
    # were calling it Relatable, otherwise its just a Neutral comment
    if has_trigger or polarity < -0.1:
        return "Relatable"
    else:
        return "Neutral"


# -------------------------------------------------------------------
# MODULE 4: A/B Testing Framework
# -------------------------------------------------------------------

def run_ab_test(df):
    short_scores = df[df["post_format"] == "Short"]["viral_score"]
    long_scores = df[df["post_format"] == "Long"]["viral_score"]

    t_stat, p_value_result = ttest_ind(short_scores, long_scores, equal_var=False)

    # throwing in a quick t-test here to prove the format difference isnt just random luck
    # if p_value_result is below 0.05 we can say the difference is statistically significant
    is_significant = p_value_result < 0.05

    return t_stat, p_value_result, is_significant


# -------------------------------------------------------------------
# MODULE 5: Engagement Optimization Recommender
# -------------------------------------------------------------------

def build_recommender(df):
    model_df = df.copy()

    # sklearn models need numbers not strings so just mapping the categories manually
    model_df["hook_encoded"] = model_df["hook_type"].map({"Visual": 1, "Text": 0})
    model_df["format_encoded"] = model_df["post_format"].map({"Short": 1, "Long": 0})

    median_score = model_df["viral_score"].median()
    model_df["high_performer"] = (model_df["viral_score"] > median_score).astype(int)

    features = model_df[["hour_posted", "hook_encoded", "format_encoded"]]
    target = model_df["high_performer"]

    # using a basic decision tree here since a neural net would be way overkill for 60 rows of data
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(features, target)

    return clf, model_df


def recommend_best_combo(clf):
    # brute forcing every combo of hour/hook/format and letting the tree score each one
    # instead of just eyeballing averages like a basic recommender would
    best_combo = None
    best_prob = -1

    for hour in range(6, 24):
        for hook_encoded, hook_label in [(1, "Visual"), (0, "Text")]:
            for format_encoded, format_label in [(1, "Short"), (0, "Long")]:
                test_row = pd.DataFrame(
                    [[hour, hook_encoded, format_encoded]],
                    columns=["hour_posted", "hook_encoded", "format_encoded"]
                )
                prob = clf.predict_proba(test_row)[0][1]
                if prob > best_prob:
                    best_prob = prob
                    best_combo = (hour, hook_label, format_label)

    return best_combo, best_prob


# -------------------------------------------------------------------
# MODULE 7: Trend Forecasting Module
# -------------------------------------------------------------------

def forecast_trending_topics():
    # again no real access to twitter/insta trend apis so mocking a scraped hashtag page
    mock_trend_html = """
    <ul class="trending-tags">
        <li data-count="1200">#CollegeBurnout</li>
        <li data-count="950">#Dating2026</li>
        <li data-count="1500">#SocialAnxietyCheck</li>
        <li data-count="600">#MoneyStressReal</li>
    </ul>
    """
    soup = BeautifulSoup(mock_trend_html, "html.parser")
    tags = soup.find_all("li")

    trend_list = []
    for tag in tags:
        trend_list.append({
            "hashtag": tag.text,
            "mentions": int(tag["data-count"])
        })

    trend_df = pd.DataFrame(trend_list).sort_values("mentions", ascending=False)
    return trend_df


# -------------------------------------------------------------------
# MODULE 6: Growth Visualization Dashboard (Streamlit frontend)
# -------------------------------------------------------------------

st.title("Data-Driven Social Engagement Initiative")
st.write("Unlox Data Science Major Project - tracking virality, sentiment and posting strategy")

st.subheader("Step 1: Data Source")
uploaded_file = st.file_uploader("Upload your own engagement CSV (optional)", type=["csv"])

if uploaded_file is not None:
    main_df = load_local_csv(uploaded_file)
    if main_df is None:
        main_df = build_mock_dataset()
else:
    main_df = build_mock_dataset()

# quick peek at one scraped post just to show module 1 scraping logic actually works
sample_scrape = scrape_engagement_data()
st.caption(f"Sample scraped post (mock): {sample_scrape}")

# calculating viral score for every row in the dataset
main_df["viral_score"] = main_df.apply(calc_viral_score, axis=1)
main_df["sentiment_tag"] = main_df["insta_comments"].apply(tag_sentiment)
main_df["save_to_share_ratio"] = main_df["saves"] / main_df["shares"]

st.subheader("Step 2: Raw + Processed Data")
st.dataframe(main_df)

st.subheader("Step 3: Virality by Topic")
topic_scores = main_df.groupby("topic")["viral_score"].mean().sort_values(ascending=False)
st.bar_chart(topic_scores)

st.subheader("Step 4: Save-to-Share Ratio (shows when people felt truly understood)")
ratio_by_topic = main_df.groupby("topic")["save_to_share_ratio"].mean().sort_values(ascending=False)
st.bar_chart(ratio_by_topic)

st.subheader("Step 5: Sentiment Breakdown")
sentiment_counts = main_df["sentiment_tag"].value_counts()
st.bar_chart(sentiment_counts)

st.subheader("Step 6: A/B Test - Short vs Long Format")
t_stat, p_value_result, is_significant = run_ab_test(main_df)
st.write(f"T-statistic: {t_stat:.3f}")
st.write(f"P-value: {p_value_result:.4f}")
if is_significant:
    st.success("Result is statistically significant (p < 0.05) - format really does matter here")
else:
    st.info("Result is NOT statistically significant - the difference could just be noise")

st.subheader("Step 7: Posting Strategy Recommendation (Decision Tree)")
clf_model, model_df_used = build_recommender(main_df)
best_combo, best_prob = recommend_best_combo(clf_model)
st.write(
    f"Recommended combo -> Hour: {best_combo[0]}:00, Hook: {best_combo[1]}, "
    f"Format: {best_combo[2]} (confidence: {best_prob:.2f})"
)

st.subheader("Step 8: Trend Forecasting")
trend_df = forecast_trending_topics()
st.dataframe(trend_df)
st.bar_chart(trend_df.set_index("hashtag")["mentions"])

st.subheader("Step 9: Export Strategy Report")

top_topic = topic_scores.index[0]
report_text = f"""STRATEGY REPORT - Data-Driven Social Engagement Initiative

Top performing topic (by viral score): {top_topic}
Recommended posting combo: {best_combo[2]} format, {best_combo[1]} hook, around {best_combo[0]}:00
A/B Test p-value: {p_value_result:.4f} ({'significant' if is_significant else 'not significant'})
Top trending hashtag to watch: {trend_df.iloc[0]['hashtag']}

Generated for the DADS major project.
"""

st.download_button(
    label="Download Strategy Report (.txt)",
    data=report_text,
    file_name="strategy_report.txt",
    mime="text/plain"
)
