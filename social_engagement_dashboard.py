# Project: Data-Driven Social Engagement Initiative
# Author: Harshil Chauhan

import pandas as pd
import numpy as np
import sqlite3
import streamlit as st
from textblob import TextBlob

# ===================== MODULE 1: DATA LOADING / DATABASE =====================

def load_social_data(source_type, source_path):
    # supporting both csv and sqlite here since we went back and forth
    # on which one to use during the semester, easier to just support both
    if source_type == "csv":
        insta_data = pd.read_csv(source_path)
    elif source_type == "sqlite":
        conn = sqlite3.connect(source_path)
        insta_data = pd.read_sql("SELECT * FROM content_performance", conn)
        conn.close()
    else:
        st.error("Unknown source type, use 'csv' or 'sqlite'")
        return pd.DataFrame()

    # dropping rows missing the core engagement numbers, can't score
    # virality without shares/saves/likes anyway
    insta_data = insta_data.dropna(subset=["likes", "shares", "saves", "retention"])

    return insta_data


def load_comments(source_type, source_path):
    if source_type == "csv":
        insta_comments = pd.read_csv(source_path)
    elif source_type == "sqlite":
        conn = sqlite3.connect(source_path)
        insta_comments = pd.read_sql("SELECT * FROM comments", conn)
        conn.close()
    else:
        return pd.DataFrame()

    insta_comments = insta_comments.dropna(subset=["comment_text"])
    return insta_comments


# ===================== MODULE 2: VIRALITY PREDICTION =====================

def calculate_viral_score(shares, saves, likes):
    # keeping the weights simple here because the API data was inconsistent,
    # shares and saves matter way more than passive likes for real growth
    viral_score = (shares * 2.5) + (saves * 1.5) + (likes * 0.2)
    return viral_score


def add_viral_scores(content_df):
    score_list = []
    for index, row in content_df.iterrows():
        score = calculate_viral_score(row["shares"], row["saves"], row["likes"])
        score_list.append(score)

    content_df["viral_score"] = score_list
    return content_df


def get_topic_growth(content_df):
    # grouping by topic to see which struggle topics are actually
    # driving the most organic growth
    topic_growth = content_df.groupby("topic")["viral_score"].mean().sort_values(ascending=False)
    return topic_growth


# ===================== MODULE 3: NLP SENTIMENT ANALYZER =====================

relatable_trigger_words = ["me", "relate", "relatable", "struggle", "same", "literally me", "felt this"]


def check_relatable_keywords(comment_text):
    comment_lower = comment_text.lower()
    for trigger in relatable_trigger_words:
        if trigger in comment_lower:
            return True
    return False


def tag_comment(comment_text):
    # combining keyword triggers with TextBlob polarity so we're not
    # only relying on exact phrase matches
    has_keyword = check_relatable_keywords(comment_text)
    polarity = TextBlob(comment_text).sentiment.polarity

    if has_keyword or abs(polarity) > 0.5:
        return "Relatable"
    else:
        return "Neutral"


def analyze_comments(comment_list):
    tag_list = []
    for comment in comment_list:
        tag_list.append(tag_comment(comment))

    results_df = pd.DataFrame({"comment": comment_list, "tag": tag_list})
    return results_df


# ===================== MODULE 4: A/B TESTING (FORMAT COMPARISON) =====================

def run_ab_test(content_df):
    # simple comparison, not a full statistical test, just averaging
    # viral score by format since that's enough to make a decision here
    ab_test_results = content_df.groupby("format")["viral_score"].mean().sort_values(ascending=False)
    return ab_test_results


def get_winning_format(ab_test_results):
    winning_format = ab_test_results.idxmax()
    return winning_format


# ===================== MODULE 5: RECOMMENDER =====================

def recommend_next_week(content_df, ab_test_results):
    # prescriptive part, just picking the best topic and best format
    # from what we already calculated instead of building a whole new model
    topic_growth = get_topic_growth(content_df)
    best_topic = topic_growth.idxmax()
    best_format = ab_test_results.idxmax()

    recommendation = {
        "recommended_topic": best_topic,
        "recommended_format": best_format,
        "recommended_caption_style": "short, first-person, direct 'this is me' style"
    }

    return recommendation


# ===================== MODULE 6: STREAMLIT DASHBOARD =====================

def run_dashboard():
    # using Streamlit because it's much faster to deploy for the final
    # presentation than building a whole separate frontend
    st.title("Data-Driven Social Engagement Initiative Dashboard")

    st.write("Upload your content performance data to get started")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Waiting for a CSV upload (needs columns: topic, format, likes, shares, saves, retention)")
        return

    content_df = pd.read_csv(uploaded_file)
    content_df = content_df.dropna(subset=["likes", "shares", "saves", "retention"])

    content_df = add_viral_scores(content_df)

    st.subheader("Viral Score by Post")
    st.dataframe(content_df)

    st.subheader("Average Viral Score by Topic")
    topic_growth = get_topic_growth(content_df)
    st.bar_chart(topic_growth)

    # save to share ratio, mapping how often people save vs share stuff
    # since that ratio tells us when people actually felt understood
    content_df["save_to_share_ratio"] = content_df["saves"] / content_df["shares"].replace(0, np.nan)
    st.subheader("Save-to-Share Ratio by Post")
    st.bar_chart(content_df.set_index(content_df.index)["save_to_share_ratio"])

    if "format" in content_df.columns:
        st.subheader("A/B Test: Format Comparison")
        ab_test_results = run_ab_test(content_df)
        st.bar_chart(ab_test_results)

        winning_format = get_winning_format(ab_test_results)
        st.write("Winning format:", winning_format)

        st.subheader("Recommendation for Next Week")
        recommendation = recommend_next_week(content_df, ab_test_results)
        st.json(recommendation)

    if "follower_count" in content_df.columns:
        st.subheader("Follower Growth Over Time")
        st.line_chart(content_df["follower_count"])

    st.write("Optionally paste some comments below to test the sentiment analyzer")
    comment_input = st.text_area("Enter comments, one per line")

    if comment_input:
        comment_list = comment_input.split("\n")
        comment_list = [c.strip() for c in comment_list if c.strip() != ""]

        sentiment_results_df = analyze_comments(comment_list)
        st.subheader("Comment Sentiment Tags")
        st.dataframe(sentiment_results_df)

        relatable_pct = (sentiment_results_df["tag"] == "Relatable").mean() * 100
        st.write("Relatable comment percentage:", round(relatable_pct, 2), "%")


if __name__ == "__main__":
    run_dashboard()
