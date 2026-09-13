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

# Module 1: content performance and data loading

def scrape_engagement_data(source_url="https://fake-insta-mirror.com/post123"):
    # Use sample HTML because the real Instagram API is not available here.
    # Parse the same fields a real response would provide.
    html = """
    <div class="post-card">
        <span class="likes">1830</span>
        <span class="shares">245</span>
        <span class="saves">410</span>
        <span class="retention">62</span>
    </div>
    """
    soup = BeautifulSoup(html, "html.parser")
    post_data = {
        "likes": int(soup.find("span", class_="likes").text),
        "shares": int(soup.find("span", class_="shares").text),
        "saves": int(soup.find("span", class_="saves").text),
        "retention_rate": int(soup.find("span", class_="retention").text),
        "source_url": source_url
    }
    return post_data

def load_local_csv(uploaded_file):
    # Accept uploaded CSVs without requiring one exact column layout.
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.warning("couldnt read the csv, using generated data instead: " + str(exc))
        return None

    data = universal_column_mapper(data)
    return data

def find_column(df, keywords):
    # Match columns by keywords so common naming variations still work.

    for col in df.columns:
        col_lower = col.lower()
        for kw in keywords:
            if kw in col_lower:
                return col
    return None

def get_numeric_column(df, column, low, high):
    # Convert usable values to numbers and handle bad rows.
    if column is not None:
        values = pd.to_numeric(df[column], errors="coerce")
        if not values.isna().all():
            # a few bad records are fine, just patch NaNs with the column median
            return values.fillna(values.median())
    # Fall back to generated values when the field is missing or unusable.
    return pd.Series(np.random.randint(low, high, size=len(df)))

def get_hour_column(df, column):
    if column is not None:
        values = pd.to_numeric(df[column], errors="coerce")
        if not values.isna().all():
            return values.fillna(values.median()).astype(int) % 24
        # A timestamp may have the posting hour embedded in it.
        try:
            timestamps = pd.to_datetime(df[column], errors="coerce")
            if timestamps.notna().any():
                return timestamps.dt.hour.fillna(12).astype(int)
        except Exception:
            pass
    return pd.Series(np.random.randint(6, 23, size=len(df)))

def get_categorical_column(df, column, choices):
    if column is not None:
        # Keep categorical fields only when they look like actual categories.
        # category (like caption_length getting mistaken for a format column), so we bail out
        # Otherwise, use the same fallback values as the rest of the pipeline.
        if df[column].nunique() <= 15:
            return df[column].astype(str)
    return pd.Series([random.choice(choices) for _ in range(len(df))])

def get_text_column(df, column):
    if column is not None:
        return df[column].astype(str)
    return pd.Series(["no comment data provided" for _ in range(len(df))])

def universal_column_mapper(df):
    # Normalize different CSV layouts into the columns used by the dashboard.

    # Missing fields are filled so later steps can run consistently.
    random.seed(1)
    np.random.seed(1)

    likes_col = find_column(df, ["like"])
    shares_col = find_column(df, ["share"])
    saves_col = find_column(df, ["save"])
    retention_col = find_column(df, ["retention", "watch_time", "completion"])
    comments_col = find_column(df, ["comment", "caption", "text"])
    format_col = find_column(df, ["format", "post_type", "media_type"])
    hook_col = find_column(df, ["hook"])
    topic_col = find_column(df, ["topic", "category", "niche", "struggle"])
    hour_col = find_column(df, ["hour", "time", "posted_at", "timestamp"])

    clean_data = pd.DataFrame()
    clean_data["topic"] = get_categorical_column(
        df, topic_col, ["Social Anxiety", "Dating", "College Burnout", "Money Stress", "Friendship Drama"])
    clean_data["post_format"] = get_categorical_column(df, format_col, ["Short", "Long"])
    clean_data["hook_type"] = get_categorical_column(df, hook_col, ["Visual", "Text"])
    clean_data["hour_posted"] = get_hour_column(df, hour_col)
    clean_data["likes"] = get_numeric_column(df, likes_col, 500, 5000)
    clean_data["shares"] = get_numeric_column(df, shares_col, 20, 800)
    clean_data["saves"] = get_numeric_column(df, saves_col, 20, 900)
    clean_data["retention_rate"] = get_numeric_column(df, retention_col, 30, 95)
    clean_data["insta_comments"] = get_text_column(df, comments_col)

    detected = {"topic": topic_col, "post_format": format_col, "hook_type": hook_col,
                "hour_posted": hour_col, "likes": likes_col, "shares": shares_col,
                "saves": saves_col, "retention_rate": retention_col, "insta_comments": comments_col}
    missing_fields = [k for k, v in detected.items() if v is None]
    if missing_fields:
        st.info(f"couldnt find real columns for {missing_fields} in your csv, mocked those instead")

    return clean_data

def build_mock_dataset(n=60):
    # Generate a repeatable sample history for the dashboard.
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

    records = []
    for i in range(n):
        topic = random.choice(topics)
        post_format = random.choice(formats)
        hook = random.choice(hooks)
        hour_posted = random.randint(6, 23)

        likes = np.random.randint(500, 5000)
        shares = np.random.randint(20, 800)
        saves = np.random.randint(20, 900)
        retention_rate = np.random.randint(30, 95)

        comment = random.choice(relatable_lines + neutral_lines)

        records.append({
            "topic": topic,
            "post_format": post_format,
            "hook_type": hook,
            "hour_posted": hour_posted,
            "likes": likes,
            "shares": shares,
            "saves": saves,
            "retention_rate": retention_rate,
            "insta_comments": comment
        })

    return pd.DataFrame(records)

# Module 2: virality scoring

def calc_viral_score(row):
    # Shares and saves carry more weight because they represent active engagement.

    score = (row["shares"] * 3) + (row["saves"] * 2.5) + (row["likes"] * 0.5)
    return score

# Module 3: audience sentiment analysis

trigger_words = ["struggle", "relate", "same", "felt this", "understand", "deeply"]

def tag_sentiment(comment):
    comment_lower = comment.lower()
    blob = TextBlob(comment)
    polarity = blob.sentiment.polarity

    has_trigger = any(word in comment_lower for word in trigger_words)

    # Flag comments that match the project's relatable-language rule.

    if has_trigger or polarity < -0.1:
        return "Relatable"
    else:
        return "Neutral"

# Module 4: A/B testing

def run_ab_test(df):
    # Compare the two most common formats in the current dataset.
    # instead of assuming everyone calls them "Short" and "Long"
    formats = df["post_format"].value_counts().index[:2].tolist()

    if len(formats) < 2:
        # There is nothing to compare when only one format is present.
        return None, None, False, formats

    group_a = df[df["post_format"] == formats[0]]["score"]
    group_b = df[df["post_format"] == formats[1]]["score"]

    t_stat, p_value = ttest_ind(group_a, group_b, equal_var=False)

    # Welch's t-test checks whether the observed difference is likely to be meaningful.
    # if p_value is below 0.05 we can say the difference is statistically significant
    significant = p_value < 0.05

    return t_stat, p_value, significant, formats

# Module 5: engagement recommendation

def build_recommender(df):
    model_data = df.copy()

    # Encode whatever category labels are actually present in the data.

    model_data["hook_type"] = model_data["hook_type"].astype("category")
    model_data["post_format"] = model_data["post_format"].astype("category")
    model_data["hook_encoded"] = model_data["hook_type"].cat.codes
    model_data["format_encoded"] = model_data["post_format"].cat.codes

    hook_options = list(model_data["hook_type"].cat.categories)
    format_options = list(model_data["post_format"].cat.categories)

    median_score = model_data["score"].median()
    model_data["high_performer"] = (model_data["score"] > median_score).astype(int)

    features = model_data[["hour_posted", "hook_encoded", "format_encoded"]]
    target = model_data["high_performer"]

    # A small decision tree is enough for this dataset.
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(features, target)

    return clf, model_data, hook_options, format_options

def recommend_best_combo(clf, hook_options, format_options):
    # Score every supported hour, hook, and format combination.

    best_combo = None
    best_probability = -1

    # A probability needs both outcome classes to be present.

    if len(clf.classes_) < 2:
        return None, None

    for hour in range(6, 24):
        for hook_code, hook_label in enumerate(hook_options):
            for format_code, format_label in enumerate(format_options):
                test_data = pd.DataFrame(
                    [[hour, hook_code, format_code]],
                    columns=["hour_posted", "hook_encoded", "format_encoded"]
                )
                probability = clf.predict_proba(test_data)[0][1]
                if probability > best_probability:
                    best_probability = probability
                    best_combo = (hour, hook_label, format_label)

    return best_combo, best_probability

# Module 7: trend forecasting

def forecast_trending_topics():
    # Use sample trend HTML because live social trend APIs are not connected.
    html = """
    <ul class="trending-tags">
        <li data-count="1200">#CollegeBurnout</li>
        <li data-count="950">#Dating2026</li>
        <li data-count="1500">#SocialAnxietyCheck</li>
        <li data-count="600">#MoneyStressReal</li>
    </ul>
    """
    soup = BeautifulSoup(html, "html.parser")
    tags = soup.find_all("li")

    trends = []
    for tag in tags:
        trends.append({
            "hashtag": tag.text,
            "mentions": int(tag["data-count"])
        })

    trend_data = pd.DataFrame(trends).sort_values("mentions", ascending=False)
    return trend_data

# Module 6: Streamlit dashboard

st.title("Data-Driven Social Engagement Initiative")
st.write("Unlox Data Science Major Project - tracking virality, sentiment and posting strategy")

st.subheader("Step 1: Data Source")
uploaded_file = st.file_uploader("Upload your own engagement CSV (optional)", type=["csv"])

if uploaded_file is not None:
    data = load_local_csv(uploaded_file)
    if data is None:
        data = build_mock_dataset()
else:
    data = build_mock_dataset()

# Show the scraping example used by Module 1.
sample_post = scrape_engagement_data()
st.caption(f"Sample scraped post (mock): {sample_post}")

# Add the calculated engagement metrics used by the dashboard.
data["score"] = data.apply(calc_viral_score, axis=1)
data["sentiment_tag"] = data["insta_comments"].apply(tag_sentiment)

# Avoid infinite ratios when a post has zero shares.

safe_shares = data["shares"].replace(0, np.nan)
data["save_to_share_ratio"] = data["saves"] / safe_shares

st.subheader("Step 2: Raw + Processed Data")
st.dataframe(data)

st.subheader("Step 3: Virality by Topic")
topic_scores = data.groupby("topic")["score"].mean().sort_values(ascending=False)
st.bar_chart(topic_scores)

st.subheader("Step 4: Save-to-Share Ratio (shows when people felt truly understood)")
ratio_by_topic = data.groupby("topic")["save_to_share_ratio"].mean().sort_values(ascending=False)
st.bar_chart(ratio_by_topic)

st.subheader("Step 5: Sentiment Breakdown")
sentiment_counts = data["sentiment_tag"].value_counts()
st.bar_chart(sentiment_counts)

st.subheader("Step 6: A/B Test - Top 2 Post Formats")
t_stat, p_value, significant, formats = run_ab_test(data)
if t_stat is None:
    st.warning("only found one post format in this data, cant run an A/B test on just one group")
else:
    st.write(f"Comparing: {formats[0]} vs {formats[1]}")
    st.write(f"T-statistic: {t_stat:.3f}")
    st.write(f"P-value: {p_value:.4f}")
    if significant:
        st.success("Result is statistically significant (p < 0.05) - format really does matter here")
    else:
        st.info("Result is NOT statistically significant - the difference could just be noise")

st.subheader("Step 7: Posting Strategy Recommendation (Decision Tree)")
model, model_data, hook_options, format_options = build_recommender(data)
best_combo, best_probability = recommend_best_combo(model, hook_options, format_options)
if best_combo is None:
    st.warning("not enough variety in the data to build a real recommendation")
else:
    st.write(
        f"Recommended combo -> Hour: {best_combo[0]}:00, Hook: {best_combo[1]}, "
        f"Format: {best_combo[2]} (confidence: {best_probability:.2f})"
    )

st.subheader("Step 8: Trend Forecasting")
trend_data = forecast_trending_topics()
st.dataframe(trend_data)
st.bar_chart(trend_data.set_index("hashtag")["mentions"])

st.subheader("Step 9: Export Strategy Report")

top_topic = topic_scores.index[0]
combo_line = (f"{best_combo[2]} format, {best_combo[1]} hook, around {best_combo[0]}:00"
              if best_combo is not None else "not enough data to recommend a combo")
ab_line = (f"{p_value:.4f} ({'significant' if significant else 'not significant'})"
           if t_stat is not None else "not enough format variety to run")

report_text = f"""STRATEGY REPORT - Data-Driven Social Engagement Initiative

Top performing topic (by viral score): {top_topic}
Recommended posting combo: {combo_line}
A/B Test p-value: {ab_line}
Top trending hashtag to watch: {trend_data.iloc[0]['hashtag']}

Generated for the DADS major project.
"""

st.download_button(
    label="Download Strategy Report (.txt)",
    data=report_text,
    file_name="strategy_report.txt",
    mime="text/plain"
)
