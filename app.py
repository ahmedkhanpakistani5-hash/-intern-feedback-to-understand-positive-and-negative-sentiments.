import os
import re
import json
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from groq import Groq


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="InternPulse AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DARK GLASS + YELLOW / SKY BLUE DESIGN
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(56,189,248,.13), transparent 28%),
            radial-gradient(circle at 90% 15%, rgba(250,204,21,.10), transparent 25%),
            linear-gradient(135deg, #06111f 0%, #081827 45%, #050b14 100%);
        color: #e5f6ff;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071522 0%, #08101c 100%);
        border-right: 1px solid rgba(56,189,248,.18);
    }

    [data-testid="stSidebar"] * {
        color: #dff7ff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .hero {
        padding: 28px 30px;
        border: 1px solid rgba(56,189,248,.24);
        border-radius: 24px;
        background:
            linear-gradient(135deg, rgba(14,35,55,.88), rgba(8,18,31,.88));
        box-shadow: 0 18px 60px rgba(0,0,0,.25);
        margin-bottom: 24px;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(250,204,21,.10);
        border: 1px solid rgba(250,204,21,.32);
        color: #fde047;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .hero h1 {
        margin: 12px 0 8px 0;
        font-size: 42px;
        line-height: 1.08;
        font-weight: 800;
        color: #f8fdff;
    }

    .hero h1 span {
        color: #38bdf8;
    }

    .hero p {
        margin: 0;
        color: #a9c7d8;
        font-size: 16px;
        max-width: 900px;
        line-height: 1.65;
    }

    .section-title {
        color: #f8fdff;
        font-size: 22px;
        font-weight: 800;
        margin: 22px 0 12px 0;
    }

    .metric-card {
        padding: 20px;
        min-height: 130px;
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(15,37,56,.92), rgba(8,21,35,.92));
        border: 1px solid rgba(56,189,248,.18);
        box-shadow: 0 12px 35px rgba(0,0,0,.18);
    }

    .metric-label {
        color: #8fb0c2;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .06em;
    }

    .metric-value {
        color: #f8fdff;
        font-size: 32px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-accent {
        color: #fde047;
    }

    .glass-card {
        padding: 20px;
        border-radius: 20px;
        background: rgba(10,27,43,.74);
        border: 1px solid rgba(56,189,248,.16);
        box-shadow: 0 12px 38px rgba(0,0,0,.16);
    }

    .insight {
        padding: 15px 17px;
        border-radius: 15px;
        margin: 8px 0;
        background: rgba(15,36,54,.78);
        border-left: 4px solid #38bdf8;
        color: #d9f4ff;
        line-height: 1.55;
    }

    .insight strong {
        color: #fde047;
    }

    .positive {
        border-left-color: #38bdf8;
    }

    .negative {
        border-left-color: #facc15;
    }

    .stButton > button {
        border-radius: 12px !important;
        border: 1px solid rgba(56,189,248,.35) !important;
        background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        color: white !important;
        font-weight: 800 !important;
        box-shadow: 0 8px 24px rgba(14,165,233,.18) !important;
        transition: .2s ease !important;
    }

    .stButton > button:hover {
        border-color: #fde047 !important;
        box-shadow: 0 8px 28px rgba(250,204,21,.16) !important;
        transform: translateY(-1px);
    }

    .stDownloadButton > button {
        border-radius: 12px !important;
        background: rgba(250,204,21,.10) !important;
        border: 1px solid rgba(250,204,21,.35) !important;
        color: #fde047 !important;
        font-weight: 800 !important;
    }

    .stTextArea textarea,
    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] > div,
    .stFileUploader section {
        background: #0b1d2d !important;
        color: #e8f8ff !important;
        border-color: rgba(56,189,248,.22) !important;
        border-radius: 12px !important;
    }

    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {
        color: #6f8fa2 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(8,21,35,.65);
        padding: 7px;
        border-radius: 14px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #8fb0c2;
        border-radius: 10px;
        padding: 8px 16px;
    }

    .stTabs [aria-selected="true"] {
        color: #fde047 !important;
        background: rgba(250,204,21,.08);
    }

    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
    }

    .small-note {
        color: #7595a7;
        font-size: 12px;
        line-height: 1.5;
    }

    .footer {
        text-align: center;
        color: #638092;
        font-size: 12px;
        padding: 25px 0 5px 0;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SAMPLE TRAINING DATA
# ============================================================
TRAINING_DATA = pd.DataFrame(
    {
        "text": [
            "The mentor was supportive and always answered my questions.",
            "I learned a lot and received useful feedback every week.",
            "The internship tasks were interesting and relevant to my career.",
            "The team welcomed me and gave me clear guidance.",
            "I enjoyed working on real projects and improving my skills.",
            "The supervisor gave constructive feedback and helped me grow.",
            "Communication with the team was excellent.",
            "The training sessions were practical and useful.",
            "I was given meaningful responsibilities and learned quickly.",
            "The work environment was friendly and motivating.",
            "My mentor explained difficult concepts clearly.",
            "The internship improved my technical and communication skills.",
            "The tasks were well organized and easy to understand.",
            "I received recognition for completing my work.",
            "The company provided a great learning environment.",
            "I felt ignored when I asked for help.",
            "The tasks were repetitive and did not teach me anything.",
            "My mentor rarely gave feedback.",
            "The workload was unreasonable and stressful.",
            "Communication from the team was poor.",
            "I was not given clear instructions.",
            "The internship did not provide enough practical experience.",
            "The deadlines were unrealistic.",
            "My work was never recognized.",
            "The training material was confusing and outdated.",
            "I had to solve problems without any guidance.",
            "The environment was stressful and unwelcoming.",
            "I did not receive useful feedback.",
            "The assigned tasks were unrelated to my learning goals.",
            "There was too much work and not enough support.",
            "The experience was okay but could be improved.",
            "The internship was average and not very exciting.",
            "Some tasks were useful but others were repetitive.",
            "Communication was sometimes good and sometimes confusing.",
            "The mentor helped occasionally, but feedback was limited.",
            "The workload was manageable, although some deadlines were tight.",
            "I learned a few things but expected more practical work.",
            "The experience was neither great nor terrible.",
            "The internship experience was mixed and could be improved in several areas.",
        ],
        "label": [
            "Positive", "Positive", "Positive", "Positive", "Positive",
            "Positive", "Positive", "Positive", "Positive", "Positive",
            "Positive", "Positive", "Positive", "Positive", "Positive",
            "Negative", "Negative", "Negative", "Negative", "Negative",
            "Negative", "Negative", "Negative", "Negative", "Negative",
            "Negative", "Negative", "Negative", "Negative", "Negative",
            "Neutral", "Neutral", "Neutral", "Neutral", "Neutral",
            "Neutral", "Neutral",
        ],
    }
)


# ============================================================
# HELPERS
# ============================================================
@st.cache_resource
def train_model():
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(TRAINING_DATA["text"], TRAINING_DATA["label"])
    return model


MODEL = train_model()


def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentiment_class(text):
    text = clean_text(text)
    prediction = MODEL.predict([text])[0]
    probabilities = MODEL.predict_proba([text])[0]
    classes = MODEL.classes_
    confidence = float(probabilities[list(classes).index(prediction)])
    return prediction, confidence


def get_groq_key():
    # Streamlit Cloud secrets first, then environment variable.
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
        if secret_key:
            return secret_key
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY", "")


def groq_client(api_key):
    return Groq(api_key=api_key)


def call_groq(prompt, api_key, max_tokens=1200):
    client = groq_client(api_key)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert HR analytics and employee-experience analyst. "
                    "Analyze intern feedback objectively. Identify themes, practical "
                    "improvements, and risks. Do not invent statistics. Keep answers "
                    "clear, concise, and useful for an internship program manager."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_completion_tokens=max_tokens,
    )

    return response.choices[0].message.content


def extract_themes(texts):
    stop_words = {
        "the", "and", "was", "were", "for", "with", "this", "that", "have",
        "had", "from", "very", "but", "not", "are", "our", "they", "their",
        "too", "did", "its", "into", "you", "your", "about", "more", "some",
        "just", "been", "also", "when", "what", "would", "could", "can",
        "my", "me", "i", "a", "an", "to", "of", "in", "on", "is", "it",
    }

    words = []
    for text in texts:
        tokens = re.findall(r"[a-zA-Z]{4,}", str(text).lower())
        words.extend([w for w in tokens if w not in stop_words])

    return Counter(words).most_common(12)


def make_report(df, ai_insight=""):
    lines = [
        "INTERNPULSE AI — INTERN FEEDBACK REPORT",
        "=" * 50,
        "",
        f"Total feedback records: {len(df)}",
        "",
        "SENTIMENT SUMMARY",
        "-" * 30,
    ]

    for label in ["Positive", "Neutral", "Negative"]:
        count = int((df["Sentiment"] == label).sum())
        pct = (count / len(df) * 100) if len(df) else 0
        lines.append(f"{label}: {count} ({pct:.1f}%)")

    lines.extend(
        [
            "",
            "CONFIDENCE",
            "-" * 30,
            f"Average model confidence: {df['Confidence'].mean():.1%}",
            "",
            "MOST COMMON TERMS",
            "-" * 30,
        ]
    )

    for word, count in extract_themes(df["Feedback"].tolist()):
        lines.append(f"{word}: {count}")

    if ai_insight:
        lines.extend(["", "GROQ AI ANALYSIS", "-" * 30, ai_insight])

    return "\n".join(lines)


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">⚡ AI Internship Analytics</div>
        <h1>InternPulse <span>AI</span></h1>
        <p>
            Turn intern feedback into clear sentiment insights, satisfaction trends,
            and practical improvement areas using Machine Learning + Groq AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚡ InternPulse")
    st.caption("Feedback intelligence dashboard")

    st.markdown("### 🔐 Groq API")
    saved_key = get_groq_key()

    if saved_key:
        api_key = saved_key
        st.success("Groq API key detected.")
    else:
        api_key = st.text_input(
            "Enter Groq API key",
            type="password",
            placeholder="gsk_...",
            help="For Streamlit Cloud, add GROQ_API_KEY under App Settings → Secrets.",
        )

    st.markdown("---")
    st.markdown("### 🧠 AI Architecture")
    st.markdown(
        """
        **1. TF-IDF**  
        Converts feedback into numerical features.

        **2. Logistic Regression**  
        Classifies feedback as Positive, Neutral, or Negative.

        **3. Groq GPT-OSS 20B**  
        Explains themes and suggests practical improvements.

        **4. Streamlit**  
        Presents the results in an interactive dashboard.
        """
    )

    st.markdown("---")
    st.markdown(
        '<div class="small-note">Tip: upload a CSV with a column named '
        '<b>feedback</b> or <b>text</b>. A sentiment/label column is optional.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# INPUT
# ============================================================
tab1, tab2 = st.tabs(["💬 Analyze Feedback", "📁 Upload CSV"])

uploaded_df = None

with tab1:
    st.markdown('<div class="section-title">Analyze a single intern review</div>', unsafe_allow_html=True)

    feedback = st.text_area(
        "Intern feedback",
        placeholder="Example: My mentor was helpful, but I wish I had received more practical tasks and weekly feedback.",
        height=150,
        label_visibility="collapsed",
    )

    if st.button("⚡ Analyze Sentiment", use_container_width=True):
        if not feedback.strip():
            st.warning("Please enter some intern feedback first.")
        else:
            sentiment, confidence = sentiment_class(feedback)

            emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "⚠️"}[sentiment]

            st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Sentiment</div>'
                    f'<div class="metric-value">{emoji} {sentiment}</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Confidence</div>'
                    f'<div class="metric-value metric-accent">{confidence:.1%}</div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    '<div class="metric-card"><div class="metric-label">Model</div>'
                    '<div class="metric-value">Logistic ML</div></div>',
                    unsafe_allow_html=True,
                )

            if api_key:
                with st.spinner("Groq AI is extracting actionable insights..."):
                    prompt = f"""
Analyze this intern feedback:

"{feedback}"

Return:
1. Main theme
2. What is working well
3. What should be improved
4. One practical action for the internship team
5. A short professional summary

Do not invent facts.
"""
                    try:
                        ai_result = call_groq(prompt, api_key, 900)
                        st.markdown('<div class="section-title">🤖 Groq AI Insight</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="glass-card">{ai_result}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Groq request failed: {e}")
            else:
                st.info("Add your Groq API key to enable AI-powered recommendations.")


with tab2:
    st.markdown('<div class="section-title">Upload your feedback dataset</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "CSV file",
        type=["csv"],
        help="Recommended: feedback column. Optional: sentiment/label column.",
    )

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)

            text_candidates = [
                c for c in raw_df.columns
                if c.lower().strip() in ["feedback", "text", "review", "comment", "comments"]
            ]

            if text_candidates:
                text_col = text_candidates[0]
            else:
                text_col = st.selectbox(
                    "Which column contains the intern feedback?",
                    raw_df.columns,
                )

            uploaded_df = raw_df.copy()
            uploaded_df["Feedback"] = uploaded_df[text_col].astype(str).map(clean_text)
            uploaded_df = uploaded_df[uploaded_df["Feedback"].str.len() > 0].copy()

            uploaded_df["Sentiment"], uploaded_df["Confidence"] = zip(
                *uploaded_df["Feedback"].map(sentiment_class)
            )

            st.success(f"Loaded {len(uploaded_df)} feedback records.")

            st.dataframe(
                uploaded_df[["Feedback", "Sentiment", "Confidence"]],
                use_container_width=True,
                hide_index=True,
            )

        except Exception as e:
            st.error(f"Could not read this CSV: {e}")


# ============================================================
# DATASET DASHBOARD
# ============================================================
if uploaded_df is not None and len(uploaded_df) > 0:
    df = uploaded_df.copy()

    st.markdown('<div class="section-title">📊 Feedback Intelligence Dashboard</div>', unsafe_allow_html=True)

    total = len(df)
    positive = int((df["Sentiment"] == "Positive").sum())
    neutral = int((df["Sentiment"] == "Neutral").sum())
    negative = int((df["Sentiment"] == "Negative").sum())
    satisfaction = positive / total * 100 if total else 0
    avg_conf = df["Confidence"].mean() if total else 0

    c1, c2, c3, c4, c5 = st.columns(5)

    metrics = [
        ("Total Reviews", total, ""),
        ("Positive", positive, ""),
        ("Neutral", neutral, ""),
        ("Negative", negative, ""),
        ("Positive Rate", f"{satisfaction:.1f}%", ""),
    ]

    for col, (label, value, _) in zip([c1, c2, c3, c4, c5], metrics):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    left, right = st.columns(2)

    with left:
        counts = (
            df["Sentiment"]
            .value_counts()
            .reindex(["Positive", "Neutral", "Negative"])
            .fillna(0)
            .reset_index()
        )
        counts.columns = ["Sentiment", "Count"]

        fig = px.pie(
            counts,
            names="Sentiment",
            values="Count",
            hole=0.62,
            title="Sentiment Distribution",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dff7ff",
            legend_title_text="",
            margin=dict(t=55, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig2 = px.bar(
            counts,
            x="Sentiment",
            y="Count",
            text="Count",
            title="Feedback Volume by Sentiment",
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dff7ff",
            xaxis_title="",
            yaxis_title="Reviews",
            margin=dict(t=55, b=10, l=10, r=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">🔎 Common Feedback Themes</div>', unsafe_allow_html=True)

    themes = extract_themes(df["Feedback"].tolist())
    theme_df = pd.DataFrame(themes, columns=["Term", "Frequency"])

    if not theme_df.empty:
        fig3 = px.bar(
            theme_df.head(10).sort_values("Frequency"),
            x="Frequency",
            y="Term",
            orientation="h",
            title="Most Frequent Terms",
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dff7ff",
            margin=dict(t=55, b=10, l=10, r=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-title">🤖 AI Improvement Analysis</div>', unsafe_allow_html=True)

    if api_key:
        if st.button("✨ Generate Management Insights", use_container_width=True):
            sample_for_ai = df["Feedback"].head(80).tolist()

            prompt = f"""
You are analyzing intern feedback for an organization.

Here are the feedback records:
{json.dumps(sample_for_ai, ensure_ascii=False)}

Dataset summary:
- Total records: {total}
- Positive: {positive}
- Neutral: {neutral}
- Negative: {negative}
- Positive rate: {satisfaction:.1f}%
- Average ML confidence: {avg_conf:.1%}

Create a concise management report with these sections:
1. Overall sentiment picture
2. Positive themes
3. Negative themes / pain points
4. Top 5 areas where intern satisfaction could be improved
5. 5 practical actions the internship team can take
6. A short executive summary

Important:
- Use the supplied feedback only.
- Do not invent statistics or quotes.
- If evidence for a claim is weak, say so.
- Keep it professional and actionable.
"""

            with st.spinner("Analyzing feedback with Groq AI..."):
                try:
                    ai_insight = call_groq(prompt, api_key, 2200)
                    st.session_state["ai_insight"] = ai_insight
                except Exception as e:
                    st.error(f"Groq analysis failed: {e}")

    if "ai_insight" in st.session_state:
        st.markdown(
            f'<div class="glass-card">{st.session_state["ai_insight"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Click 'Generate Management Insights' to get AI-powered improvement recommendations.")

    st.markdown('<div class="section-title">📋 Classified Feedback</div>', unsafe_allow_html=True)

    display_df = df[["Feedback", "Sentiment", "Confidence"]].copy()
    display_df["Confidence"] = display_df["Confidence"].map(lambda x: f"{x:.1%}")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    report = make_report(
        df,
        st.session_state.get("ai_insight", ""),
    )

    st.download_button(
        "⬇️ Download Analysis Report",
        data=report,
        file_name="internpulse_feedback_report.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="footer">InternPulse AI • Logistic Regression + TF-IDF + Groq GPT-OSS 20B • Built with Streamlit</div>',
    unsafe_allow_html=True,
)

