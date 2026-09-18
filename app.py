import os
import re
import json
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from groq import Groq


# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="Intern Feedback Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS - REFERENCE UI
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Caveat:wght@600&display=swap');

:root {
    --bg: #031327;
    --bg2: #061a31;
    --card: #081d35;
    --card2: #0b2440;
    --line: rgba(116, 203, 255, .20);
    --blue: #42c7f4;
    --blue2: #1ea7df;
    --yellow: #ffdc3d;
    --green: #42e3a1;
    --pink: #ff6682;
    --text: #f3f8ff;
    --muted: #8ca9c2;
}

.stApp {
    background:
        radial-gradient(circle at 83% 5%, rgba(48, 180, 235, .08), transparent 24%),
        radial-gradient(circle at 12% 70%, rgba(255, 220, 61, .045), transparent 20%),
        linear-gradient(135deg, #020e1e 0%, #04172c 48%, #021022 100%);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stToolbar"] {
    display: none;
}

.block-container {
    max-width: 100%;
    padding: 0 28px 45px 28px;
}

section[data-testid="stSidebar"] {
    width: 300px !important;
    background:
        linear-gradient(180deg, #061a30 0%, #031426 100%);
    border-right: 1px solid rgba(70, 193, 242, .18);
}

section[data-testid="stSidebar"] > div {
    padding: 0 15px 20px 15px;
}

section[data-testid="stSidebar"] * {
    color: #eaf6ff;
}

/* hide Streamlit default sidebar decorations */
[data-testid="stSidebarNav"] {
    display: none;
}

.sidebar-brand {
    padding: 25px 12px 20px 12px;
    border-bottom: 1px solid rgba(115, 200, 245, .12);
    margin-bottom: 15px;
}

.brand-mark {
    font-size: 28px;
    margin-right: 8px;
}

.brand-name {
    font-size: 20px;
    font-weight: 800;
    color: #f6fbff;
}

.brand-name .yellow {
    color: var(--yellow);
}

.brand-name .blue {
    color: var(--blue);
}

.brand-sub {
    color: #7896ae;
    font-size: 11px;
    margin-top: 5px;
    margin-left: 40px;
}

.nav-title {
    color: #607f98;
    text-transform: uppercase;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .13em;
    margin: 16px 9px 8px 9px;
}

div[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    min-height: 46px;
    margin: 3px 0;
    padding: 0 14px;
    border-radius: 11px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    color: #d7e9f6 !important;
    text-align: left !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    transform: none !important;
}

div[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(56, 189, 248, .09) !important;
    border-color: rgba(56, 189, 248, .14) !important;
}

div[data-testid="stSidebar"] .nav-active > div > button,
div[data-testid="stSidebar"] .nav-active button {
    background: linear-gradient(90deg, #43c7f3, #38b6eb) !important;
    color: #06203a !important;
    box-shadow: 0 8px 25px rgba(55, 190, 241, .16) !important;
}

.side-message {
    margin: 80px 5px 10px 5px;
    padding: 22px 18px;
    min-height: 150px;
    border: 1px solid rgba(255, 220, 61, .28);
    border-radius: 17px;
    background: linear-gradient(145deg, rgba(8, 29, 51, .92), rgba(3, 17, 32, .95));
    box-shadow: 0 12px 35px rgba(0,0,0,.22);
    position: relative;
    overflow: hidden;
}

.side-message:after {
    content: "";
    position: absolute;
    width: 100px;
    height: 35px;
    border-bottom: 3px solid var(--yellow);
    border-radius: 50%;
    left: 18px;
    bottom: 13px;
    transform: rotate(-5deg);
}

.side-message p {
    margin: 2px 0;
    font-family: 'Caveat', cursive;
    font-size: 20px;
    color: #f2f7ff;
}

.side-message p:nth-child(2) { color: var(--yellow); }
.side-message p:nth-child(3) { color: #f4d84e; }

.topbar {
    height: 88px;
    margin: 0 -28px 24px -28px;
    padding: 0 30px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(104, 191, 239, .14);
    background: rgba(2, 15, 31, .72);
}

.topbar-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.logo-brain {
    font-size: 39px;
}

.top-title {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -.03em;
}

.top-title .yellow { color: var(--yellow); }
.top-title .blue { color: var(--blue); }

.top-subtitle {
    font-size: 12px;
    color: #7695ae;
    margin-top: 2px;
}

.top-slogan {
    font-family: 'Caveat', cursive;
    font-size: 20px;
    color: var(--yellow);
    border-bottom: 2px solid rgba(255,220,61,.6);
    padding-bottom: 3px;
}

.main-title {
    font-size: 28px;
    font-weight: 800;
    margin: 3px 0 3px 0;
}

.main-title .yellow { color: var(--yellow); }
.main-title .blue { color: var(--blue); }

.main-subtitle {
    color: #91aec4;
    font-size: 14px;
    margin-bottom: 21px;
}

.metric {
    min-height: 115px;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 16px 17px;
    background:
        radial-gradient(circle at 85% 30%, rgba(43, 151, 209, .08), transparent 35%),
        linear-gradient(145deg, rgba(9, 30, 53, .94), rgba(4, 20, 38, .96));
    position: relative;
    overflow: hidden;
}

.metric:after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 4px;
}

.metric-blue:after { background: #42c7f4; }
.metric-green:after { background: #42e3a1; }
.metric-yellow:after { background: #ffdc3d; }
.metric-pink:after { background: #ff6682; }

.metric-top {
    display: flex;
    align-items: center;
    gap: 12px;
}

.metric-icon {
    width: 49px;
    height: 49px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 23px;
    background: rgba(67,199,243,.15);
    border: 1px solid rgba(67,199,243,.25);
}

.metric-green .metric-icon { background: rgba(66,227,161,.13); }
.metric-yellow .metric-icon { background: rgba(255,220,61,.14); }
.metric-pink .metric-icon { background: rgba(255,102,130,.13); }

.metric-label {
    color: #d9e9f4;
    font-size: 13px;
    font-weight: 600;
}

.metric-number {
    font-size: 27px;
    font-weight: 800;
    line-height: 1.05;
    margin-top: 5px;
}

.metric-number.green { color: #ecfff8; }
.metric-number.yellow { color: #fff4b0; }
.metric-number.pink { color: #ffc0cc; }

.metric-small {
    font-size: 11px;
    color: #6f8da5;
    margin-top: 3px;
}

.panel {
    border: 1px solid var(--line);
    border-radius: 15px;
    background: linear-gradient(145deg, rgba(7, 28, 49, .95), rgba(3, 17, 33, .96));
    overflow: hidden;
    min-height: 275px;
}

.panel-title {
    padding: 15px 17px 8px 17px;
    font-size: 15px;
    font-weight: 800;
}

.panel-body {
    padding: 4px 17px 16px 17px;
}

.quick {
    min-height: 275px;
    border: 1px solid var(--line);
    border-radius: 15px;
    padding: 15px 16px;
    background: linear-gradient(145deg, rgba(7, 29, 50, .95), rgba(3, 18, 34, .97));
}

.quick-title {
    font-size: 15px;
    font-weight: 800;
    margin-bottom: 13px;
}

.ai-panel {
    min-height: 275px;
    border: 1px solid var(--line);
    border-radius: 15px;
    padding: 15px 16px;
    background: linear-gradient(145deg, rgba(7, 29, 50, .95), rgba(3, 18, 34, .97));
}

.ai-title {
    font-size: 15px;
    font-weight: 800;
    margin-bottom: 12px;
}

.ai-text {
    background: rgba(16, 46, 76, .65);
    border-radius: 10px;
    padding: 13px;
    color: #a8c1d5;
    font-size: 12px;
    line-height: 1.55;
    min-height: 130px;
}

.section-space {
    height: 17px;
}

.stButton > button {
    border-radius: 10px !important;
    border: 1px solid rgba(67,199,243,.35) !important;
    background: linear-gradient(135deg, #42c7f4, #1da8df) !important;
    color: #032038 !important;
    font-weight: 800 !important;
    min-height: 43px;
    box-shadow: 0 8px 22px rgba(24, 161, 218, .12) !important;
}

.stButton > button:hover {
    border-color: var(--yellow) !important;
    box-shadow: 0 8px 26px rgba(255, 220, 61, .15) !important;
}

.quick .stButton:nth-child(2) > button {
    background: linear-gradient(135deg, #ffe85b, #ffd52e) !important;
}

.ai-panel .stButton > button {
    background: linear-gradient(135deg, #ffe85b, #ffd52e) !important;
}

.stFileUploader section {
    background: #071c31 !important;
    border: 1px dashed rgba(66,199,244,.35) !important;
    border-radius: 11px !important;
}

.stTextArea textarea,
.stTextInput input {
    background: #071c31 !important;
    color: #effaff !important;
    border: 1px solid rgba(66,199,244,.25) !important;
    border-radius: 11px !important;
}

.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: var(--blue) !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background: #071c31 !important;
    color: #effaff !important;
    border-color: rgba(66,199,244,.25) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 5px;
}

.stTabs [data-baseweb="tab"] {
    color: #7f9bb0;
}

.stTabs [aria-selected="true"] {
    color: var(--yellow) !important;
}

.data-panel {
    border: 1px solid var(--line);
    border-radius: 15px;
    background: linear-gradient(145deg, rgba(7,28,49,.95), rgba(3,17,33,.96));
    padding: 14px;
}

.footer {
    color: #58768e;
    font-size: 11px;
    text-align: center;
    padding-top: 25px;
}

@media (max-width: 1000px) {
    .top-slogan { display: none; }
    .top-title { font-size: 20px; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# TRAINING DATA - EXACTLY MATCHED PAIRS
# ============================================================
TRAINING_PAIRS = [
    ("The mentor was supportive and always answered my questions.", "Positive"),
    ("I learned a lot and received useful feedback every week.", "Positive"),
    ("The internship tasks were interesting and relevant to my career.", "Positive"),
    ("The team welcomed me and gave me clear guidance.", "Positive"),
    ("I enjoyed working on real projects and improving my skills.", "Positive"),
    ("The supervisor gave constructive feedback and helped me grow.", "Positive"),
    ("Communication with the team was excellent.", "Positive"),
    ("The training sessions were practical and useful.", "Positive"),
    ("I was given meaningful responsibilities and learned quickly.", "Positive"),
    ("The work environment was friendly and motivating.", "Positive"),
    ("My mentor explained difficult concepts clearly.", "Positive"),
    ("The internship improved my technical and communication skills.", "Positive"),
    ("The tasks were well organized and easy to understand.", "Positive"),
    ("I received recognition for completing my work.", "Positive"),
    ("The company provided a great learning environment.", "Positive"),

    ("I felt ignored when I asked for help.", "Negative"),
    ("The tasks were repetitive and did not teach me anything.", "Negative"),
    ("My mentor rarely gave feedback.", "Negative"),
    ("The workload was unreasonable and stressful.", "Negative"),
    ("Communication from the team was poor.", "Negative"),
    ("I was not given clear instructions.", "Negative"),
    ("The internship did not provide enough practical experience.", "Negative"),
    ("The deadlines were unrealistic.", "Negative"),
    ("My work was never recognized.", "Negative"),
    ("The training material was confusing and outdated.", "Negative"),
    ("I had to solve problems without any guidance.", "Negative"),
    ("The environment was stressful and unwelcoming.", "Negative"),
    ("I did not receive useful feedback.", "Negative"),
    ("The assigned tasks were unrelated to my learning goals.", "Negative"),
    ("There was too much work and not enough support.", "Negative"),

    ("The experience was okay but could be improved.", "Neutral"),
    ("The internship was average and not very exciting.", "Neutral"),
    ("Some tasks were useful but others were repetitive.", "Neutral"),
    ("Communication was sometimes good and sometimes confusing.", "Neutral"),
    ("The mentor helped occasionally, but feedback was limited.", "Neutral"),
    ("The workload was manageable, although some deadlines were tight.", "Neutral"),
    ("I learned a few things but expected more practical work.", "Neutral"),
    ("The experience was neither great nor terrible.", "Neutral"),
]

TRAINING_DATA = pd.DataFrame(TRAINING_PAIRS, columns=["text", "label"])


# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def get_model():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    model.fit(TRAINING_DATA["text"], TRAINING_DATA["label"])
    return model


MODEL = get_model()


def clean_text(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def predict_sentiment(text):
    text = clean_text(text)
    prediction = MODEL.predict([text])[0]
    probs = MODEL.predict_proba([text])[0]
    classes = MODEL.classes_
    confidence = float(probs[list(classes).index(prediction)])
    return prediction, confidence


def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


def groq_analysis(prompt):
    key = get_api_key()
    if not key:
        return None, "Groq API key is not configured."

    client = Groq(api_key=key)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert HR analytics assistant. "
                    "Analyze only the feedback supplied by the user. "
                    "Never invent statistics, quotes, or facts. "
                    "Give concise, professional and actionable recommendations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_completion_tokens=1800,
    )
    return response.choices[0].message.content, None


def common_terms(texts, n=6):
    stop = {
        "the","and","was","were","this","that","with","from","have","had",
        "very","but","not","are","our","they","their","too","did","its",
        "into","you","your","about","more","some","just","been","also",
        "when","what","would","could","can","my","me","i","a","an","to",
        "of","in","on","is","it","for","than","there","very"
    }
    words = []
    for text in texts:
        words.extend(
            w for w in re.findall(r"[a-zA-Z]{4,}", text.lower())
            if w not in stop
        )
    return Counter(words).most_common(n)


def make_chart_layout(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dceefa", size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    fig.update_xaxes(
        gridcolor="rgba(120,180,220,.10)",
        zerolinecolor="rgba(120,180,220,.10)",
        tickfont=dict(color="#8da8bd"),
    )
    fig.update_yaxes(
        gridcolor="rgba(120,180,220,.10)",
        zerolinecolor="rgba(120,180,220,.10)",
        tickfont=dict(color="#8da8bd"),
    )
    return fig


# ============================================================
# SESSION STATE
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "feedback_df" not in st.session_state:
    st.session_state.feedback_df = None

if "single_result" not in st.session_state:
    st.session_state.single_result = None

if "ai_report" not in st.session_state:
    st.session_state.ai_report = ""


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div>
            <span class="brand-mark">🧠</span>
            <span class="brand-name">
                Intern <span class="yellow">Feedback</span> <span class="blue">AI</span>
            </span>
        </div>
        <div class="brand-sub">Turn feedback into better experiences</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-title">Workspace</div>', unsafe_allow_html=True)

    nav_items = [
        ("🏠  Dashboard", "Dashboard"),
        ("📄  Upload Feedback", "Upload Feedback"),
        ("🔍  Analyze Single Feedback", "Analyze Single Feedback"),
        ("▣  View Classified Data", "View Classified Data"),
        ("📈  Generate Report", "Generate Report"),
    ]

    for label, page in nav_items:
        if st.button(
            label,
            key=f"nav_{page}",
            use_container_width=True,
        ):
            st.session_state.page = page
            st.rerun()

    st.markdown("""
    <div class="side-message">
        <p>Real Feedback</p>
        <p>Real Insights</p>
        <p>Better Intern Experience</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TOP HEADER
# ============================================================
st.markdown("""
<div class="topbar">
    <div class="topbar-left">
        <div class="logo-brain">🧠</div>
        <div>
            <div class="top-title">
                Intern <span class="yellow">Feedback</span> <span class="blue">Analyzer</span>
            </div>
            <div class="top-subtitle">Turn feedback into better experiences</div>
        </div>
    </div>
    <div class="top-slogan">Better Feedback → Better Interns → A Stronger Team ✨</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# DASHBOARD
# ============================================================
def get_active_df():
    if st.session_state.feedback_df is not None:
        return st.session_state.feedback_df.copy()

    demo = TRAINING_DATA.rename(columns={"text": "Feedback"}).copy()
    demo["Sentiment"], demo["Confidence"] = zip(
        *demo["Feedback"].map(predict_sentiment)
    )
    return demo


def dashboard():
    df = get_active_df()

    total = len(df)
    positive = int((df["Sentiment"] == "Positive").sum())
    neutral = int((df["Sentiment"] == "Neutral").sum())
    negative = int((df["Sentiment"] == "Negative").sum())

    positive_pct = positive / total * 100 if total else 0
    neutral_pct = neutral / total * 100 if total else 0
    negative_pct = negative / total * 100 if total else 0

    st.markdown("""
    <div class="main-title">
        👋 Welcome to <span class="yellow">Intern Feedback</span> <span class="blue">Analyzer</span>
    </div>
    <div class="main-subtitle">
        Analyze intern feedback using AI and get meaningful insights to improve your internship program.
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4, gap="medium")

    cards = [
        (m1, "metric-blue", "💬", "Total Feedbacks", total, "in dataset", ""),
        (m2, "metric-green", "😊", "Positive", positive, f"({positive_pct:.1f}%)", "green"),
        (m3, "metric-yellow", "😐", "Neutral", neutral, f"({neutral_pct:.1f}%)", "yellow"),
        (m4, "metric-pink", "☹️", "Negative", negative, f"({negative_pct:.1f}%)", "pink"),
    ]

    for col, css, icon, label, value, small, value_css in cards:
        with col:
            st.markdown(f"""
            <div class="metric {css}">
                <div class="metric-top">
                    <div class="metric-icon">{icon}</div>
                    <div>
                        <div class="metric-label">{label}</div>
                        <div class="metric-number {value_css}">{value}</div>
                        <div class="metric-small">{small}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)

    p1, p2, p3 = st.columns([1.15, 1.15, 1.0], gap="medium")

    counts = pd.DataFrame({
        "Sentiment": ["Positive", "Neutral", "Negative"],
        "Count": [positive, neutral, negative],
    })

    with p1:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">📊 Sentiment Distribution</div>
            <div class="panel-body">
        """, unsafe_allow_html=True)

        fig = go.Figure(go.Pie(
            labels=counts["Sentiment"],
            values=counts["Count"],
            hole=.62,
            textinfo="none",
            marker=dict(
                colors=["#42e3a1", "#ffdc3d", "#ff6682"],
                line=dict(color="#061a30", width=2),
            ),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=0,b=0),
            height=220,
            legend=dict(
                orientation="v",
                x=.92,
                y=.5,
                font=dict(color="#dceefa", size=10),
            ),
            annotations=[dict(
                text=f"<b>{total}</b><br><span style='font-size:10px'>Total</span>",
                x=.5, y=.5, showarrow=False,
                font=dict(color="#eaf6ff", size=18)
            )],
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("</div></div>", unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">📊 Sentiment Comparison</div>
            <div class="panel-body">
        """, unsafe_allow_html=True)

        fig = go.Figure(go.Bar(
            x=counts["Sentiment"],
            y=counts["Count"],
            text=counts["Count"],
            textposition="outside",
            marker_color=["#42e3a1", "#ffdc3d", "#ff6682"],
            marker_line_width=0,
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=220,
            margin=dict(l=0,r=0,t=20,b=10),
            font=dict(color="#dceefa", size=10),
            yaxis=dict(
                gridcolor="rgba(120,180,220,.10)",
                zeroline=False,
                tickfont=dict(color="#8da8bd"),
            ),
            xaxis=dict(
                tickfont=dict(color="#8da8bd"),
                showgrid=False,
            ),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("</div></div>", unsafe_allow_html=True)

    with p3:
        terms = common_terms(df["Feedback"].tolist(), 6)
        term_labels = [x[0].title() for x in terms]
        term_values = [x[1] for x in terms]

        st.markdown("""
        <div class="panel">
            <div class="panel-title">💡 Common Feedback Themes</div>
            <div class="panel-body">
        """, unsafe_allow_html=True)

        if terms:
            fig = go.Figure(go.Pie(
                labels=term_labels,
                values=term_values,
                hole=.56,
                textinfo="none",
                marker=dict(
                    colors=["#42c7f4","#ffdc3d","#42e3a1","#9b7cff","#ff9b4a","#b9c8d6"],
                    line=dict(color="#061a30", width=2),
                ),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0,r=0,t=0,b=0),
                height=220,
                legend=dict(
                    orientation="v",
                    x=.91,
                    y=.5,
                    font=dict(color="#dceefa", size=9),
                ),
                annotations=[dict(
                    text="Top<br>Themes",
                    x=.5, y=.5, showarrow=False,
                    font=dict(color="#eaf6ff", size=12)
                )],
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)

    q, ai = st.columns([1.0, 1.0], gap="medium")

    with q:
        st.markdown("""
        <div class="quick">
            <div class="quick-title">⚡ Quick Actions</div>
        """, unsafe_allow_html=True)

        if st.button("📤  Upload Feedback CSV", key="dash_upload", use_container_width=True):
            st.session_state.page = "Upload Feedback"
            st.rerun()

        if st.button("✨  Generate AI Report", key="dash_report", use_container_width=True):
            st.session_state.page = "Generate Report"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with ai:
        st.markdown("""
        <div class="ai-panel">
            <div class="ai-title">✨ AI Insights & Recommendations</div>
            <div class="ai-text">
                Based on the current feedback, the dashboard separates positive,
                neutral and negative experiences so management can quickly identify
                satisfaction patterns, communication concerns, workload issues and
                learning opportunities.
            </div>
        """, unsafe_allow_html=True)

        if st.button("🧠  View Full AI Report", key="dash_ai", use_container_width=True):
            st.session_state.page = "Generate Report"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="data-panel">
        <div class="panel-title">📋 Classified Feedback</div>
    """, unsafe_allow_html=True)

    display = df[["Feedback", "Sentiment", "Confidence"]].head(8).copy()
    display["Confidence"] = display["Confidence"].map(lambda x: f"{x:.1%}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# UPLOAD PAGE
# ============================================================
def upload_page():
    st.markdown('<div class="main-title">📤 <span class="yellow">Upload</span> <span class="blue">Feedback</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Upload your intern feedback CSV and classify every review automatically.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="data-panel">
        <b>Recommended CSV format</b><br>
        Use a column named <code>feedback</code>, <code>text</code>, <code>review</code>,
        <code>comment</code>, or select your text column after uploading.
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    file = st.file_uploader("Choose CSV file", type=["csv"])

    if file:
        try:
            raw = pd.read_csv(file)

            candidates = [
                c for c in raw.columns
                if c.lower().strip() in {"feedback", "text", "review", "comment", "comments"}
            ]

            text_col = candidates[0] if candidates else st.selectbox(
                "Select the feedback column",
                list(raw.columns),
            )

            df = raw.copy()
            df["Feedback"] = df[text_col].fillna("").astype(str).map(clean_text)
            df = df[df["Feedback"].str.len() > 0].copy()

            predictions = df["Feedback"].map(predict_sentiment)
            df["Sentiment"] = [x[0] for x in predictions]
            df["Confidence"] = [x[1] for x in predictions]

            st.session_state.feedback_df = df

            st.success(f"Successfully analyzed {len(df)} feedback records.")
            st.dataframe(
                df[["Feedback", "Sentiment", "Confidence"]],
                use_container_width=True,
                hide_index=True,
            )

            if st.button("🏠 View Dashboard", use_container_width=True):
                st.session_state.page = "Dashboard"
                st.rerun()

        except Exception as e:
            st.error(f"Could not process the CSV: {e}")


# ============================================================
# SINGLE FEEDBACK
# ============================================================
def single_page():
    st.markdown('<div class="main-title">🔍 <span class="yellow">Analyze</span> <span class="blue">Single Feedback</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Classify one intern review and optionally ask Groq AI for deeper insight.</div>', unsafe_allow_html=True)

    text = st.text_area(
        "Intern feedback",
        height=180,
        placeholder="Example: My mentor was helpful, but I wish I had received more practical tasks and weekly feedback.",
    )

    if st.button("⚡ Analyze Sentiment", use_container_width=True):
        if not text.strip():
            st.warning("Please enter feedback first.")
            return

        sentiment, confidence = predict_sentiment(text)
        st.session_state.single_result = (sentiment, confidence, text)

    if st.session_state.single_result:
        sentiment, confidence, original = st.session_state.single_result
        icon = {"Positive":"😊", "Neutral":"😐", "Negative":"☹️"}[sentiment]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric metric-green"><div class="metric-top"><div class="metric-icon">{icon}</div><div><div class="metric-label">Sentiment</div><div class="metric-number">{sentiment}</div><div class="metric-small">ML classification</div></div></div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric metric-blue"><div class="metric-top"><div class="metric-icon">🎯</div><div><div class="metric-label">Confidence</div><div class="metric-number">{confidence:.1%}</div><div class="metric-small">Model confidence</div></div></div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric metric-yellow"><div class="metric-top"><div class="metric-icon">🤖</div><div><div class="metric-label">AI Layer</div><div class="metric-number">Groq</div><div class="metric-small">GPT-OSS 20B</div></div></div></div>', unsafe_allow_html=True)

        if st.button("✨ Get Groq AI Insight", use_container_width=True):
            prompt = f"""
Analyze this intern feedback:

"{original}"

Return exactly these sections:
### Main Theme
### Positive Signal
### Improvement Area
### Recommended Action
### Short Summary

Keep it concise and do not invent facts.
"""
            with st.spinner("Generating AI insight..."):
                result, error = groq_analysis(prompt)

            if error:
                st.error(error)
            else:
                st.markdown('<div class="data-panel">', unsafe_allow_html=True)
                st.markdown(result)
                st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# CLASSIFIED DATA
# ============================================================
def classified_page():
    st.markdown('<div class="main-title">📋 <span class="yellow">Classified</span> <span class="blue">Feedback</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Review sentiment predictions and confidence scores for your dataset.</div>', unsafe_allow_html=True)

    df = st.session_state.feedback_df

    if df is None:
        st.info("No uploaded dataset yet. The dashboard demo data is available, but upload your CSV to classify your own feedback.")
        return

    show = df[["Feedback", "Sentiment", "Confidence"]].copy()
    show["Confidence"] = show["Confidence"].map(lambda x: f"{x:.1%}")
    st.dataframe(show, use_container_width=True, hide_index=True)

    csv = df[["Feedback", "Sentiment", "Confidence"]].to_csv(index=False)
    st.download_button(
        "⬇️ Download Classified CSV",
        csv,
        "classified_intern_feedback.csv",
        "text/csv",
        use_container_width=True,
    )


# ============================================================
# REPORT
# ============================================================
def report_page():
    st.markdown('<div class="main-title">📈 <span class="yellow">Generate</span> <span class="blue">AI Report</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Use Groq AI to turn classified feedback into practical management recommendations.</div>', unsafe_allow_html=True)

    df = get_active_df()

    total = len(df)
    pos = int((df["Sentiment"] == "Positive").sum())
    neu = int((df["Sentiment"] == "Neutral").sum())
    neg = int((df["Sentiment"] == "Negative").sum())

    st.markdown(f"""
    <div class="data-panel">
        <b>Current dataset:</b> {total} feedback records &nbsp; • &nbsp;
        <b>Positive:</b> {pos} &nbsp; • &nbsp;
        <b>Neutral:</b> {neu} &nbsp; • &nbsp;
        <b>Negative:</b> {neg}
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    if st.button("✨ Generate Full AI Report", use_container_width=True):
        feedbacks = df["Feedback"].head(100).tolist()

        prompt = f"""
You are an HR analytics expert analyzing intern feedback.

Feedback records:
{json.dumps(feedbacks, ensure_ascii=False)}

ML summary:
Total: {total}
Positive: {pos}
Neutral: {neu}
Negative: {neg}

Create a professional report with:
1. Overall sentiment picture
2. Positive themes
3. Negative themes and pain points
4. Top areas where intern satisfaction can be improved
5. Five practical actions for the internship team
6. Executive summary

Use only the supplied feedback and summary.
Do not invent statistics, quotes, or facts.
"""
        with st.spinner("Groq is analyzing the feedback..."):
            result, error = groq_analysis(prompt)

        if error:
            st.error(error)
        else:
            st.session_state.ai_report = result

    if st.session_state.ai_report:
        st.markdown('<div class="data-panel">', unsafe_allow_html=True)
        st.markdown(st.session_state.ai_report)
        st.markdown('</div>', unsafe_allow_html=True)

        report_text = st.session_state.ai_report
        st.download_button(
            "⬇️ Download AI Report",
            report_text,
            "intern_feedback_ai_report.txt",
            "text/plain",
            use_container_width=True,
        )


# ============================================================
# ROUTER
# ============================================================
page = st.session_state.page

if page == "Dashboard":
    dashboard()
elif page == "Upload Feedback":
    upload_page()
elif page == "Analyze Single Feedback":
    single_page()
elif page == "View Classified Data":
    classified_page()
elif page == "Generate Report":
    report_page()
else:
    dashboard()

st.markdown(
    '<div class="footer">Intern Feedback Analyzer • TF-IDF + Logistic Regression + Groq GPT-OSS 20B • Streamlit</div>',
    unsafe_allow_html=True,
)
