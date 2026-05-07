import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import time

from ai_insights import get_insights, chat_with_data
from report_generator import generate_report

from utils import (
    calculate_health_score,
    get_confidence_level,
    detect_dataset_type,
    generate_executive_summary,
    generate_priority_insights,
    generate_suggested_questions
)


# =========================
# 🎨 UI ICONS
# =========================
BAR_CHART = "📊"
PAGE = "📄"
PUZZLE = "🧩"
TRENDING_UP = "📈"
FIRE = "🔥"
ROBOT = "🤖"
POINT_LEFT = "👈"
CHECK = "✅"


# =========================
# ⚙️ PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Data Analyzer",
    page_icon=BAR_CHART,
    layout="wide"
)


# =========================
# 📂 LOAD FILE
# =========================
@st.cache_data(show_spinner=False)
def load_file(uploaded_file):

    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "csv":
        return pd.read_csv(uploaded_file)

    elif file_type == "json":
        return pd.read_json(uploaded_file)

    elif file_type == "xlsx":
        return pd.read_excel(uploaded_file)

    else:
        raise ValueError("Unsupported file type")


# =========================
# 🧩 MISSING VALUES TABLE
# =========================
def build_missing_values_table(df):

    missing_count = df.isna().sum()

    missing_percent = (
        missing_count / len(df) * 100
    ).round(2)

    return pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count": missing_count.values,
            "missing_percent": missing_percent.values,
        }
    )


# =========================
# 🧠 SIDEBAR
# =========================
st.sidebar.header("⚙️ Settings")

persona = st.sidebar.selectbox(
    "🎭 AI Persona",
    ["Analyst", "CEO", "Marketing"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload file",
    type=["csv", "json", "xlsx"]
)


# =========================
# 🚀 MAIN APP
# =========================
st.title(f"{BAR_CHART} AI Data Analyzer")

st.caption(
    "Upload a dataset and receive AI-powered analysis, "
    "executive summaries, insights, and conversational intelligence."
)


# =========================
# ⛔ NO FILE
# =========================
if uploaded_file is None:
    st.info(f"{POINT_LEFT} Upload a file from the sidebar to begin.")
    st.stop()


# =========================
# 📂 LOAD DATAFRAME
# =========================
try:
    df = load_file(uploaded_file)

except Exception:
    st.error("Invalid or unsupported file.")
    st.stop()


# =========================
# 🔁 SESSION STATE
# =========================
if "last_file" not in st.session_state:
    st.session_state.last_file = None

if "last_persona" not in st.session_state:
    st.session_state.last_persona = None

if "insights" not in st.session_state:
    st.session_state.insights = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================
# 🧠 DATASET INTELLIGENCE
# =========================
numeric_columns = df.select_dtypes(include="number").columns.tolist()

dataset_type = detect_dataset_type(df)

health_score = calculate_health_score(df)

confidence_level = get_confidence_level(df)

executive_summary = generate_executive_summary(df)

priority_insights = generate_priority_insights(df)

suggested_questions = generate_suggested_questions(dataset_type)


# =========================
# 🔄 AUTO GENERATE INSIGHTS
# =========================
if (
    st.session_state.last_file != uploaded_file.name
    or st.session_state.last_persona != persona
):

    with st.spinner("Generating AI insights..."):

        st.session_state.insights = get_insights(
            df,
            persona
        )

    st.session_state.last_file = uploaded_file.name

    st.session_state.last_persona = persona


# =========================
# ✅ SUCCESS
# =========================
st.success(f"{CHECK} File uploaded successfully.")


# =========================
# 📊 DATASET METRICS
# =========================
metric_cols = st.columns(4)

metric_cols[0].metric(
    "Rows",
    f"{df.shape[0]:,}"
)

metric_cols[1].metric(
    "Columns",
    f"{df.shape[1]:,}"
)

metric_cols[2].metric(
    "Numeric Columns",
    f"{len(numeric_columns):,}"
)

metric_cols[3].metric(
    "Missing Cells",
    f"{int(df.isna().sum().sum()):,}"
)


# =========================
# 🧑‍💼 EXECUTIVE SNAPSHOT
# =========================
st.divider()

st.subheader("🧑‍💼 Executive Snapshot")

exec_cols = st.columns(4)

exec_cols[0].metric(
    "🔥 Strongest Metric",
    executive_summary["Strongest Metric"]
)

exec_cols[1].metric(
    "⚠️ Risk Level",
    executive_summary["Risk Level"]
)

exec_cols[2].metric(
    "📈 Growth Trend",
    executive_summary["Growth Trend"]
)

exec_cols[3].metric(
    "🎯 AI Confidence",
    confidence_level
)

st.progress(health_score / 100)

st.caption(
    f"🧹 Dataset Health Score: {health_score}/100 "
    f"• Dataset Type: {dataset_type}"
)


# =========================
# 📄 DATA PREVIEW
# =========================
preview_col, stats_col = st.columns(2)

with preview_col:

    st.subheader(f"{PAGE} Data Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

with stats_col:

    st.subheader(f"{BAR_CHART} Statistics")

    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )


# =========================
# 🧩 MISSING VALUES
# =========================
st.divider()

st.subheader(f"{PUZZLE} Missing Values")

st.dataframe(
    build_missing_values_table(df),
    use_container_width=True,
    hide_index=True
)


# =========================
# 📈 VISUALIZATION
# =========================
st.divider()

st.subheader(f"{TRENDING_UP} Visualization")

if numeric_columns:

    chart_type = st.selectbox(
        "Chart Type",
        ["Histogram", "Box Plot", "Scatter Plot"]
    )

    if chart_type in {"Histogram", "Box Plot"}:

        col = st.selectbox(
            "Select Column",
            numeric_columns
        )

        if chart_type == "Histogram":

            fig = px.histogram(
                df,
                x=col
            )

        else:

            fig = px.box(
                df,
                y=col
            )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif len(numeric_columns) >= 2:

        x = st.selectbox(
            "X-axis",
            numeric_columns
        )

        y = st.selectbox(
            "Y-axis",
            [c for c in numeric_columns if c != x]
        )

        fig = px.scatter(
            df,
            x=x,
            y=y
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================
# 🔥 PRIORITY INSIGHTS
# =========================
st.divider()

st.subheader("🔥 Priority Insights")

for insight in priority_insights:
    st.info(insight)


# =========================
# 🤖 AI INSIGHTS
# =========================
st.divider()

st.subheader(f"{ROBOT} AI Insights")

if st.session_state.insights:

    st.markdown(
        st.session_state.insights
    )

    report = generate_report(
        df,
        st.session_state.insights,
        persona
    )

    st.download_button(
        label="📄 Download PDF Report",
        data=report,
        file_name="data_analysis_report.pdf",
        mime="application/pdf"
    )


# =========================
# 💡 SUGGESTED QUESTIONS
# =========================
st.divider()

st.subheader("💡 Suggested Questions")

suggestion_cols = st.columns(
    len(suggested_questions)
)

for i, question in enumerate(suggested_questions):

    if suggestion_cols[i].button(question):

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        response = chat_with_data(
            df,
            question,
            persona,
            st.session_state.chat_history
        )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })

        st.rerun()


# =========================
# 💬 CHATBOT
# =========================
st.divider()

st.subheader("💬 Chat with your data")

# Display previous messages
for msg in st.session_state.chat_history:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================
# 💬 USER INPUT
# =========================
user_input = st.chat_input(
    "Ask something about your data..."
)

if user_input:

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):

        response = chat_with_data(
            df,
            user_input,
            persona,
            st.session_state.chat_history
        )

    # =========================
    # 🤖 STREAMING EFFECT
    # =========================
    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        full_response = ""

        words = response.split()

        for word in words:

            full_response += word + " "

            time.sleep(0.02)

            message_placeholder.markdown(
                full_response + "▌"
            )

        message_placeholder.markdown(
            full_response
        )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": full_response
    })


# =========================
# 🗑️ CLEAR CHAT
# =========================
if st.button("🗑️ Clear Chat"):

    st.session_state.chat_history = []

    st.rerun()