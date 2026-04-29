import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from ai_insights import get_insights, chat_with_data
from report_generator import generate_report


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
st.set_page_config(page_title="AI Data Analyzer", page_icon=BAR_CHART, layout="wide")


# =========================
# 📂 LOAD DATA
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


def build_missing_values_table(df):
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)

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
st.caption("Upload a file to explore and get AI-powered insights instantly.")


if uploaded_file is None:
    st.info(f"{POINT_LEFT} Upload a file from the sidebar to begin.")
    st.stop()


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
# 🔄 AUTO-GENERATE INSIGHTS
# =========================
if (
    st.session_state.last_file != uploaded_file.name
    or st.session_state.last_persona != persona
):
    with st.spinner("Generating insights..."):
        st.session_state.insights = get_insights(df, persona)

    st.session_state.last_file = uploaded_file.name
    st.session_state.last_persona = persona


# =========================
# 📊 METRICS
# =========================
numeric_columns = df.select_dtypes(include="number").columns.tolist()

st.success(f"{CHECK} File uploaded successfully.")

metric_cols = st.columns(4)
metric_cols[0].metric("Rows", f"{df.shape[0]:,}")
metric_cols[1].metric("Columns", f"{df.shape[1]:,}")
metric_cols[2].metric("Numeric columns", f"{len(numeric_columns):,}")
metric_cols[3].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")


# =========================
# 📄 PREVIEW + STATS
# =========================
preview_col, stats_col = st.columns(2)

with preview_col:
    st.subheader(f"{PAGE} Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

with stats_col:
    st.subheader(f"{BAR_CHART} Statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)


# =========================
# 🧩 MISSING VALUES
# =========================
st.divider()
st.subheader(f"{PUZZLE} Missing Values")
st.dataframe(build_missing_values_table(df), use_container_width=True, hide_index=True)


# =========================
# 📈 VISUALIZATION
# =========================
st.divider()
st.subheader(f"{TRENDING_UP} Visualization")

if numeric_columns:
    chart_type = st.selectbox("Chart Type", ["Histogram", "Box Plot", "Scatter Plot"])

    if chart_type in {"Histogram", "Box Plot"}:
        col = st.selectbox("Select Column", numeric_columns)

        fig = px.histogram(df, x=col) if chart_type == "Histogram" else px.box(df, y=col)
        st.plotly_chart(fig, use_container_width=True)

    elif len(numeric_columns) >= 2:
        x = st.selectbox("X-axis", numeric_columns)
        y = st.selectbox("Y-axis", [c for c in numeric_columns if c != x])

        fig = px.scatter(df, x=x, y=y)
        st.plotly_chart(fig, use_container_width=True)


# =========================
# 🤖 AI INSIGHTS (AUTO)
# =========================
st.divider()
st.subheader(f"{ROBOT} AI Insights")

if st.session_state.insights:
    st.markdown(st.session_state.insights)

    report = generate_report(df, st.session_state.insights, persona)

    st.download_button(
        label="📄 Download PDF Report",
        data=report,
        file_name="data_analysis_report.pdf",
        mime="application/pdf"
    )


# =========================
# 💬 CHAT WITH MEMORY
# =========================
st.divider()
st.subheader("💬 Chat with your data")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_input = st.chat_input("Ask something about your data...")

if user_input:
    # Save user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.spinner("Thinking..."):
        response = chat_with_data(df, user_input, persona)

    # Save AI response
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)