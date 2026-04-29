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
# 📂 LOAD DATA (UPDATED 🔥)
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

ai_mode = st.sidebar.toggle("🤖 Enable Real AI (Groq)", value=True)

uploaded_file = st.sidebar.file_uploader(
    "Upload file",
    type=["csv", "json", "xlsx"]  # 🔥 UPDATED
)


# =========================
# 🚀 MAIN APP
# =========================
st.title(f"{BAR_CHART} AI Data Analyzer")
st.caption("Upload a CSV, JSON, or Excel file to explore and generate AI insights.")


if uploaded_file is None:
    st.info(f"{POINT_LEFT} Upload a file from the sidebar to begin.")
    st.stop()


try:
    df = load_file(uploaded_file)  # 🔥 UPDATED
except Exception:
    st.error("Invalid or unsupported file.")
    st.stop()


# Reset insights if new file
if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
    st.session_state.insights = None
    st.session_state.last_file = uploaded_file.name

if "last_persona" not in st.session_state or st.session_state.last_persona != persona:
    st.session_state.insights = None
    st.session_state.last_persona = persona


if df.empty:
    st.warning("The uploaded file is empty.")
    st.stop()


# Handle large dataset
if df.shape[0] > 100000:
    df = df.sample(100000)
    st.warning("Dataset too large. Showing a sample of 100,000 rows.")


numeric_columns = df.select_dtypes(include="number").columns.tolist()

st.success(f"{CHECK} File uploaded successfully.")


# =========================
# 📊 METRICS
# =========================
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
# 🚨 ALERT SYSTEM
# =========================
missing_percent = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100

alerts = []

if missing_percent > 30:
    alerts.append("⚠️ Critical: High missing data detected")

if df.shape[0] < 50:
    alerts.append("⚠️ Dataset is very small — results may be unreliable")

if alerts:
    st.divider()
    st.subheader("🚨 Alerts")
    for alert in alerts:
        st.warning(alert)


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
st.subheader(f"{TRENDING_UP} Interactive Visualization")

if numeric_columns:
    chart_type = st.selectbox("Chart Type", ["Histogram", "Box Plot", "Scatter Plot"])

    if chart_type in {"Histogram", "Box Plot"}:
        col = st.selectbox("Select Column", numeric_columns)

        if chart_type == "Histogram":
            fig = px.histogram(df, x=col, nbins=30)
        else:
            fig = px.box(df, y=col)

        st.plotly_chart(fig, use_container_width=True)

    elif len(numeric_columns) >= 2:
        x = st.selectbox("X-axis", numeric_columns)
        y = st.selectbox("Y-axis", [c for c in numeric_columns if c != x])

        fig = px.scatter(df, x=x, y=y)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No numeric columns found.")


# =========================
# 🔥 CORRELATION
# =========================
st.divider()
st.subheader(f"{FIRE} Correlation Heatmap")

if len(numeric_columns) > 1:
    corr = df[numeric_columns].corr()

    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.round(2).values,
        colorscale="Blues",
        showscale=True,
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Not enough numeric columns.")


# =========================
# 🤖 AI INSIGHTS
# =========================
st.divider()
st.subheader(f"{ROBOT} AI Insights")

if st.button("Generate AI Insights", type="primary"):
    with st.spinner("Analyzing data..."):
        persona = st.sidebar.selectbox(
    "AI Persona",
    ["Analyst", "CEO", "Marketing"]
    )
        st.session_state.insights = get_insights(df,persona)

if st.session_state.get("insights"):
    st.markdown(st.session_state.insights)

    report = generate_report(df, st.session_state.insights,persona)

    st.download_button(
        label="📄 Download PDF Report",
        data=report,
        file_name="data_analysis_report.pdf",
        mime="application/pdf"
    )


# =========================
# 💬 CHAT MODE
# =========================
st.divider()
st.subheader("💬 Ask Questions About Your Data")

query = st.text_input("Ask something...")

if query:
    with st.spinner("Thinking..."):
        response = chat_with_data(df, query)

    st.success(response)