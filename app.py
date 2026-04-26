import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from ai_insights import get_insights
from report_generator import generate_report


BAR_CHART = "\U0001F4CA"
PAGE = "\U0001F4C4"
PUZZLE = "\U0001F9E9"
TRENDING_UP = "\U0001F4C8"
FIRE = "\U0001F525"
ROBOT = "\U0001F916"
INBOX = "\U0001F4E5"
POINT_LEFT = "\U0001F448"
CHECK = "\U00002705"


st.set_page_config(page_title="AI Data Analyzer", page_icon=BAR_CHART, layout="wide")


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


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


st.title(f"{BAR_CHART} AI Data Analyzer")
st.caption("Upload a CSV file, explore the data, generate charts, and create AI-assisted insights.")

st.sidebar.header("Upload & Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info(f"{POINT_LEFT} Upload a CSV file from the sidebar to begin.")
    st.stop()

try:
    df = load_csv(uploaded_file)
except Exception as exc:
    st.error(f"Invalid CSV file: {exc}")
    st.stop()

if df.empty:
    st.warning("The uploaded file is empty.")
    st.stop()

numeric_columns = df.select_dtypes(include="number").columns.tolist()

st.success(f"{CHECK} File uploaded successfully.")

metric_cols = st.columns(4)
metric_cols[0].metric("Rows", f"{df.shape[0]:,}")
metric_cols[1].metric("Columns", f"{df.shape[1]:,}")
metric_cols[2].metric("Numeric columns", f"{len(numeric_columns):,}")
metric_cols[3].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

preview_col, stats_col = st.columns(2)

with preview_col:
    st.subheader(f"{PAGE} Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

with stats_col:
    st.subheader(f"{BAR_CHART} Statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

st.divider()

st.subheader(f"{PUZZLE} Missing Values")
st.dataframe(build_missing_values_table(df), use_container_width=True, hide_index=True)

st.divider()

st.subheader(f"{TRENDING_UP} Interactive Visualization")

if numeric_columns:
    chart_type = st.selectbox("Select chart type", ["Histogram", "Box Plot", "Scatter Plot"])

    if chart_type in {"Histogram", "Box Plot"}:
        selected_column = st.selectbox("Select numeric column", numeric_columns)
        if chart_type == "Histogram":
            fig = px.histogram(df, x=selected_column, nbins=30, title=f"Distribution of {selected_column}")
        else:
            fig = px.box(df, y=selected_column, title=f"Box plot of {selected_column}")

        st.plotly_chart(fig, use_container_width=True)

    elif len(numeric_columns) >= 2:
        x_axis = st.selectbox("X-axis", numeric_columns)
        y_axis_options = [column for column in numeric_columns if column != x_axis]
        y_axis = st.selectbox("Y-axis", y_axis_options)

        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Scatter plots need at least two numeric columns.")
else:
    st.warning("No numeric columns found for visualization.")

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
    fig.update_layout(title="Numeric Column Correlations")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Not enough numeric columns for a correlation heatmap.")

st.divider()

st.subheader(f"{ROBOT} AI Insights")

if st.button("Generate AI Insights", type="primary"):
    with st.spinner("Analyzing data..."):
        st.session_state.insights = get_insights(df)

if "insights" in st.session_state:
    st.markdown(st.session_state.insights)

    report = generate_report(df, st.session_state.insights)
    st.download_button(
        label=f"{INBOX} Download Report",
        data=report,
        file_name="data_analysis_report.txt",
        mime="text/plain",
    )
