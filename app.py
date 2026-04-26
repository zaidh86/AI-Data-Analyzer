import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from ai_insights import get_insights
from report_generator import generate_report

# Page config
st.set_page_config(page_title="AI Data Analyzer", layout="wide")

# Title
st.title("📊 AI Data Analyzer")

# Sidebar
st.sidebar.header("Upload & Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    # Safe CSV loading
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    except Exception:
        st.error("Invalid file. Please upload a proper CSV.")
        st.stop()

    # Empty file check
    if df.empty:
        st.warning("The uploaded file is empty.")
        st.stop()

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

    with col2:
        st.subheader("📊 Statistics")
        st.dataframe(df.describe())

    st.divider()

    # Missing Values
    st.subheader("🧩 Missing Values")
    st.dataframe(df.isnull().sum())

    st.divider()

    # Visualization
    st.subheader("📈 Interactive Visualization")

    numeric_columns = df.select_dtypes(include='number').columns

    if len(numeric_columns) > 0:
        column = st.selectbox("Select column", numeric_columns)

        chart_type = st.selectbox(
            "Select chart type",
            ["Histogram", "Box Plot", "Scatter Plot"]
        )

        if chart_type == "Histogram":
            fig = px.histogram(df, x=column, nbins=30)

        elif chart_type == "Box Plot":
            fig = px.box(df, y=column)

        elif chart_type == "Scatter Plot":
            x_axis = st.selectbox("X-axis", numeric_columns)
            y_axis = st.selectbox("Y-axis", numeric_columns)

            if x_axis == y_axis:
                st.warning("X and Y axis must be different.")
                st.stop()

            fig = px.scatter(df, x=x_axis, y=y_axis)

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No numeric columns found for visualization.")

    st.divider()

    # Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap")

    if len(numeric_columns) > 1:
        corr = df[numeric_columns].corr()

        fig = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            annotation_text=corr.round(2).values,
            showscale=True
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough numeric columns for correlation heatmap.")

    st.divider()

    # AI Insights
    st.subheader("🤖 AI Insights")

    if st.button("Generate AI Insights"):
        with st.spinner("Analyzing data..."):
            insights = get_insights(df)

        st.write(insights)

        report = generate_report(df, insights)

        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="data_analysis_report.txt",
            mime="text/plain"
        )

else:
    st.info("👈 Upload a CSV file from the sidebar to begin.")