import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# =========================
# 🧩 MISSING VALUES CHART
# =========================
def plot_missing_values(df):

    missing = df.isna().sum()

    missing = missing[missing > 0]

    if missing.empty:
        return None

    fig = px.bar(
        x=missing.index,
        y=missing.values,
        labels={
            "x": "Columns",
            "y": "Missing Values"
        },
        title="Missing Values by Column"
    )

    return fig


# =========================
# ⚠️ OUTLIER CHART
# =========================
def plot_outlier_distribution(df, column):

    fig = px.box(
        df,
        y=column,
        title=f"Outlier Analysis - {column}"
    )

    return fig


# =========================
# 🎯 QUALITY SCORE GAUGE
# =========================
def plot_quality_score(score):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Dataset Quality Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 50], "color": "#dc2626"},
                    {"range": [50, 80], "color": "#f59e0b"},
                    {"range": [80, 100], "color": "#16a34a"}
                ]
            }
        )
    )

    return fig