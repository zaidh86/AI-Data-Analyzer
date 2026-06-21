import plotly.express as px
import plotly.figure_factory as ff


# =========================
# 📊 HISTOGRAM
# =========================
def create_histogram(df, column):

    fig = px.histogram(
        df,
        x=column,
        title=f"Distribution of {column}"
    )

    return fig


# =========================
# 📦 BOXPLOT
# =========================
def create_boxplot(df, column):

    fig = px.box(
        df,
        y=column,
        title=f"Outlier Analysis - {column}"
    )

    return fig


# =========================
# 📈 SCATTER PLOT
# =========================
def create_scatter_plot(df, x_col, y_col):

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        title=f"{x_col} vs {y_col}"
    )

    return fig


# =========================
# 🔥 CORRELATION HEATMAP
# =========================
def create_correlation_heatmap(df):

    numeric = df.select_dtypes(include="number")

    if len(numeric.columns) < 2:
        return None

    corr = numeric.corr().round(2)

    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.columns),
        annotation_text=corr.values,
        showscale=True
    )

    return fig