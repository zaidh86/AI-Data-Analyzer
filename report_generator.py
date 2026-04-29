from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from io import BytesIO

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def generate_report(df, insights, persona="Analyst"):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    content = []

    # =========================
    # 🎭 PERSONA HEADER
    # =========================
    if persona == "CEO":
        subtitle = "Strategic Business Report"
    elif persona == "Marketing":
        subtitle = "Marketing & Growth Insights Report"
    else:
        subtitle = "Detailed Data Analysis Report"

    # =========================
    # 🏷️ TITLE
    # =========================
    content.append(Paragraph("AI DATA ANALYSIS REPORT", styles["Title"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(subtitle, styles["Italic"]))
    content.append(Spacer(1, 20))

    # =========================
    # 📊 SUMMARY TABLE
    # =========================
    data = [
        ["Metric", "Value"],
        ["Rows", str(df.shape[0])],
        ["Columns", str(df.shape[1])],
        ["Missing Values", str(int(df.isna().sum().sum()))],
    ]

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))

    content.append(table)
    content.append(Spacer(1, 20))

    # =========================
    # 🚨 ALERTS
    # =========================
    missing_percent = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100

    alerts = []
    if missing_percent > 30:
        alerts.append("⚠️ High missing data detected — results may be unreliable.")
    if df.shape[0] < 50:
        alerts.append("⚠️ Dataset is very small — insights may lack statistical significance.")

    if alerts:
        content.append(Paragraph("Alerts", styles["Heading2"]))
        content.append(Spacer(1, 10))
        for alert in alerts:
            content.append(Paragraph(alert, styles["Normal"]))
            content.append(Spacer(1, 6))
        content.append(Spacer(1, 20))

    # =========================
    # 📈 HISTOGRAMS
    # =========================
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    for col in numeric_cols[:2]:
        fig, ax = plt.subplots()
        df[col].hist(ax=ax)
        ax.set_title(f"Distribution of {col}")

        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close()

        img_buffer.seek(0)

        content.append(Paragraph(f"Distribution of {col}", styles["Heading2"]))
        content.append(Image(img_buffer, width=400, height=250))
        content.append(Spacer(1, 20))

    # =========================
    # 🔥 CORRELATION HEATMAP
    # =========================
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")

        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close()

        img_buffer.seek(0)

        content.append(Paragraph("Correlation Heatmap", styles["Heading2"]))
        content.append(Image(img_buffer, width=400, height=300))
        content.append(Spacer(1, 20))

    # =========================
    # 🧠 AI INSIGHTS
    # =========================
    content.append(Paragraph("AI Insights & Analysis", styles["Heading2"]))
    content.append(Spacer(1, 10))

    for line in insights.split("\n"):
        if line.strip():
            if line.startswith("📌") or line.startswith("🤖") or line.startswith("💡"):
                content.append(Paragraph(f"<b>{line}</b>", styles["Normal"]))
            else:
                content.append(Paragraph(line, styles["Normal"]))
            content.append(Spacer(1, 6))

    # =========================
    # 💡 PERSONA FOOTER
    # =========================
    content.append(Spacer(1, 20))

    if persona == "CEO":
        footer = (
            "This report highlights key business insights to support strategic decision-making. "
            "Focus on leveraging strengths and addressing risks effectively."
        )
    elif persona == "Marketing":
        footer = (
            "This report emphasizes customer behavior and growth opportunities. "
            "Use these insights to optimize campaigns and maximize engagement."
        )
    else:
        footer = (
            "This report provides a detailed analytical view of the dataset. "
            "Use these findings to guide data-driven decisions."
        )

    content.append(Paragraph(footer, styles["Italic"]))

    # =========================
    # 🏗️ BUILD PDF
    # =========================
    doc.build(content)

    buffer.seek(0)
    return buffer