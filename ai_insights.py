import pandas as pd
import numpy as np
import random
from groq import Groq
import os


# =========================
# ⚙️ SETTINGS
# =========================
USE_GROQ = True   # 🔁 set False if you want only local AI


# =========================
# 🤖 GROQ AI FUNCTION
# =========================
def get_groq_insights(df):
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        prompt = f"""
You are a professional data analyst.

Analyze this dataset and provide a clear, human-like, and insightful report.

DATA SUMMARY:
{df.describe().to_string()}

Your response must include:
1. Executive Summary
2. Key Insights
3. Data Quality Issues
4. Trends & Patterns
5. Business Recommendations

Style:
- Natural, human-like tone
- Explain reasoning
- Give actionable advice
- Avoid robotic language
"""

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Groq AI Error: {e}"


# =========================
# 🤖 MAIN INSIGHTS ENGINE
# =========================
def get_insights(df):
    sections = []

    tone_intro = [
        "From an analytical perspective, the dataset reveals several meaningful patterns.",
        "Taking a structured approach, the dataset highlights both strengths and potential concerns.",
        "A deeper evaluation of the dataset uncovers insights that can guide strategic decisions."
    ]

    connectors = [
        "Additionally,",
        "Furthermore,",
        "From a broader perspective,",
        "This suggests that",
        "As a result,"
    ]

    # =========================
    # 🎯 CONFIDENCE
    # =========================
    missing_percent = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100

    if missing_percent < 5:
        confidence = "High"
    elif missing_percent < 20:
        confidence = "Moderate"
    else:
        confidence = "Low"

    # =========================
    # 📌 EXECUTIVE SUMMARY
    # =========================
    summary = []
    summary.append("📌 EXECUTIVE SUMMARY")
    summary.append("-" * 50)

    if missing_percent == 0:
        summary.append("The dataset is clean and reliable, with no significant data quality issues detected.")
    elif missing_percent < 10:
        summary.append("The dataset is mostly reliable, though minor data quality issues are present.")
    else:
        summary.append("The dataset contains notable data quality concerns that may affect analysis accuracy.")

    numeric = df.select_dtypes(include=np.number)

    if not numeric.empty:
        best_col = numeric.mean().idxmax()
        summary.append(f"The strongest performing metric appears to be '{best_col}', indicating a key area of strength.")

    summary.append(f"Overall confidence in the analysis is assessed as {confidence}.")

    # =========================
    # 🧠 INTRO
    # =========================
    sections.append("🤖 AI DATA ANALYSIS REPORT")
    sections.append("=" * 60)
    sections.append("\n" + random.choice(tone_intro))

    # =========================
    # ⚠️ DATA QUALITY
    # =========================
    missing = df.isna().sum()

    if missing.sum() > 0:
        sections.append("\n⚠️ Data Quality Observations")

        for col, val in missing.items():
            if val > 0:
                percent = (val / len(df)) * 100

                if percent > 30:
                    sections.append(f"The column '{col}' contains a high proportion of missing values ({percent:.1f}%).")
                elif percent > 10:
                    sections.append(f"The column '{col}' shows a moderate level of missing data ({percent:.1f}%).")
                else:
                    sections.append(f"The column '{col}' contains a small amount of missing data ({percent:.1f}%).")

    else:
        sections.append("\nThe dataset demonstrates strong data integrity with no missing values detected.")

    # =========================
    # 📈 PERFORMANCE ANALYSIS
    # =========================
    if not numeric.empty:
        sections.append("\n📈 Performance Analysis")

        for col in numeric.columns:
            mean = df[col].mean()

            if mean > df[col].quantile(0.75):
                sentence = f"The metric '{col}' demonstrates strong performance."
            elif mean > df[col].quantile(0.4):
                sentence = f"The metric '{col}' shows moderate performance."
            else:
                sentence = f"The metric '{col}' appears relatively low."

            sections.append(sentence)
            sections.append(f"{random.choice(connectors)} focusing on this metric could influence results.")

    # =========================
    # 💡 FINAL RECOMMENDATIONS
    # =========================
    sections.append("\n💡 Strategic Recommendations")

    recommendations = [
        "Focus on strengthening high-performing areas.",
        "Address data quality issues.",
        "Investigate underperforming metrics.",
        "Leverage relationships between variables.",
        "Monitor trends continuously."
    ]

    for rec in recommendations:
        sections.append(f"- {rec}")

    sections.append(f"\n🎯 Confidence Level: {confidence}")

    # =========================
    # 📦 COMBINE LOCAL OUTPUT
    # =========================
    local_output = []
    local_output.extend(summary)
    local_output.append("\n")
    local_output.extend(sections)

    final_text = "\n".join(local_output)

    # =========================
    # 🤖 GROQ ENHANCEMENT
    # =========================
    if USE_GROQ:
        ai_output = get_groq_insights(df)

        final_text += "\n\n🤖 AI ENHANCED INSIGHTS\n" + "-" * 50 + "\n" + ai_output

    return final_text


# =========================
# 💬 CHAT MODE
# =========================
def chat_with_data(df, query):
    query = query.lower()
    numeric = df.select_dtypes(include='number')

    if "best" in query:
        return f"Best metric: {numeric.mean().idxmax()}"

    elif "worst" in query:
        return f"Worst metric: {numeric.mean().idxmin()}"

    elif "missing" in query:
        return df.isna().sum().to_string()

    elif "rows" in query:
        return f"{df.shape[0]} rows and {df.shape[1]} columns"

    else:
        return "Try asking about best/worst/missing data."