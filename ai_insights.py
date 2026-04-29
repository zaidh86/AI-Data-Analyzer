import pandas as pd
import numpy as np
import random
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()


# =========================
# ⚙️ SETTINGS
# =========================
USE_GROQ = True   # 🔁 set False if you want only local AI


# =========================
# 🤖 GROQ AI FUNCTION
# =========================
def get_groq_insights(df):
    print("API KEY:", os.getenv("GROQ_API_KEY"))
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
            model="llama-3.1-70b-versatile",
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


def chat_with_data(df, query):
    query = query.lower()
    numeric = df.select_dtypes(include='number')

    # =========================
    # 🤖 TRY REAL AI (GROQ)
    # =========================
    try:
        if os.getenv("GROQ_API_KEY"):
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            prompt = f"""
You are a smart data analyst assistant.

Dataset summary:
{df.describe().to_string()}

User question:
{query}

Respond in a:
- Human-like tone
- Clear explanation
- Include advice if possible
- Keep it concise but insightful
"""

            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6
            )

            return response.choices[0].message.content

    except:
        pass  # fallback to local logic

    # =========================
    # 🧠 SMART LOCAL RESPONSES
    # =========================

    # 📈 BEST
    if "best" in query or "highest" in query:
        if not numeric.empty:
            best = numeric.mean().idxmax()
            return (
                f"📈 Looking at the overall performance, **'{best}' stands out as the strongest metric**.\n\n"
                f"This suggests that this area is performing consistently well. "
                f"You might want to focus on maintaining or scaling this success further."
            )

    # 📉 WORST
    elif "worst" in query or "lowest" in query:
        if not numeric.empty:
            worst = numeric.mean().idxmin()
            return (
                f"📉 The data indicates that **'{worst}' is underperforming compared to other metrics**.\n\n"
                f"This could be a potential area of concern. It may be worth investigating the reasons behind this trend."
            )

    # ⚠️ MISSING
    elif "missing" in query:
        missing = df.isna().sum()
        total = missing.sum()

        if total == 0:
            return "✅ Good news — your dataset looks clean with no missing values detected."

        return (
            f"⚠️ There are **{total} missing values** in your dataset.\n\n"
            f"Here’s a breakdown:\n{missing.to_string()}\n\n"
            f"👉 It's generally a good idea to handle missing data before making important decisions."
        )

    # 📊 SIZE
    elif "rows" in query or "size" in query:
        return (
            f"📊 Your dataset contains **{df.shape[0]} rows and {df.shape[1]} columns**.\n\n"
            f"This gives a decent amount of data to analyze, though more data usually improves reliability."
        )

    # 🔥 CORRELATION
    elif "correlation" in query:
        if len(numeric.columns) > 1:
            corr = numeric.corr()
            return (
                "🔥 Here’s the correlation matrix between your numeric variables:\n\n"
                f"{corr.to_string()}\n\n"
                "👉 Strong relationships (close to 1 or -1) can reveal important patterns."
            )
        else:
            return "Not enough numeric data to analyze correlations."

    # 🤖 DEFAULT SMART RESPONSE
    return (
        "🤖 I didn’t fully catch that, but I can help!\n\n"
        "Try asking things like:\n"
        "• Which metric is performing best?\n"
        "• Are there any missing values?\n"
        "• Show correlations\n"
        "• What’s the dataset size?\n\n"
        "👉 You can also ask more specific questions!"
    )