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
USE_GROQ = True


# =========================
# 🤖 GROQ AI FUNCTION
# =========================
def get_groq_insights(df, persona="Analyst"):
    try:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            return "⚠️ Groq API key not found. Please check your .env file."

        client = Groq(api_key=api_key)

        # 🎭 PERSONA STYLE
        if persona == "CEO":
            role_prompt = """
Focus on high-level strategy, business impact, risks, and opportunities.
Keep it concise and decision-oriented.
"""

        elif persona == "Marketing":
            role_prompt = """
Focus on customer behavior, product trends, engagement, and growth opportunities.
Highlight what drives sales and demand.
"""

        else:
            role_prompt = """
Focus on detailed analysis, trends, patterns, and statistical insights.
Be precise and analytical.
"""

        prompt = f"""
You are a professional data analyst.

{role_prompt}

Analyze this dataset and provide a clear, human-like, and insightful report.

DATA OVERVIEW:
Rows: {df.shape[0]}
Columns: {df.shape[1]}

COLUMNS:
{list(df.columns)}

DATA TYPES:
{df.dtypes.to_string()}

MISSING VALUES:
{df.isna().sum().to_string()}

STATISTICS:
{df.describe(include='all').to_string()}

Your response must include:
1. Executive Summary
2. Key Insights
3. Data Quality Issues
4. Trends & Patterns
5. Business Recommendations

Style:
- Natural tone
- Clear reasoning
- Actionable advice
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Groq AI Error: {e}"


# =========================
# 🤖 MAIN INSIGHTS ENGINE
# =========================
def get_insights(df, persona="Analyst"):
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

    # 🎯 CONFIDENCE
    missing_percent = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100

    if missing_percent < 5:
        confidence = "High"
    elif missing_percent < 20:
        confidence = "Moderate"
    else:
        confidence = "Low"

    # 📌 SUMMARY
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

    # 🧠 INTRO
    sections.append("🤖 AI DATA ANALYSIS REPORT")
    sections.append("=" * 60)
    sections.append("\n" + random.choice(tone_intro))

    # ⚠️ DATA QUALITY
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

    # 📈 PERFORMANCE
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

    # 💡 RECOMMENDATIONS
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

    # 📦 COMBINE LOCAL
    final_text = "\n".join(summary + [""] + sections)

    # 🤖 GROQ
    if USE_GROQ:
        ai_output = get_groq_insights(df, persona)
        final_text += "\n\n🤖 AI ENHANCED INSIGHTS\n" + "-" * 50 + "\n" + ai_output

    return final_text


# =========================
# 💬 CHAT MODE
# =========================
def chat_with_data(df, query, persona="Analyst"):
    query = query.lower()
    numeric = df.select_dtypes(include='number')

    try:
        if os.getenv("GROQ_API_KEY"):
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            # 🎭 PERSONA
            if persona == "CEO":
                role_prompt = "You are a CEO giving strategic, high-level advice."

            elif persona == "Marketing":
                role_prompt = "You are a marketing expert focusing on customer trends and growth."

            else:
                role_prompt = "You are a data analyst focusing on insights and patterns."

            prompt = f"""
{role_prompt}

Dataset summary:
{df.describe().to_string()}

User question:
{query}

Respond:
- Clearly
- Naturally
- With reasoning
- Include actionable advice
"""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6
            )

            return response.choices[0].message.content

    except:
        pass

    # 🧠 FALLBACK
    if "best" in query:
        return f"Best metric: {numeric.mean().idxmax()}"

    elif "worst" in query:
        return f"Worst metric: {numeric.mean().idxmin()}"

    elif "missing" in query:
        return df.isna().sum().to_string()

    elif "rows" in query:
        return f"{df.shape[0]} rows and {df.shape[1]} columns"

    return "Ask about best/worst/missing/correlation."