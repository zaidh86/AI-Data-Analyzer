import pandas as pd
import numpy as np
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
            return "⚠️ Groq API key not found."

        client = Groq(api_key=api_key)

        # 🎭 PERSONA STYLE
        if persona == "CEO":
            role_prompt = "Focus on strategy, risks, business growth, and decisions. Keep it sharp and impactful."
        elif persona == "Marketing":
            role_prompt = "Focus on customer behavior, product demand, engagement, and growth opportunities."
        else:
            role_prompt = "Focus on detailed analysis, trends, correlations, and statistical insights."

        prompt = f"""
You are an expert data analyst.

{role_prompt}

Dataset:
{df.describe(include='all').to_string()}

Respond in a clean, human-like way:
- Keep it insightful but not overly long
- Use natural language (not robotic)
- Include key takeaways and actionable advice
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Groq Error: {e}"


# =========================
# 🧠 LOCAL SMART INSIGHTS
# =========================
def get_local_insights(df, persona="Analyst"):
    insights = []
    numeric = df.select_dtypes(include=np.number)

    # 🎭 Persona intro
    if persona == "CEO":
        insights.append("🧑‍💼 EXECUTIVE OVERVIEW")
        insights.append("At a high level, the dataset highlights where the business is performing well — and where attention is needed.")

    elif persona == "Marketing":
        insights.append("📣 MARKETING INSIGHTS")
        insights.append("This dataset tells a story about customer behavior, product performance, and growth opportunities.")

    else:
        insights.append("📊 ANALYTICAL SUMMARY")
        insights.append("A closer look at the dataset reveals patterns, strengths, and areas worth investigating.")

    # 📊 Overview
    insights.append(f"\nThe dataset contains **{df.shape[0]} rows and {df.shape[1]} columns**, providing a solid base for analysis.")

    missing = df.isna().sum().sum()

    if missing == 0:
        insights.append("The data is clean — no missing values detected, which increases confidence in the insights.")
    else:
        insights.append(f"There are **{missing} missing values**, which may affect reliability and should be addressed.")

    # 📈 Performance
    if not numeric.empty:
        best = numeric.mean().idxmax()
        worst = numeric.mean().idxmin()

        if persona == "CEO":
            insights.append(f"\nOne clear strength is **{best}**, which stands out as a key business driver.")
            insights.append(f"On the other hand, **{worst}** appears to be lagging and may require attention.")

        elif persona == "Marketing":
            insights.append(f"\n**{best}** is performing strongly, suggesting high demand or engagement.")
            insights.append(f"Meanwhile, **{worst}** could indicate weaker customer interest or conversion.")

        else:
            insights.append(f"\nThe metric **{best}** shows the strongest performance.")
            insights.append(f"In contrast, **{worst}** appears to be underperforming.")

        variability = numeric.std().mean()

        if variability > 1000:
            insights.append("Performance varies significantly across the dataset, indicating inconsistency.")
        else:
            insights.append("Performance appears relatively stable with minimal fluctuations.")

    # 🔥 Relationships
    if len(numeric.columns) > 1:
        corr = numeric.corr().abs()
        strong = []

        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if corr.iloc[i, j] > 0.7:
                    strong.append((corr.columns[i], corr.columns[j]))

        if strong:
            insights.append("\nSome variables show strong relationships:")
            for a, b in strong:
                insights.append(f"• {a} ↔ {b}")

    # 💡 Recommendations
    insights.append("\n💡 What this means:")

    if persona == "CEO":
        insights.append("• Double down on high-performing areas — they drive growth.")
        insights.append("• Address weaker areas before they impact overall performance.")
        insights.append("• Reducing variability could improve stability.")
        insights.append("• Use insights to guide strategic decisions.")

    elif persona == "Marketing":
        insights.append("• Focus on high-demand products or segments.")
        insights.append("• Improve weaker engagement areas.")
        insights.append("• Refine campaigns based on trends.")
        insights.append("• Use insights to drive conversions.")

    else:
        insights.append("• Investigate anomalies or unexpected patterns.")
        insights.append("• Optimize strong-performing variables.")
        insights.append("• Clean data where necessary.")
        insights.append("• Explore correlations for deeper insights.")

    return "\n".join(insights)


# =========================
# 🤖 MAIN INSIGHTS FUNCTION
# =========================
def get_insights(df, persona="Analyst"):
    local_output = get_local_insights(df, persona)

    if USE_GROQ:
        ai_output = get_groq_insights(df, persona)

        return (
            local_output
            + "\n\n"
            + "🤖 AI ENHANCED INSIGHTS\n"
            + "-" * 50
            + "\n"
            + ai_output
        )

    return local_output


# =========================
# 💬 CHAT MODE
# =========================
def chat_with_data(df, query, persona="Analyst"):
    query = query.lower()
    numeric = df.select_dtypes(include='number')

    try:
        if os.getenv("GROQ_API_KEY"):
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            if persona == "CEO":
                role_prompt = "Answer like a CEO focusing on strategy and decisions."
            elif persona == "Marketing":
                role_prompt = "Answer like a marketing expert focusing on growth and engagement."
            else:
                role_prompt = "Answer like a data analyst focusing on insights."

            prompt = f"""
{role_prompt}

Dataset:
{df.describe().to_string()}

Question:
{query}

Answer clearly with reasoning and advice.
"""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6
            )

            return response.choices[0].message.content

    except:
        pass

    # fallback
    if "best" in query:
        return f"Best metric: {numeric.mean().idxmax()}"

    elif "worst" in query:
        return f"Worst metric: {numeric.mean().idxmin()}"

    elif "missing" in query:
        return df.isna().sum().to_string()

    elif "rows" in query:
        return f"{df.shape[0]} rows and {df.shape[1]} columns"

    return "Try asking about performance, missing data, or trends."