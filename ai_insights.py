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
            return "⚠️ Groq API key not found."

        client = Groq(api_key=api_key)

        if persona == "CEO":
            role_prompt = "Focus on strategy, risks, and business decisions."
        elif persona == "Marketing":
            role_prompt = "Focus on customer behavior, demand, and growth opportunities."
        else:
            role_prompt = "Focus on detailed analysis, trends, and patterns."

        prompt = f"""
You are an expert data analyst.

{role_prompt}

Dataset:
{df.describe(include='all').to_string()}

Give:
- Clear insights
- Key patterns
- Business advice
- Keep it concise but powerful
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
# 🧠 SMART LOCAL ENGINE
# =========================
def get_local_insights(df, persona="Analyst"):
    insights = []

    numeric = df.select_dtypes(include=np.number)

    # 🎭 Persona intro
    if persona == "CEO":
        insights.append("🧑‍💼 EXECUTIVE OVERVIEW")
        insights.append("This dataset highlights key performance areas and potential risks.")

    elif persona == "Marketing":
        insights.append("📣 MARKETING INSIGHTS")
        insights.append("The data reveals trends in demand, customer behavior, and product performance.")

    else:
        insights.append("📊 DATA ANALYSIS SUMMARY")
        insights.append("The dataset provides useful insights into trends and performance.")

    # 📊 Basic overview
    insights.append(f"\nDataset size: {df.shape[0]} rows × {df.shape[1]} columns")

    # ⚠️ Missing data
    missing = df.isna().sum().sum()
    if missing == 0:
        insights.append("No missing values detected — data quality is strong.")
    else:
        insights.append(f"{missing} missing values detected — cleaning recommended.")

    # 📈 Performance logic (smarter)
    if not numeric.empty:
        best = numeric.mean().idxmax()
        worst = numeric.mean().idxmin()

        insights.append(f"\nTop performing metric: {best}")
        insights.append(f"Underperforming metric: {worst}")

        # variation insight
        variability = numeric.std().mean()

        if variability > 1000:
            insights.append("There is high variability in the data — performance is inconsistent.")
        else:
            insights.append("Data shows relatively stable performance across metrics.")

    # 🔥 correlation insight
    if len(numeric.columns) > 1:
        corr = numeric.corr().abs()
        strong_pairs = []

        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if corr.iloc[i, j] > 0.7:
                    strong_pairs.append((corr.columns[i], corr.columns[j]))

        if strong_pairs:
            insights.append("\nStrong relationships detected between variables:")
            for a, b in strong_pairs:
                insights.append(f"- {a} & {b}")

    # 💡 advice (persona-based)
    insights.append("\n💡 Recommendations:")

    if persona == "CEO":
        insights.append("- Focus on improving weaker metrics.")
        insights.append("- Reduce performance variability.")
        insights.append("- Scale high-performing areas.")

    elif persona == "Marketing":
        insights.append("- Focus on high-demand products.")
        insights.append("- Improve customer satisfaction.")
        insights.append("- Target top-performing segments.")

    else:
        insights.append("- Investigate anomalies.")
        insights.append("- Optimize strong variables.")
        insights.append("- Clean missing data if present.")

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
                role_prompt = "Answer like a CEO focusing on strategy."
            elif persona == "Marketing":
                role_prompt = "Answer like a marketing expert."
            else:
                role_prompt = "Answer like a data analyst."

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

    return "Ask about best/worst/missing/correlation."