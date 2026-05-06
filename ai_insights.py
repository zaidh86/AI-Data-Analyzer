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
def chat_with_data(df, query, persona="Analyst", history=None):
    import random
    import os
    from groq import Groq

    query_clean = query.lower().strip()
    numeric = df.select_dtypes(include='number')

    # =========================
    # 🧠 INTENT DETECTION
    # =========================
    greetings = ["hi", "hello", "hey", "yo", "sup"]
    farewells = ["bye", "goodbye", "see you"]
    thanks = ["thanks", "thank you"]

    advice_words = ["should", "recommend", "advice", "improve", "suggest"]
    explain_words = ["why", "explain", "reason"]
    trend_words = ["trend", "growth", "increase", "decrease"]
    risk_words = ["risk", "problem", "issue", "danger"]

    # =========================
    # 👋 GREETINGS
    # =========================
    if any(word in query_clean for word in greetings):
        if persona == "CEO":
            return "Hello. Ready to review strategic insights and performance?"
        elif persona == "Marketing":
            return "Hey! Ready to explore customer trends and growth opportunities? 🚀"
        else:
            return "Hey! 👋 What would you like to explore in your dataset?"

    if any(word in query_clean for word in farewells):
        return "See you later 👋 Feel free to return anytime for more insights."

    if any(word in query_clean for word in thanks):
        return "You're welcome 😄 Let me know if you'd like deeper analysis."

    # =========================
    # 🎯 DETECT INTENT
    # =========================
    intent = "general"

    if any(word in query_clean for word in advice_words):
        intent = "advice"

    elif any(word in query_clean for word in explain_words):
        intent = "explain"

    elif any(word in query_clean for word in trend_words):
        intent = "trend"

    elif any(word in query_clean for word in risk_words):
        intent = "risk"

    # =========================
    # 🧠 MEMORY
    # =========================
    conversation = ""

    if history:
        for msg in history[-5:]:
            role = msg["role"]
            content = msg["content"]
            conversation += f"{role}: {content}\n"

    # =========================
    # 🎭 PERSONA STYLE
    # =========================
    if persona == "CEO":
        role_prompt = (
            "Respond like a strategic business executive focused on "
            "growth, performance, and decisions."
        )

    elif persona == "Marketing":
        role_prompt = (
            "Respond like a marketing strategist focused on "
            "customer behavior, engagement, and growth."
        )

    else:
        role_prompt = (
            "Respond like a professional data analyst focused on "
            "insights and patterns."
        )

    # =========================
    # 🎯 INTENT PROMPTS
    # =========================
    if intent == "advice":
        intent_prompt = (
            "Provide actionable recommendations and practical next steps."
        )

    elif intent == "explain":
        intent_prompt = (
            "Explain clearly in a simple and human way."
        )

    elif intent == "trend":
        intent_prompt = (
            "Focus on patterns, changes over time, and trends."
        )

    elif intent == "risk":
        intent_prompt = (
            "Focus on risks, weaknesses, anomalies, and concerns."
        )

    else:
        intent_prompt = (
            "Answer naturally with useful insights."
        )

    # =========================
    # 🤖 AI RESPONSE
    # =========================
    try:
        if os.getenv("GROQ_API_KEY"):
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            prompt = f"""
{role_prompt}

{intent_prompt}

Previous conversation:
{conversation}

Dataset summary:
{df.describe(include='all').to_string()}

User question:
{query}

Instructions:
- Be conversational
- Sound natural and intelligent
- Keep responses concise but insightful
- Include reasoning
"""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            ai_response = response.choices[0].message.content

            # =========================
            # 🔥 FOLLOW-UP QUESTIONS
            # =========================
            followups = [
                "Would you like a deeper breakdown of this insight?",
                "Want me to identify patterns or anomalies?",
                "Should I analyze trends over time?",
                "Do you want recommendations based on this data?",
                "Want me to explain this in simpler terms?",
                "Should I highlight potential risks or opportunities?",
                "Want a business-focused interpretation of these results?",
                "Would you like a quick executive summary?"
            ]

            ai_response += "\n\n👉 " + random.choice(followups)

            return ai_response

    except Exception as e:
        return f"⚠️ AI Error: {e}"

    # =========================
    # 🔙 FALLBACK
    # =========================
    if "best" in query_clean:
        return f"The strongest metric appears to be '{numeric.mean().idxmax()}'."

    elif "worst" in query_clean:
        return f"The weakest metric appears to be '{numeric.mean().idxmin()}'."

    elif "missing" in query_clean:
        return df.isna().sum().to_string()

    return "I can help analyze trends, risks, recommendations, and performance insights."