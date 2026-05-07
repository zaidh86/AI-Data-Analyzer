import pandas as pd
import numpy as np
import os
import random

from groq import Groq
from dotenv import load_dotenv

from utils import (
    detect_dataset_type,
    calculate_health_score,
    get_confidence_level
)

load_dotenv()


# =========================
# ⚙️ SETTINGS
# =========================
USE_GROQ = True


# =========================
# 🤖 GROQ AI INSIGHTS
# =========================
def get_groq_insights(df, persona="Analyst"):

    try:

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            return "⚠️ Groq API key not found."

        client = Groq(api_key=api_key)

        # =========================
        # 🧠 DATASET INTELLIGENCE
        # =========================
        dataset_type = detect_dataset_type(df)

        health_score = calculate_health_score(df)

        confidence = get_confidence_level(df)

        # =========================
        # 🎭 PERSONA PROMPTS
        # =========================
        if persona == "CEO":

            role_prompt = (
                "Focus on strategy, risks, growth, "
                "opportunities, and executive decisions."
            )

        elif persona == "Marketing":

            role_prompt = (
                "Focus on customer behavior, engagement, "
                "campaigns, growth opportunities, and trends."
            )

        else:

            role_prompt = (
                "Focus on detailed analysis, patterns, "
                "statistics, correlations, and insights."
            )

        # =========================
        # 🤖 AI PROMPT
        # =========================
        prompt = f"""
You are an advanced AI business analyst.

Persona:
{persona}

Dataset Type:
{dataset_type}

Dataset Health Score:
{health_score}/100

Confidence Level:
{confidence}

{role_prompt}

Dataset Summary:
{df.describe(include='all').to_string()}

Instructions:
- Sound natural and intelligent
- Avoid robotic wording
- Explain why insights matter
- Provide actionable recommendations
- Highlight opportunities and risks
- Use structured sections
- Make the response feel premium
- Be conversational but professional
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7
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

    # =========================
    # 🧠 DATASET INTELLIGENCE
    # =========================
    dataset_type = detect_dataset_type(df)

    confidence = get_confidence_level(df)

    health_score = calculate_health_score(df)

    # =========================
    # 🎭 PERSONA INTRO
    # =========================
    if persona == "CEO":

        insights.append("🧑‍💼 EXECUTIVE OVERVIEW")

        insights.append(
            "This analysis highlights strategic performance, "
            "growth patterns, operational risks, and opportunities."
        )

    elif persona == "Marketing":

        insights.append("📣 MARKETING INSIGHTS")

        insights.append(
            "This dataset reveals customer behavior, "
            "engagement trends, and growth opportunities."
        )

    else:

        insights.append("📊 ANALYTICAL SUMMARY")

        insights.append(
            "This analysis explores patterns, relationships, "
            "performance trends, and data quality."
        )

    # =========================
    # 🧠 DATASET CONTEXT
    # =========================
    insights.append(
        f"\n📂 Detected Dataset Type: **{dataset_type}**"
    )

    insights.append(
        f"🎯 Confidence Level: **{confidence}**"
    )

    insights.append(
        f"🧹 Dataset Health Score: **{health_score}/100**"
    )

    # =========================
    # 📊 OVERVIEW
    # =========================
    insights.append(
        f"\nThe dataset contains **{df.shape[0]} rows** "
        f"and **{df.shape[1]} columns**."
    )

    missing = df.isna().sum().sum()

    if missing == 0:

        insights.append(
            "The dataset appears clean with no missing values detected."
        )

    else:

        insights.append(
            f"The dataset contains **{missing} missing values**, "
            f"which may affect analytical reliability."
        )

    # =========================
    # 📈 PERFORMANCE ANALYSIS
    # =========================
    if not numeric.empty:

        strongest = numeric.mean().idxmax()

        weakest = numeric.mean().idxmin()

        variability = numeric.std().mean()

        insights.append("\n📈 Performance Analysis")

        if persona == "CEO":

            insights.append(
                f"**{strongest}** currently appears to be the "
                f"strongest business driver."
            )

            insights.append(
                f"**{weakest}** may require strategic attention "
                f"to avoid long-term impact."
            )

        elif persona == "Marketing":

            insights.append(
                f"**{strongest}** demonstrates strong customer "
                f"or market performance."
            )

            insights.append(
                f"**{weakest}** may indicate weaker engagement "
                f"or lower conversion effectiveness."
            )

        else:

            insights.append(
                f"The metric **{strongest}** shows the "
                f"strongest overall performance."
            )

            insights.append(
                f"Meanwhile, **{weakest}** appears to "
                f"underperform compared to other variables."
            )

        if variability > 1000:

            insights.append(
                "Performance variability appears high, "
                "suggesting inconsistencies across observations."
            )

        else:

            insights.append(
                "Performance remains relatively stable "
                "throughout the dataset."
            )

    # =========================
    # 🔥 CORRELATIONS
    # =========================
    if len(numeric.columns) > 1:

        corr = numeric.corr().abs()

        strong_relationships = []

        for i in range(len(corr.columns)):

            for j in range(i + 1, len(corr.columns)):

                if corr.iloc[i, j] > 0.7:

                    strong_relationships.append(
                        (
                            corr.columns[i],
                            corr.columns[j]
                        )
                    )

        if strong_relationships:

            insights.append("\n🔥 Relationship Insights")

            for a, b in strong_relationships:

                insights.append(
                    f"• Strong relationship detected between "
                    f"**{a}** and **{b}**."
                )

    # =========================
    # 💡 ACTIONABLE INSIGHTS
    # =========================
    insights.append("\n💡 Recommended Actions")

    if persona == "CEO":

        insights.append(
            "• Strengthen high-performing areas to maximize growth."
        )

        insights.append(
            "• Address operational weaknesses before they scale."
        )

        insights.append(
            "• Use data-driven insights for strategic decisions."
        )

        insights.append(
            "• Monitor performance variability closely."
        )

    elif persona == "Marketing":

        insights.append(
            "• Focus campaigns around high-performing segments."
        )

        insights.append(
            "• Improve weaker engagement areas."
        )

        insights.append(
            "• Use trends to optimize marketing strategy."
        )

        insights.append(
            "• Analyze customer behavior continuously."
        )

    else:

        insights.append(
            "• Investigate anomalies and outliers further."
        )

        insights.append(
            "• Improve overall data quality where needed."
        )

        insights.append(
            "• Explore deeper statistical relationships."
        )

        insights.append(
            "• Monitor key metrics consistently."
        )

    return "\n".join(insights)


# =========================
# 🤖 MAIN INSIGHTS FUNCTION
# =========================
def get_insights(df, persona="Analyst"):

    local_output = get_local_insights(
        df,
        persona
    )

    if USE_GROQ:

        ai_output = get_groq_insights(
            df,
            persona
        )

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
# 💬 CHATBOT MODE
# =========================
def chat_with_data(
    df,
    query,
    persona="Analyst",
    history=None
):

    query_clean = query.lower().strip()

    numeric = df.select_dtypes(include='number')

    # =========================
    # 🧠 DATASET CONTEXT
    # =========================
    dataset_type = detect_dataset_type(df)

    confidence = get_confidence_level(df)

    # =========================
    # 🧠 INTENT DETECTION
    # =========================
    greetings = [
        "hi",
        "hello",
        "hey",
        "yo",
        "sup"
    ]

    farewells = [
        "bye",
        "goodbye",
        "see you"
    ]

    thanks = [
        "thanks",
        "thank you"
    ]

    advice_words = [
        "should",
        "recommend",
        "advice",
        "improve",
        "suggest"
    ]

    explain_words = [
        "why",
        "explain",
        "reason"
    ]

    trend_words = [
        "trend",
        "growth",
        "increase",
        "decrease"
    ]

    risk_words = [
        "risk",
        "problem",
        "issue",
        "danger"
    ]

    # =========================
    # 👋 CONVERSATIONAL LOGIC
    # =========================
    if any(word in query_clean for word in greetings):

        if persona == "CEO":

            return (
                "Hello. Ready to review strategic insights "
                "and performance indicators?"
            )

        elif persona == "Marketing":

            return (
                "Hey! Ready to explore customer behavior "
                "and growth opportunities? 🚀"
            )

        else:

            return (
                "Hey! 👋 What would you like to "
                "explore in your dataset?"
            )

    if any(word in query_clean for word in farewells):

        return (
            "See you later 👋 "
            "Feel free to return anytime for more insights."
        )

    if any(word in query_clean for word in thanks):

        return (
            "You're welcome 😄 "
            "Let me know if you'd like deeper analysis."
        )

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
    # 🧠 CHAT MEMORY
    # =========================
    conversation = ""

    if history:

        for msg in history[-5:]:

            role = msg["role"]

            content = msg["content"]

            conversation += (
                f"{role}: {content}\n"
            )

    # =========================
    # 🎭 PERSONA STYLE
    # =========================
    if persona == "CEO":

        role_prompt = (
            "Respond like a strategic executive focused "
            "on growth, risks, performance, and decisions."
        )

    elif persona == "Marketing":

        role_prompt = (
            "Respond like a marketing strategist focused "
            "on engagement, customers, and growth."
        )

    else:

        role_prompt = (
            "Respond like a professional data analyst "
            "focused on insights and reasoning."
        )

    # =========================
    # 🎯 INTENT PROMPTS
    # =========================
    if intent == "advice":

        intent_prompt = (
            "Provide actionable recommendations "
            "and practical next steps."
        )

    elif intent == "explain":

        intent_prompt = (
            "Explain clearly in a simple and human way."
        )

    elif intent == "trend":

        intent_prompt = (
            "Focus on patterns, changes over time, "
            "and growth trends."
        )

    elif intent == "risk":

        intent_prompt = (
            "Focus on weaknesses, risks, anomalies, "
            "and concerns."
        )

    else:

        intent_prompt = (
            "Answer naturally with useful insights."
        )

    # =========================
    # 🤖 AI CHAT RESPONSE
    # =========================
    try:

        api_key = os.getenv("GROQ_API_KEY")

        if api_key:

            client = Groq(api_key=api_key)

            prompt = f"""
You are an advanced AI data assistant.

Persona:
{persona}

Dataset Type:
{dataset_type}

Confidence:
{confidence}

{role_prompt}

{intent_prompt}

Previous Conversation:
{conversation}

Dataset Summary:
{df.describe(include='all').to_string()}

User Question:
{query}

Instructions:
- Sound intelligent and natural
- Avoid robotic responses
- Explain WHY insights matter
- Be conversational
- Keep answers insightful
- Make the response feel premium
"""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7
            )

            ai_response = (
                response.choices[0]
                .message.content
            )

            # =========================
            # 🔥 FOLLOW-UP QUESTIONS
            # =========================
            followups = [
                "Would you like a deeper breakdown?",
                "Want me to identify risks or opportunities?",
                "Should I analyze trends further?",
                "Do you want actionable recommendations?",
                "Would you like a quick executive summary?",
                "Want a more beginner-friendly explanation?"
            ]

            ai_response += (
                "\n\n👉 "
                + random.choice(followups)
            )

            return ai_response

    except Exception as e:

        return f"⚠️ AI Error: {e}"

    # =========================
    # 🔙 FALLBACK RESPONSES
    # =========================
    if "best" in query_clean:

        return (
            f"The strongest metric currently appears "
            f"to be **{numeric.mean().idxmax()}**."
        )

    elif "worst" in query_clean:

        return (
            f"The weakest metric currently appears "
            f"to be **{numeric.mean().idxmin()}**."
        )

    elif "missing" in query_clean:

        return df.isna().sum().to_string()

    return (
        "I can help analyze trends, risks, "
        "performance, recommendations, and insights."
    )