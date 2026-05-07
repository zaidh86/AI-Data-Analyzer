import pandas as pd
import numpy as np


# =========================
# 📊 DATASET HEALTH SCORE
# =========================
def calculate_health_score(df):

    score = 100

    # Missing values
    missing_percent = (
        df.isna().sum().sum()
        / (df.shape[0] * df.shape[1])
    ) * 100

    if missing_percent > 20:
        score -= 30

    elif missing_percent > 5:
        score -= 15

    # Duplicate rows
    duplicates = df.duplicated().sum()

    if duplicates > 0:
        score -= min(20, duplicates)

    # Small dataset penalty
    if df.shape[0] < 50:
        score -= 10

    return max(score, 0)


# =========================
# 🎯 CONFIDENCE LEVEL
# =========================
def get_confidence_level(df):

    missing = (
        df.isna().sum().sum()
        / (df.shape[0] * df.shape[1])
    ) * 100

    if missing == 0 and df.shape[0] > 100:
        return "High"

    elif missing < 15:
        return "Moderate"

    return "Low"


# =========================
# 🔍 DATASET TYPE DETECTION
# =========================
def detect_dataset_type(df):

    columns = " ".join(df.columns).lower()

    if any(word in columns for word in ["revenue", "sales", "profit"]):
        return "Sales"

    elif any(word in columns for word in ["student", "marks", "grade"]):
        return "Education"

    elif any(word in columns for word in ["employee", "salary", "department"]):
        return "HR"

    elif any(word in columns for word in ["patient", "hospital", "medical"]):
        return "Healthcare"

    elif any(word in columns for word in ["campaign", "click", "engagement"]):
        return "Marketing"

    return "General"


# =========================
# 🧑‍💼 EXECUTIVE SNAPSHOT
# =========================
def generate_executive_summary(df):

    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:
        return {
            "Strongest Metric": "N/A",
            "Weakest Metric": "N/A",
            "Growth Trend": "Unknown",
            "Risk Level": "Moderate"
        }

    strongest = numeric.mean().idxmax()
    weakest = numeric.mean().idxmin()

    variability = numeric.std().mean()

    if variability > 1000:
        trend = "Volatile"
        risk = "High"

    else:
        trend = "Stable"
        risk = "Moderate"

    return {
        "Strongest Metric": strongest,
        "Weakest Metric": weakest,
        "Growth Trend": trend,
        "Risk Level": risk
    }


# =========================
# 🔥 PRIORITY INSIGHTS
# =========================
def generate_priority_insights(df):

    insights = []

    numeric = df.select_dtypes(include=np.number)

    if not numeric.empty:

        strongest = numeric.mean().idxmax()

        insights.append(
            f"🔥 {strongest} is currently the strongest performing metric."
        )

        missing = df.isna().sum().sum()

        if missing > 0:
            insights.append(
                "⚠️ Missing values detected which may affect reliability."
            )

        if len(numeric.columns) > 1:

            corr = numeric.corr().abs()

            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):

                    if corr.iloc[i, j] > 0.7:

                        insights.append(
                            f"📈 Strong relationship between "
                            f"{corr.columns[i]} and {corr.columns[j]}."
                        )

                        return insights[:3]

    return insights[:3]


# =========================
# 💡 AI QUESTION SUGGESTIONS
# =========================
def generate_suggested_questions(dataset_type):

    base_questions = [
        "What trends do you see?",
        "What are the biggest risks?",
        "What should I improve?"
    ]

    if dataset_type == "Sales":

        base_questions.extend([
            "Which product performs best?",
            "What drives revenue growth?"
        ])

    elif dataset_type == "Marketing":

        base_questions.extend([
            "Which campaigns perform best?",
            "How can engagement improve?"
        ])

    return base_questions