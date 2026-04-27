import pandas as pd
import numpy as np
import random


def get_insights(df):
    sections = []

    # =========================
    # 🎭 PERSONA STYLE
    # =========================
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
                    sections.append(
                        f"The column '{col}' contains a high proportion of missing values ({percent:.1f}%), which could significantly impact analytical reliability."
                    )
                elif percent > 10:
                    sections.append(
                        f"The column '{col}' shows a moderate level of missing data ({percent:.1f}%), suggesting potential inconsistencies."
                    )
                else:
                    sections.append(
                        f"The column '{col}' contains a small amount of missing data ({percent:.1f}%), which is unlikely to heavily affect outcomes."
                    )

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
                sentence = f"The metric '{col}' demonstrates strong performance, indicating effective underlying factors."
            elif mean > df[col].quantile(0.4):
                sentence = f"The metric '{col}' shows moderate performance, suggesting stability but room for improvement."
            else:
                sentence = f"The metric '{col}' appears relatively low, which may point to inefficiencies or underperformance."

            sections.append(sentence)

            sections.append(
                f"{random.choice(connectors)} focusing on this metric could influence overall results significantly."
            )

            # Outliers
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]

            if len(outliers) > 0:
                sections.append(
                    f"There are {len(outliers)} unusual observations in '{col}', which may represent anomalies or special cases."
                )

    # =========================
    # 🔥 RELATIONSHIPS
    # =========================
    if len(numeric.columns) > 1:
        corr = numeric.corr()
        sections.append("\n🔗 Relationship Analysis")

        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]

                if abs(val) > 0.7:
                    relation = "strong positive" if val > 0 else "strong negative"
                    sections.append(
                        f"There is a {relation} relationship between '{corr.columns[i]}' and '{corr.columns[j]}', indicating a meaningful connection."
                    )

    # =========================
    # 📅 TRENDS
    # =========================
    for col in df.columns:
        try:
            temp = pd.to_datetime(df[col], errors='coerce')

            if temp.notna().sum() > len(df) * 0.5:
                df_sorted = df.copy()
                df_sorted[col] = temp
                df_sorted = df_sorted.sort_values(by=col)

                sections.append("\n📅 Trend Analysis")

                for num_col in numeric.columns:
                    trend = df_sorted[num_col].iloc[-1] - df_sorted[num_col].iloc[0]

                    if trend > 0:
                        sections.append(
                            f"The metric '{num_col}' shows an upward trend over time, suggesting improving performance."
                        )
                    elif trend < 0:
                        sections.append(
                            f"The metric '{num_col}' shows a downward trend, which may require further investigation."
                        )

                break
        except:
            continue

    # =========================
    # 💡 FINAL RECOMMENDATIONS
    # =========================
    sections.append("\n💡 Strategic Recommendations")

    recommendations = [
        "Focus on strengthening high-performing areas to maximize impact.",
        "Address data quality issues to improve reliability.",
        "Investigate underperforming metrics to uncover root causes.",
        "Leverage relationships between variables to optimize outcomes.",
        "Continuously monitor trends to support proactive decision-making."
    ]

    for rec in recommendations:
        sections.append(f"- {rec}")

    # =========================
    # 🧠 FINAL NARRATIVE
    # =========================
    sections.append("\n🧠 Final Analysis")

    sections.append(
        "Overall, the dataset presents a combination of strengths and improvement opportunities. "
        "While certain metrics demonstrate solid performance, others may require targeted attention. "
        "By aligning strategies with these insights, more informed and effective decisions can be made."
    )

    sections.append(f"\n🎯 Confidence Level: {confidence}")

    # =========================
    # 📦 COMBINE EVERYTHING
    # =========================
    final_output = []
    final_output.extend(summary)
    final_output.append("\n")
    final_output.extend(sections)

    return "\n".join(final_output)