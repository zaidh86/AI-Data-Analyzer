import pandas as pd
import numpy as np


def get_insights(df):
    insights = []

    insights.append("🤖 AI DATA ANALYSIS REPORT")
    insights.append("=" * 50)

    # 📊 Overview
    insights.append("\n📊 DATASET OVERVIEW")
    insights.append(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    # 📂 Data Types
    insights.append("\n📂 DATA STRUCTURE")
    for col in df.columns:
        insights.append(f"- '{col}' is of type {df[col].dtype}")

    # ⚠️ Missing Values
    missing = df.isna().sum()
    total_missing = missing.sum()

    if total_missing > 0:
        insights.append("\n⚠️ DATA QUALITY ISSUES DETECTED")
        for col, val in missing.items():
            if val > 0:
                percent = (val / len(df)) * 100
                insights.append(f"- {col} has {val} missing values ({percent:.2f}%)")
        insights.append("👉 Recommendation: Handle missing values before making critical decisions.")
    else:
        insights.append("\n✅ Data appears clean with no missing values.")

    # 📈 Numeric Analysis
    numeric = df.select_dtypes(include=np.number)

    if not numeric.empty:
        insights.append("\n📈 NUMERIC ANALYSIS")

        for col in numeric.columns:
            mean = df[col].mean()
            std = df[col].std()
            insights.append(f"\n🔹 {col}")
            insights.append(f"  - Average: {mean:.2f}")
            insights.append(f"  - Standard Deviation: {std:.2f}")
            insights.append(f"  - Range: {df[col].min()} → {df[col].max()}")

            # 🚨 Outlier Detection
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]

            if not outliers.empty:
                insights.append(f"  - ⚠️ {len(outliers)} potential outliers detected")
            else:
                insights.append("  - No significant outliers detected")

    # 🔥 Correlation Analysis
    if len(numeric.columns) > 1:
        corr = numeric.corr()
        insights.append("\n🔥 CORRELATION INSIGHTS")

        found = False
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                value = corr.iloc[i, j]
                if abs(value) > 0.6:
                    direction = "positive" if value > 0 else "negative"
                    insights.append(
                        f"- Strong {direction} relationship between {corr.columns[i]} and {corr.columns[j]} ({value:.2f})"
                    )
                    found = True

        if not found:
            insights.append("- No strong relationships found between variables")

    # 📅 Time-based Trends
    date_cols = df.select_dtypes(include=['datetime64', 'object'])

    for col in date_cols.columns:
        try:
            temp = pd.to_datetime(df[col], errors='coerce')
            if temp.notna().sum() > 0:
                df_sorted = df.copy()
                df_sorted[col] = temp
                df_sorted = df_sorted.sort_values(by=col)

                insights.append(f"\n📅 TIME-BASED ANALYSIS ({col})")

                if not numeric.empty:
                    for num_col in numeric.columns:
                        trend = df_sorted[num_col].iloc[-1] - df_sorted[num_col].iloc[0]
                        if trend > 0:
                            insights.append(f"- {num_col} shows an upward trend over time 📈")
                        elif trend < 0:
                            insights.append(f"- {num_col} shows a downward trend 📉")
                        else:
                            insights.append(f"- {num_col} remains relatively stable")
                break
        except:
            continue

    # 🧩 Categorical Analysis
    categorical = df.select_dtypes(include='object')

    if not categorical.empty:
        insights.append("\n🧩 CATEGORICAL INSIGHTS")

        for col in categorical.columns:
            top_values = df[col].value_counts().head(3)
            insights.append(f"\n🔹 {col}")
            for idx, val in top_values.items():
                insights.append(f"- {idx}: {val} occurrences")

    # 💡 Business Insights
    insights.append("\n💡 BUSINESS INSIGHTS & RECOMMENDATIONS")

    if not numeric.empty:
        best_col = numeric.mean().idxmax()
        insights.append(f"- '{best_col}' is performing the strongest on average")

    insights.append("- Focus on high-performing segments to maximize outcomes")
    insights.append("- Investigate anomalies or sudden drops in performance")
    insights.append("- Improve data collection to reduce missing values")
    insights.append("- Use trends to guide future decision-making")

    insights.append("\n🚀 Overall: The dataset provides actionable insights that can drive smarter decisions if leveraged correctly.")

    return "\n".join(insights)