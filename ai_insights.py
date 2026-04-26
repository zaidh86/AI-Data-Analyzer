import pandas as pd


def get_insights(df):
    insights = []

    # 📊 Basic Overview
    insights.append("📊 DATASET OVERVIEW")
    insights.append(f"- Rows: {df.shape[0]}")
    insights.append(f"- Columns: {df.shape[1]}")

    # 📂 Column Types
    insights.append("\n📂 COLUMN TYPES")
    for col in df.columns:
        insights.append(f"- {col}: {df[col].dtype}")

    # ⚠️ Missing Values
    missing = df.isna().sum()
    if missing.sum() > 0:
        insights.append("\n⚠️ MISSING VALUES DETECTED")
        for col, val in missing.items():
            if val > 0:
                insights.append(f"- {col}: {val} missing ({(val/len(df)*100):.2f}%)")
    else:
        insights.append("\n✅ No missing values found")

    # 📈 Numeric Analysis
    numeric = df.select_dtypes(include='number')

    if not numeric.empty:
        insights.append("\n📈 NUMERIC INSIGHTS")

        # Mean values
        means = numeric.mean()
        for col in means.index:
            insights.append(f"- Avg {col}: {means[col]:.2f}")

        # Highest & lowest values
        for col in numeric.columns:
            insights.append(f"\n🔹 {col}")
            insights.append(f"  - Max: {df[col].max()}")
            insights.append(f"  - Min: {df[col].min()}")

    # 🔥 Correlation Analysis
    if len(numeric.columns) > 1:
        corr = numeric.corr()

        insights.append("\n🔥 STRONG CORRELATIONS (> 0.7)")

        found = False
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                value = corr.iloc[i, j]
                if abs(value) > 0.7:
                    insights.append(
                        f"- {corr.columns[i]} & {corr.columns[j]}: {value:.2f}"
                    )
                    found = True

        if not found:
            insights.append("- No strong correlations found")

    # 🚨 Simple Business Insights
    insights.append("\n💡 BUSINESS INSIGHTS")

    if not numeric.empty:
        top_col = numeric.mean().idxmax()
        insights.append(f"- {top_col} has the highest average values")

    insights.append("- Consider focusing on high-performing metrics")
    insights.append("- Check columns with missing data for data quality issues")

    return "\n".join(insights)