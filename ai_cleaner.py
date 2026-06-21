from data_cleaner import (
    detect_missing_values,
    detect_duplicates,
    detect_dtype_issues,
    detect_outliers,
    calculate_quality_score
)


# =========================
# 🤖 CLEANING SUMMARY
# =========================
def generate_cleaning_summary(df):

    missing = detect_missing_values(df)

    duplicates = detect_duplicates(df)

    dtype_issues = detect_dtype_issues(df)

    outliers = detect_outliers(df)

    quality = calculate_quality_score(df)

    summary = []

    summary.append(
        f"🎯 Dataset Quality Score: {quality}/100"
    )

    total_missing = missing["Missing Values"].sum()

    summary.append(
        f"🧩 Missing Values Found: {total_missing}"
    )

    summary.append(
        f"📄 Duplicate Rows Found: {duplicates}"
    )

    summary.append(
        f"🔍 Data Type Issues Found: {len(dtype_issues)}"
    )

    total_outliers = sum(
        outliers.values()
    )

    summary.append(
        f"⚠️ Outliers Detected: {total_outliers}"
    )

    return summary


# =========================
# 💡 CLEANING RECOMMENDATIONS
# =========================
def generate_cleaning_recommendations(df):

    recommendations = []

    missing = detect_missing_values(df)

    duplicates = detect_duplicates(df)

    dtype_issues = detect_dtype_issues(df)

    outliers = detect_outliers(df)

    if missing["Missing Values"].sum() > 0:

        recommendations.append(
            "Fill missing values using median for numeric columns and mode for categorical columns."
        )

    if duplicates > 0:

        recommendations.append(
            "Remove duplicate rows to improve data quality."
        )

    if dtype_issues:

        recommendations.append(
            "Convert incorrectly stored text columns into numeric format."
        )

    if sum(outliers.values()) > 0:

        recommendations.append(
            "Review extreme outliers before analysis."
        )

    if not recommendations:

        recommendations.append(
            "Dataset appears clean and analysis-ready."
        )

    return recommendations