import pandas as pd
import numpy as np


# =========================
# 🔍 MISSING VALUES
# =========================
def detect_missing_values(df):

    missing = df.isna().sum()

    return pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values,
        "Missing %": (
            missing.values / len(df) * 100
        ).round(2)
    })


# =========================
# 📄 DUPLICATES
# =========================
def detect_duplicates(df):

    return int(
        df.duplicated().sum()
    )


# =========================
# 🧩 DATA TYPE ISSUES
# =========================
def detect_dtype_issues(df):

    issues = []

    for col in df.columns:

        if df[col].dtype == "object":

            try:

                pd.to_numeric(df[col])

                issues.append(
                    f"{col} may be numeric but is stored as text."
                )

            except:
                pass

    return issues


# =========================
# ⚠️ OUTLIERS
# =========================
def detect_outliers(df):

    numeric = df.select_dtypes(
        include=np.number
    )

    report = {}

    for col in numeric.columns:

        q1 = numeric[col].quantile(0.25)

        q3 = numeric[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - (1.5 * iqr)

        upper = q3 + (1.5 * iqr)

        count = numeric[
            (numeric[col] < lower)
            | (numeric[col] > upper)
        ][col].count()

        report[col] = count

    return report


# =========================
# 🎯 QUALITY SCORE
# =========================
def calculate_quality_score(df):

    score = 100

    missing_percent = (
        df.isna().sum().sum()
        /
        (df.shape[0] * df.shape[1])
    ) * 100

    score -= min(
        40,
        missing_percent
    )

    duplicates = df.duplicated().sum()

    score -= min(
        20,
        duplicates
    )

    return round(
        max(score, 0)
    )


# =========================
# 🧹 CLEAN DATASET
# =========================
def clean_dataset(df):

    cleaned = df.copy()

    cleaned = cleaned.drop_duplicates()

    for col in cleaned.columns:

        if cleaned[col].dtype in [
            "float64",
            "int64"
        ]:

            cleaned[col] = (
                cleaned[col]
                .fillna(
                    cleaned[col].median()
                )
            )

        else:

            cleaned[col] = (
                cleaned[col]
                .fillna("Unknown")
            )

    return cleaned