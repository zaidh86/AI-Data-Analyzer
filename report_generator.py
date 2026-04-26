import datetime

def generate_report(df, insights):
    missing_values = df.isna().sum()

    report = []

    # Title
    report.append("=" * 60)
    report.append("AI DATA ANALYSIS REPORT")
    report.append("=" * 60)

    # Timestamp
    report.append(f"Generated on: {datetime.datetime.now()}")
    report.append("")

    # Dataset Info
    report.append("DATASET OVERVIEW")
    report.append("-" * 60)
    report.append(f"Rows: {df.shape[0]}")
    report.append(f"Columns: {df.shape[1]}")
    report.append(f"Total Missing Values: {int(missing_values.sum())}")
    report.append("")

    # Data Preview
    report.append("DATA PREVIEW (First 10 Rows)")
    report.append("-" * 60)
    report.append(df.head(10).to_string())
    report.append("")

    # Statistics
    report.append("STATISTICAL SUMMARY")
    report.append("-" * 60)
    report.append(df.describe(include="all").transpose().to_string())
    report.append("")

    # Missing Values
    report.append("MISSING VALUES BY COLUMN")
    report.append("-" * 60)
    report.append(missing_values.to_string())
    report.append("")

    # AI Insights
    report.append("AI INSIGHTS & RECOMMENDATIONS")
    report.append("-" * 60)

    if insights:
        report.append(insights)
    else:
        report.append("No insights generated.")

    report.append("")
    report.append("=" * 60)
    report.append("END OF REPORT")
    report.append("=" * 60)

    return "\n".join(report)