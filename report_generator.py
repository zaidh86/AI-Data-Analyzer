def generate_report(df, insights):
    missing_values = df.isna().sum()

    report = []
    report.append("AI DATA ANALYSIS REPORT")
    report.append("=" * 50)
    report.append("")
    report.append(f"Rows: {df.shape[0]}")
    report.append(f"Columns: {df.shape[1]}")
    report.append(f"Missing cells: {int(missing_values.sum())}")
    report.append("")
    report.append("DATA PREVIEW:")
    report.append(df.head(10).to_string())
    report.append("")
    report.append("STATISTICS:")
    report.append(df.describe(include="all").transpose().to_string())
    report.append("")
    report.append("MISSING VALUES:")
    report.append(missing_values.to_string())
    report.append("")
    report.append("AI INSIGHTS:")
    report.append(insights)

    return "\n".join(report)
