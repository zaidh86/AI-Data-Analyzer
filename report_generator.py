def generate_report(df, insights):
    report = ""

    report += "AI DATA ANALYSIS REPORT\n"
    report += "=" * 50 + "\n\n"

    report += "DATA PREVIEW:\n"
    report += df.head().to_string() + "\n\n"

    report += "STATISTICS:\n"
    report += df.describe().to_string() + "\n\n"

    report += "MISSING VALUES:\n"
    report += df.isnull().sum().to_string() + "\n\n"

    report += "AI INSIGHTS:\n"
    report += insights + "\n"

    return report