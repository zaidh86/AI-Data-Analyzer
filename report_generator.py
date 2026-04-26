from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from io import BytesIO


def generate_report(df, insights):
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    content = []

    # Title
    content.append(Paragraph("AI DATA ANALYSIS REPORT", styles["Title"]))
    content.append(Spacer(1, 20))

    # Dataset Info
    content.append(Paragraph(f"Rows: {df.shape[0]}", styles["Normal"]))
    content.append(Paragraph(f"Columns: {df.shape[1]}", styles["Normal"]))
    content.append(Spacer(1, 20))

    content.append(Paragraph("Summary:", styles["Heading2"]))
    content.append(Spacer(1, 10))

    summary_text = f"""
    This dataset contains {df.shape[0]} rows and {df.shape[1]} columns.
    The analysis highlights trends, correlations, and business insights.
    """
    content.append(Paragraph(summary_text, styles["Normal"]))
    content.append(Spacer(1, 20))

    # Insights
    content.append(Paragraph("Insights:", styles["Heading2"]))
    content.append(Spacer(1, 10))

    for line in insights.split("\n"):
        content.append(Paragraph(line, styles["Normal"]))
        content.append(Spacer(1, 8))

    doc.build(content)

    buffer.seek(0)
    return buffer