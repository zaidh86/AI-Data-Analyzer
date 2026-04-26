import os
from openai import OpenAI

# Load API key securely
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_insights(df):
    try:
        # Data summary
        summary = df.describe().to_string()
        columns = df.columns.tolist()
        missing = df.isnull().sum().to_string()
        dtypes = df.dtypes.to_string()

        prompt = f"""
You are an expert data analyst.

Analyze the dataset based on the following information:

COLUMNS:
{columns}

DATA TYPES:
{dtypes}

STATISTICS:
{summary}

MISSING VALUES:
{missing}

Provide a structured response with:

1. Key Trends
2. Notable Patterns
3. Outliers or Anomalies
4. Data Quality Issues
5. Business Insights
6. Actionable Recommendations

Keep the response clear, professional, and easy to understand.
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating insights: {str(e)}"