import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

def _get_api_key():
    # 1. Try environment variable (.env for local)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    # 2. Try Streamlit secrets (for deployment)
    try:
        import streamlit as st
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


def _build_prompt(df):
    # Limit size to prevent token explosion
    summary = df.describe(include="all").transpose().head(20).to_string()
    columns = df.columns.tolist()
    missing = df.isna().sum().to_string()
    dtypes = df.dtypes.to_string()
    sample_rows = df.head(5).to_string(index=False)

    return f"""
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

SAMPLE ROWS:
{sample_rows}

Provide a structured response with:

1. Key Trends
2. Notable Patterns
3. Outliers or Anomalies
4. Data Quality Issues
5. Business Insights
6. Actionable Recommendations

Keep the response clear, professional, and easy to understand for a non-technical user.
"""


def get_insights(df):
    api_key = _get_api_key()

    if not api_key:
        return (
            "⚠️ OpenAI API key not found.\n\n"
            "Please set OPENAI_API_KEY in your environment variables "
            "or add it to Streamlit secrets."
        )

    try:
        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _build_prompt(df)}],
            temperature=0.3,
            max_tokens=500  # prevents overly long/costly responses
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {e}"