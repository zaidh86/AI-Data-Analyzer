from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

def get_insights(df):
    summary = df.describe().to_string()
    
    prompt = f"""
    You are a professional data analyst.

    Analyze this dataset summary:

    {summary}

    Provide:
    - Key trends
    - Outliers
    - Business insights
    - Recommendations
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content