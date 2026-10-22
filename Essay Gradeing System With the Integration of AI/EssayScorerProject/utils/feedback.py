from groq import Groq

def get_feedback(essay_text, score, api_key):
    """
    Send essay text to Groq API (llama-3.3-70b-versatile) and get feedback.
    """
    if not api_key:
        return "Error: Groq API Key not found. Please provide your Groq API key."

    try:
        client = Groq(api_key=api_key)

        prompt = f"""You are an expert essay grader.
The following essay has been scored {score:.2f}/10 by an automated system.

Essay:
\"\"\"{essay_text}\"\"\"

Please provide constructive feedback on how to improve this essay.
Focus on structure, grammar, content, and clarity.
Keep the feedback concise and actionable, using bullet points where helpful."""

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional essay grader who gives clear, helpful, and constructive feedback."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1024,
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Error gathering feedback: {str(e)}"
