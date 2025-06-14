def generate_cluster_summary(text_list):
    joined_text = " ".join(text_list[:50])  # Limit text size if needed
    prompt = f"Summarize the following text cluster:\n\n{joined_text}\n\nSummary:"

    # Example using OpenAI
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content