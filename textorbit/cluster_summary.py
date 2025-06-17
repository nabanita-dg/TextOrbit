from transformers import pipeline
import streamlit as st
import time


@st.cache_resource
def load_summarizer():
    #return pipeline("summarization", model="google/flan-t5-large")
    return pipeline("summarization", model="t5-small", device=-1)


def generate_cluster_summary(text_list):
    joined_text = " ".join(text_list[:50])
    # prompt = f"Summarize the following text cluster:\n\n{joined_text}\n\nSummary:"
    start_time = time.time()
    summarizer = load_summarizer()
    response = summarizer(joined_text,
        max_length=120,
        min_length=10,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        repetition_penalty=2.0,  # <--- KEY
        num_return_sequences=1)
    summary = response[0]["summary_text"]
    end_time = time.time()
    elapsed = end_time - start_time
    return summary, elapsed


'''import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Access the token
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Create the client
client = InferenceClient(
    model="google/flan-t5-large",
    token=hf_token
)

def generate_cluster_summary(text_list):
    text = " ".join(text_list[:50])
    prompt = f"<s>[INST] Summarize this: {text} [/INST]"
    response = client.text_generation(prompt, max_new_tokens=200)
    return response


from transformers import pipeline

def generate_cluster_summary(text_list):
    joined_text = " ".join(text_list[:50])
    prompt = f"Summarize the following text cluster:\n\n{joined_text}\n\nSummary:"

    summarizer = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")
    response = summarizer(prompt, max_new_tokens=150, do_sample=True)
    return response[0]["generated_text"]


def generate_cluster_summary_openai(text_list):
    joined_text = " ".join(text_list[:50])  # Limit text size if needed
    prompt = f"Summarize the following text cluster:\n\n{joined_text}\n\nSummary:"

    # Example using OpenAI
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
'''
