import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

def generate_answer(context: str, question: str) -> str:
    if not GROQ_API_KEY:
        return "Groq API key not configured."

    try:
        response = requests.post(                      
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful insurance assistant."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
                ],
                "temperature": 0.2
            }
        )
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"Error contacting Groq API: {e}"
