import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

def generate_answer(context: str, question: str) -> str:
    """
    Generate an answer using DeepSeek API based on the provided context and question.
    
    Args:
        context (str): The retrieved context from the document store
        question (str): The user's question
    
    Returns:
        str: The generated answer or error message
    """
    if not DEEPSEEK_API_KEY:
        return "Error: DeepSeek API key not configured."
    
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise corporate assistant. Answer based strictly on context. If answer isn't in context, say 'I cannot find an answer in the provided documents.'"
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ],
            temperature=0.1,
            max_tokens=1024,
            stream=False
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error contacting DeepSeek API: {str(e)}"