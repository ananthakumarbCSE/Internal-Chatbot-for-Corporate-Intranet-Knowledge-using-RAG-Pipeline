import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastapi import FastAPI, HTTPException, Header
from models import QueryRequest, QueryResponse
from document_loader import load_document_from_url
from embedding_index import EmbeddingIndex
from llm_answerer import generate_answer
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
AUTH_TOKEN = os.getenv("WEBHOOK_TOKEN", "demo-token")

embedding_index = EmbeddingIndex()

@app.get("/")
def root():
    return {"message": "Benchmark API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/Benchmark/run", response_model=QueryResponse)
def run_query(request: QueryRequest, authorization: str = Header(...)):
    if not authorization.endswith(AUTH_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Load document content
    document_text = load_document_from_url(request.documents)

    # Find relevant chunks
    relevant_chunks = embedding_index.search(document_text, request.questions)

    # Generate answers
    answers = []
    for question, context in zip(request.questions, relevant_chunks):
        answer = generate_answer(context, question)
        answers.append(answer)

    return QueryResponse(answers=answers)
