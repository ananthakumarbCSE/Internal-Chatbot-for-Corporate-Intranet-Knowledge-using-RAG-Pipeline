import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, Header
from your_project.models import QueryRequest, QueryResponse
from your_project.document_loader import load_document_from_url
from your_project.embedding_index import EmbeddingIndex
from your_project.llm_answerer import generate_answer
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/chat")
async def chat_interface():
    """
    Serve the HTML chat interface
    """
    return FileResponse("templates/index.html")

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
