import requests
from urllib.parse import urlparse
from io import BytesIO
import PyPDF2
from docx import Document

def load_document_from_url(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()

    if "pdf" in content_type:
        return extract_text_from_pdf(response.content)
    elif "word" in content_type or url.endswith(".docx"):
        return extract_text_from_docx(response.content)
    elif "text" in content_type or "email" in content_type:
        return extract_text_from_email(response.content)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def extract_text_from_pdf(pdf_bytes) -> str:
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    return "\n\n".join([page.extract_text() or "" for page in reader.pages])

def extract_text_from_docx(docx_bytes) -> str:
    doc = Document(BytesIO(docx_bytes))
    return "\n\n".join([para.text for para in doc.paragraphs])

def extract_text_from_email(email_bytes) -> str:
    from email import message_from_bytes
    msg = message_from_bytes(email_bytes)
    return msg.get_payload(decode=True).decode(errors="ignore")
