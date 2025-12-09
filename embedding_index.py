from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class EmbeddingIndex:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def chunk_text(self, text, chunk_size=100, overlap=20):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    def search(self, text: str, questions: list[str]) -> list[str]:
        chunks = self.chunk_text(text)
        chunk_embeddings = self.model.encode(chunks, convert_to_numpy=True)
        index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
        index.add(chunk_embeddings)

        results = []
        for q in questions:
            q_embed = self.model.encode([q], convert_to_numpy=True)
            _, I = index.search(q_embed, k=1)
            results.append(chunks[I[0][0]])

        return results
