from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import uuid
from datetime import datetime


class VectorStoreService:
    
    def __init__(self, host="localhost", port=6333):
        # Connect to Qdrant
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = "documents"
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        collection_exists = any(
            c.name == self.collection_name for c in collections
        )
        
        if not collection_exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Created collection: {self.collection_name}")
    
    def add_documents(self, texts: List[str], metadata: List[Dict]) -> List[str]:

        # Create embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Create points for Qdrant
        points = []
        doc_ids = []
        
        for text, embedding, meta in zip(texts, embeddings, metadata):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            points.append(
                PointStruct(
                    id=doc_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": text,
                        "filename": meta.get("filename", "unknown"),
                        "chunk_index": meta.get("chunk_index", 0),
                        "uploaded_at": meta.get("uploaded_at", datetime.now().isoformat()),
                        **meta  # Include all other metadata
                    }
                )
            )
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"✅ Added {len(points)} chunks to Qdrant")
        return doc_ids
    
    def search(self, query: str, top_k: int = 3, filters: Dict = None) -> List[Dict]:
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=filters  # Apply filters if provided
        )
        
        # Format results
        results = []
        for hit in search_results:
            results.append({
                "text": hit.payload["text"],
                "score": hit.score,
                "metadata": {
                    k: v for k, v in hit.payload.items() if k != "text"
                }
            })
        
        return results
    
    async def health_check(self) -> bool:
        """Check if Qdrant is healthy"""
        try:
            self.client.get_collections()
            return True
        except:
            return False


# Example usage (for testing)
if __name__ == "__main__":
    # Create service
    vs = VectorStoreService()
    
    # Add sample documents
    texts = [
        "The insurance policy covers collision damage.",
        "Your deductible is $500 for comprehensive coverage.",
        "Claims must be filed within 30 days."
    ]
    
    metadata = [
        {"filename": "policy.pdf", "chunk_index": 0},
        {"filename": "policy.pdf", "chunk_index": 1},
        {"filename": "policy.pdf", "chunk_index": 2}
    ]
    
    doc_ids = vs.add_documents(texts, metadata)
    print(f"Added documents: {doc_ids}")
    
    # Search
    results = vs.search("What's the deductible?", top_k=2)
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Text: {result['text']}")
        print(f"Score: {result['score']:.4f}")