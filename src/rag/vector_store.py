"""
FAISS vector store for semantic search.
"""
import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple
from pathlib import Path


class VectorStore:
    """FAISS-based vector store for semantic search."""
    
    def __init__(self, dimension: int = 384):
        """
        Initialize vector store.
        
        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict]):
        """
        Add documents and their embeddings to the store.
        
        Args:
            embeddings: numpy array of shape (n_docs, dimension)
            documents: list of document metadata
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match {self.dimension}")
        
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)
        
        print(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[Dict, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (document, distance) tuples
        """
        if len(self.documents) == 0:
            return []
        
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, min(k, len(self.documents)))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(dist)))
        
        return results
    
    def save(self, path: str):
        """Save vector store to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(path / "faiss.index"))
        with open(path / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        print(f"Saved vector store to {path}")
    
    def load(self, path: str):
        """Load vector store from disk."""
        path = Path(path)
        
        self.index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f"Loaded vector store from {path}")


def main():
    """Test vector store."""
    from embeddings import EmbeddingGenerator
    
    # Create test data
    texts = [
        "Our Q1 revenue was £2.4 million with strong growth.",
        "Customer satisfaction scores improved to 4.3 out of 5.",
        "Inventory turnover rate reached 6.8 times annually.",
        "We added 178 new enterprise customers this quarter."
    ]
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.generate_embeddings(texts)
    
    # Create documents
    docs = [{'content': text, 'id': i} for i, text in enumerate(texts)]
    
    # Build vector store
    store = VectorStore(dimension=384)
    store.add_documents(embeddings, docs)
    
    # Test search
    query = "How many customers did we get?"
    query_embedding = generator.generate_embedding(query)
    results = store.search(query_embedding, k=2)
    
    print(f"\nQuery: {query}")
    print("\nTop results:")
    for doc, distance in results:
        print(f"Distance: {distance:.3f}")
        print(f"Content: {doc['content']}\n")


if __name__ == "__main__":
    main()