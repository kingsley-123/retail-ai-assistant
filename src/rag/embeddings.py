"""
Generate embeddings using sentence-transformers.
"""
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingGenerator:
    """Generate embeddings for text using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of sentence-transformer model
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        return self.model.encode(text, convert_to_numpy=True)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)


def main():
    """Test embedding generation."""
    generator = EmbeddingGenerator()
    
    # Test with sample text
    text = "This is a test about retail sales and customer analytics."
    embedding = generator.generate_embedding(text)
    
    print(f"\nGenerated embedding:")
    print(f"Shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")


if __name__ == "__main__":
    main()