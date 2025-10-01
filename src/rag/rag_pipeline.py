"""
Complete RAG pipeline: documents -> chunks -> embeddings -> search.
"""
from pathlib import Path
from typing import List, Dict, Tuple
from document_processor import DocumentProcessor
from embeddings import EmbeddingGenerator
from vector_store import VectorStore


class RAGPipeline:
    """End-to-end RAG pipeline."""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore(dimension=384)
        
    def load_document(self, file_path: str):
        """Load and process a document into the vector store."""
        print(f"\nProcessing: {file_path}")
        
        # Load and chunk document
        text = self.doc_processor.load_text(file_path)
        chunks = self.doc_processor.chunk_text(text)
        
        # Generate embeddings
        chunk_texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_generator.generate_embeddings(chunk_texts)
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            chunk['source'] = Path(file_path).name
        
        # Store in vector database
        self.vector_store.add_documents(embeddings, chunks)
        
        print(f"Loaded {len(chunks)} chunks from {Path(file_path).name}")
    
    def query(self, question: str, k: int = 3) -> List[Tuple[Dict, float]]:
        """Query the knowledge base."""
        query_embedding = self.embedding_generator.generate_embedding(question)
        results = self.vector_store.search(query_embedding, k=k)
        return results
    
    def answer_question(self, question: str, k: int = 3) -> str:
        """Get answer with context from documents."""
        results = self.query(question, k=k)
        
        if not results:
            return "No relevant information found."
        
        # Format answer
        answer = f"Based on the documents:\n\n"
        for i, (doc, distance) in enumerate(results, 1):
            answer += f"{i}. {doc['content'][:200]}...\n"
            answer += f"   (Relevance: {1/(1+distance):.2%})\n\n"
        
        return answer


def main():
    """Test complete RAG pipeline."""
    pipeline = RAGPipeline()
    
    # Load the sample business report
    pipeline.load_document('data/sample_report.txt')
    
    # Test queries
    questions = [
        "What was the total revenue?",
        "How is customer satisfaction?",
        "Tell me about inventory management",
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 60)
        answer = pipeline.answer_question(question, k=2)
        print(answer)


if __name__ == "__main__":
    main()