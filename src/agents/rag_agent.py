"""
Agent integrated with RAG pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_pipeline import RAGPipeline
from .simple_agent import SimpleAgent


class RAGAgent:
    """Agent with full RAG capabilities."""
    
    def __init__(self):
        """Initialize agent with RAG pipeline."""
        print("Initializing RAG Agent...")
        self.rag_pipeline = RAGPipeline()
        self.agent = SimpleAgent(self.rag_pipeline)
        
    def load_documents(self, file_path: str):
        """Load documents into knowledge base."""
        self.rag_pipeline.load_document(file_path)
    
    def ask(self, question: str) -> str:
        """Ask the agent a question."""
        return self.agent.run(question)


def main():
    """Test RAG agent with business documents."""
    # Initialize agent
    agent = RAGAgent()
    
    # Load business documents
    agent.load_documents('data/sample_report.txt')
    
    print("\n" + "="*70)
    print("RAG Agent Ready")
    print("="*70)
    
    # Test questions
    questions = [
        "What was our Q1 2024 revenue?",
        "How many new customers did we acquire and calculate what percentage that is of our total customer base?"
    ]
    
    for question in questions:
        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print('='*70)
        
        answer = agent.ask(question)
        
        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()