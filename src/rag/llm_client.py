"""
LLM integration using Ollama.
"""
import ollama
from typing import List, Dict


class OllamaClient:
    """Client for Ollama LLM."""
    
    def __init__(self, model: str = "llama3.2"):
        """
        Initialize Ollama client.
        
        Args:
            model: Name of the Ollama model to use
        """
        self.model = model
        print(f"Initialized Ollama with model: {model}")
    
    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer based on question and context.
        
        Args:
            question: User's question
            context: Relevant document chunks
            
        Returns:
            Generated answer
        """
        prompt = f"""Based on the following information from business documents, please answer the question.

Context:
{context}

Question: {question}

Answer: Provide a clear, concise answer based only on the information given in the context. If the context doesn't contain relevant information, say so."""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            return response['message']['content']
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"


def main():
    """Test the LLM client."""
    client = OllamaClient()
    
    # Test with sample context
    context = """
    Total Revenue: £2,450,000 (15% increase YoY)
    Customer Satisfaction: 4.3/5.0
    New Customers: 178
    """
    
    question = "What was the total revenue?"
    
    print(f"Question: {question}\n")
    answer = client.generate_answer(question, context)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()