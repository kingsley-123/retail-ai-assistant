"""
Tools that the agent can use.
"""
from typing import Dict, Any
import json


class ToolKit:
    """Collection of tools for the agent."""
    
    def __init__(self, rag_pipeline=None):
        """
        Initialize toolkit.
        
        Args:
            rag_pipeline: RAG pipeline for document search
        """
        self.rag_pipeline = rag_pipeline
        
    def search_documents(self, query: str) -> str:
        """
        Search business documents for information.
        
        Args:
            query: Search query
            
        Returns:
            Relevant information from documents
        """
        if not self.rag_pipeline:
            return "Document search not available."
        
        results = self.rag_pipeline.query(query, k=2)
        
        if not results:
            return "No relevant information found."
        
        output = "Found information:\n\n"
        for doc, distance in results:
            output += f"{doc['content'][:300]}...\n\n"
        
        return output
    
    def calculate(self, expression: str) -> str:
        """
        Perform mathematical calculations.
        
        Args:
            expression: Math expression to evaluate
            
        Returns:
            Calculation result
        """
        try:
            result = eval(expression)
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Error in calculation: {str(e)}"
    
    def format_data(self, data: Dict[str, Any]) -> str:
        """
        Format data for presentation.
        
        Args:
            data: Dictionary of data to format
            
        Returns:
            Formatted string
        """
        try:
            formatted = "Formatted Data:\n"
            for key, value in data.items():
                formatted += f"- {key}: {value}\n"
            return formatted
        except Exception as e:
            return f"Error formatting data: {str(e)}"


def main():
    """Test the tools."""
    toolkit = ToolKit()
    
    # Test calculator
    print("Testing calculator:")
    print(toolkit.calculate("2450000 * 0.15"))
    print()
    
    # Test data formatter
    print("Testing formatter:")
    data = {
        "Revenue": "£2,450,000",
        "Growth": "15%",
        "Customers": 1245
    }
    print(toolkit.format_data(data))


if __name__ == "__main__":
    main()