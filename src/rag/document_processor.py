"""
Document processing and chunking for RAG system.
"""
import os
from typing import List, Dict
from pathlib import Path
import pandas as pd
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


class DocumentProcessor:
    """Process and chunk documents for RAG pipeline."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_text(self, file_path: str) -> str:
        """Load plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Error reading text file {file_path}: {str(e)}")
    
    def chunk_text(self, text: str) -> List[Dict]:
        """Split text into chunks."""
        chunks = self.text_splitter.split_text(text)
        
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            doc = {
                'content': chunk,
                'chunk_id': i
            }
            chunked_docs.append(doc)
        
        return chunked_docs


def main():
    """Test the document processor."""
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    
    sample_text = """
    Q1 2024 Sales Report
    Total Revenue: $1,250,000
    Units Sold: 45,000
    Customer Satisfaction: 4.3/5.0
    """ * 5
    
    chunks = processor.chunk_text(sample_text)
    print(f"Created {len(chunks)} chunks")
    print(f"First chunk: {chunks[0]['content'][:100]}...")


if __name__ == "__main__":
    main()