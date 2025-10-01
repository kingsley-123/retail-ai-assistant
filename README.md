# AI-Powered Business Intelligence Assistant

Intelligent system combining RAG, LLMs, and machine learning for retail analytics and forecasting.

## Features (In Progress)

- ✅ Document processing with intelligent chunking (Day 1)
- ✅ Vector database (FAISS) with semantic search (Day 2)
- 🚧 LLM integration - Day 3
- 🚧 AI agent framework - Day 4
- 🚧 ML models (segmentation, churn, forecasting) - Days 5-6
- 🚧 Azure deployment - Days 7-8

## Tech Stack

- **Python 3.12**
- **LangChain** - Agent framework
- **FAISS** - Vector database for semantic search
- **Sentence Transformers** - Free local embeddings (384-dim)
- **Scikit-learn** - ML models
- **Pandas, NumPy** - Data processing

## What's Working Now

✅ Upload documents (text, PDF, CSV)
✅ Automatic text chunking with overlap
✅ Generate embeddings locally (no API needed)
✅ Semantic search - find relevant info by meaning, not keywords
✅ Working RAG pipeline

## Quick Start
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/rag/rag_pipeline.py