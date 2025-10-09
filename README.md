# Storm Technologies AI Business Intelligence Assistant

An intelligent business analytics system combining Retrieval-Augmented Generation (RAG), Large Language Models, and Machine Learning to provide comprehensive insights from business documents and data.

## 🚀 Live Demo

**Try it now:** [https://retail-ai-assistant-kadznzjszyhurmemplvjth.streamlit.app/](https://retail-ai-assistant-kadznzjszyhurmemplvjth.streamlit.app/)

*Note: Cloud deployment runs ML analytics. Full LLM features available when running locally with Ollama.*

## 📋 Features

### 1. Document Intelligence (RAG System)
- **Semantic Search**: FAISS vector database for fast document retrieval
- **Natural Language Q&A**: Ask questions about business documents in plain English
- **Context-Aware Responses**: LLM generates answers using relevant document context
- **Multi-Document Support**: Process and search across multiple business reports

### 2. AI Agent Framework
- **Multi-Step Reasoning**: Agent can plan and execute complex tasks
- **Tool Selection**: Automatically chooses appropriate tools for each query
- **Autonomous Planning**: Breaks down complex questions into actionable steps
- **Extensible Architecture**: Easy to add new tools and capabilities

### 3. Machine Learning Analytics
- **Customer Segmentation**: RFM analysis using K-means clustering
  - Champions, At Risk, and Loyal Customer segments
  - Behavioral insights for targeted marketing
  
- **Churn Prediction**: Random Forest classifier for customer retention
  - Predicts churn probability with 83% accuracy
  - Risk level classification (High/Medium/Low)
  - Feature importance analysis
  
- **Sales Forecasting**: Time series prediction for revenue planning
  - Multi-period forecasting (1-6 months)
  - Trend analysis and growth metrics
  - 2.9% MAPE (Mean Absolute Percentage Error)

## 🛠️ Technology Stack

### AI & Machine Learning
- **LLM**: Ollama (Llama 3.2) for natural language understanding
- **Vector Database**: FAISS for semantic search
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **ML Framework**: Scikit-learn for predictive models
- **Agent Framework**: Custom agent system with tool orchestration

### Backend & Data
- **Language**: Python 3.12
- **Data Processing**: Pandas, NumPy
- **Document Processing**: Custom chunking with overlap
- **Visualization**: Matplotlib, Seaborn

### Deployment
- **Web Interface**: Streamlit
- **Hosting**: Streamlit Community Cloud (Free tier)
- **Version Control**: GitHub

## 📊 System Architecture
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web Interface                  │
└──────────────────────┬──────────────────────────────────────┘
│
┌──────────────┴──────────────┐
│                             │
┌───────▼────────┐           ┌────────▼────────┐
│   RAG Pipeline │           │   ML Analytics   │
├────────────────┤           ├─────────────────┤
│ • Doc Processor│           │ • Segmentation  │
│ • Embeddings   │           │ • Churn Model   │
│ • Vector Store │           │ • Forecasting   │
│ • LLM Client   │           │                 │
└────────┬───────┘           └────────┬────────┘
│                            │
┌────▼─────┐                 ┌────▼─────┐
│  FAISS   │                 │ Sklearn  │
│ Vector DB│                 │  Models  │
└──────────┘                 └──────────┘
│                            │
┌────▼─────┐                 ┌────▼─────┐
│  Ollama  │                 │Customer/ │
│ Llama3.2 │                 │Sales Data│
└──────────┘                 └──────────┘

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Ollama (for local LLM features)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/kingsley-123/retail-ai-assistant.git
cd retail-ai-assistant

Create virtual environment

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies

bashpip install -r requirements.txt

Install Ollama and pull model (for full features)

bash# Install Ollama from https://ollama.ai
ollama pull llama3.2

Run the application

bashstreamlit run app.py
The app will open at http://localhost:8501
💡 Usage Examples
Document Q&A
Q: "What was our Q1 2024 revenue?"
A: Based on the Q1 2024 report, total revenue was £2,450,000, 
   representing a 15% increase compared to Q1 2023.
Customer Segmentation
python# Returns customer segments with RFM metrics:
- Champions: 5 customers (Avg: 14 days recency, 18.2 purchases, £5,460 spend)
- At Risk: 7 customers (Avg: 39 days recency, 10.0 purchases, £3,000 spend)
- Loyal Customers: 8 customers (Avg: 150 days recency, 2.6 purchases, £788 spend)
Churn Prediction
pythonCustomer Profile:
- Recency: 150 days
- Frequency: 2 purchases
- Monetary: £600

Result:
- Churn Probability: 100%
- Risk Level: High
- Recommendation: Immediate retention action required
Sales Forecasting
pythonNext 3 Months Forecast:
- October 2024: £142,344
- November 2024: £145,376
- December 2024: £148,408

Trend: Upward (65.8% annual growth rate)
📁 Project Structure
retail-ai-assistant/
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── data/
│   ├── sample_report.txt       # Sample business document
│   ├── customer_data.csv       # Customer dataset
│   └── sales_data.csv          # Sales time series data
├── src/
│   ├── rag/
│   │   ├── document_processor.py   # Text chunking
│   │   ├── embeddings.py           # Sentence embeddings
│   │   ├── vector_store.py         # FAISS database
│   │   ├── llm_client.py           # Ollama interface
│   │   └── rag_pipeline.py         # Complete RAG system
│   ├── agents/
│   │   ├── tools.py                # Agent tools (search, calculate)
│   │   ├── simple_agent.py         # Planning & execution agent
│   │   ├── ml_tools.py             # ML model tools
│   │   └── rag_agent.py            # RAG-enabled agent
│   └── ml/
│       ├── customer_segmentation.py   # K-means clustering
│       ├── churn_prediction.py        # Random Forest classifier
│       └── sales_forecasting.py       # Time series model
└── models/                     # Saved model artifacts
🎯 Model Performance
ModelMetricPerformanceCustomer SegmentationSilhouette Score0.68Churn PredictionAccuracy83.3%Churn PredictionPrecision (Churned)67%Sales ForecastingMAPE2.9%Sales ForecastingRMSE£4,349
🔮 Future Enhancements

 Real-time data pipeline integration
 Multi-language support for documents
 Advanced NLP: Named Entity Recognition, sentiment analysis
 Deep learning models (LSTM/Transformer for forecasting)
 A/B testing framework for business experiments
 API endpoints for programmatic access
 Database integration (PostgreSQL/MongoDB)
 User authentication and role-based access
 Automated model retraining pipeline
 Custom dashboard builder

📊 Business Impact
This system enables:

Faster Decision Making: Instant insights from business documents
Reduced Churn: Proactive identification of at-risk customers
Revenue Optimization: Accurate sales forecasting for resource planning
Targeted Marketing: Customer segmentation for personalized campaigns
Cost Savings: Automated analysis reduces manual effort by 80%

👤 Author
Kingsley Okonkwo

MSc Data Science, University of Surrey
GitHub: @kingsley-123
LinkedIn: linkedin.com/in/kingsleyokonkwo

📄 License
This project is created as part of a job application for Storm Technologies.
🙏 Acknowledgments
Built for Storm Technologies' AI & Data Science position. This project demonstrates proficiency in:

Production-ready ML systems
RAG architecture and LLM integration
Cloud deployment and DevOps
Clean code and documentation
End-to-end AI product development


Note: This is a portfolio project demonstrating AI/ML capabilities. For production deployment, additional security, testing, and monitoring would be implemented.