```markdown
# Storm Technologies AI Business Intelligence Assistant

> An intelligent business analytics system combining Retrieval-Augmented Generation (RAG), Large Language Models, and Machine Learning to provide comprehensive insights from business documents and data.

---

## 🚀 Live Demo

**Try it now:** [https://retail-ai-assistant-kadznzjszyhurmemplvjth.streamlit.app/](https://retail-ai-assistant-kadznzjszyhurmemplvjth.streamlit.app/)

> **Note:** Cloud deployment runs ML analytics. Full LLM features available when running locally with Ollama.

---

## 📘 Project Overview

### Objective
This project demonstrates an end-to-end AI system that combines modern RAG architecture with production-ready machine learning models to extract actionable insights from business data. The system enables organizations to make faster, data-driven decisions through natural language queries and predictive analytics.

### Business Problem
Many organizations struggle with:
- **Information Overload**: Critical insights buried in lengthy business documents
- **Manual Analysis**: Time-consuming manual report reviews and data analysis
- **Reactive Decision Making**: Lack of predictive insights for customer churn and sales trends
- **Fragmented Tools**: Separate systems for document search, ML analytics, and reporting

### Goal
Build an intelligent system that:
- Enables natural language queries across business documents
- Provides AI-driven insights through predictive models
- Automates customer segmentation and churn risk assessment
- Delivers accurate sales forecasts for resource planning
- Runs entirely on free, open-source technology stack

---

## 🛠️ Tools & Technologies Used

| Category | Tool/Technology | Purpose |
|----------|----------------|---------|
| **LLM** | Ollama (Llama 3.2) | Natural language understanding and generation |
| **Vector Database** | FAISS | Fast semantic search and document retrieval |
| **Embeddings** | Sentence Transformers | Text vectorization (all-MiniLM-L6-v2) |
| **ML Framework** | Scikit-learn | Customer segmentation, churn prediction, forecasting |
| **Data Processing** | Pandas, NumPy | Data manipulation and numerical computing |
| **Web Interface** | Streamlit | Interactive dashboard and user interface |
| **Deployment** | Streamlit Cloud | Free cloud hosting |
| **Version Control** | Git/GitHub | Code management and collaboration |

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit Web Interface                    │
│              (User Q&A + ML Analytics Dashboard)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼────────┐           ┌────────▼────────┐
│  RAG Pipeline  │           │  ML Analytics   │
├────────────────┤           ├─────────────────┤
│• Doc Processor │           │• Segmentation   │
│• Embeddings    │           │• Churn Model    │
│• Vector Store  │           │• Forecasting    │
│• LLM Client    │           │                 │
└────────┬───────┘           └────────┬────────┘
         │                            │
    ┌────▼─────┐                 ┌────▼─────┐
    │  FAISS   │                 │ Sklearn  │
    │Vector DB │                 │  Models  │
    └──────────┘                 └──────────┘
         │                            │
    ┌────▼─────┐                 ┌────▼─────┐
    │  Ollama  │                 │Customer/ │
    │Llama 3.2 │                 │Sales Data│
    └──────────┘                 └──────────┘
```

---

## 📋 Features

### 1. 📄 Document Intelligence (RAG System)
- **Semantic Search**: FAISS vector database for lightning-fast document retrieval
- **Natural Language Q&A**: Ask questions in plain English, get context-aware answers
- **Multi-Document Support**: Process and search across multiple business reports
- **Context-Aware Responses**: LLM generates answers using relevant document snippets

### 2. 🤖 AI Agent Framework
- **Multi-Step Reasoning**: Agent breaks down complex queries into steps
- **Autonomous Planning**: Automatically selects and orchestrates tools
- **Tool Selection**: Chooses between document search, calculations, and formatting
- **Extensible Architecture**: Easy to add new capabilities and tools

### 3. 📈 Machine Learning Analytics

#### Customer Segmentation (K-means Clustering)
- RFM (Recency, Frequency, Monetary) analysis
- 3 customer segments: Champions, At Risk, Loyal Customers
- Behavioral insights for targeted marketing campaigns

#### Churn Prediction (Random Forest)
- **83.3% accuracy** on test set
- Risk level classification: High/Medium/Low
- Feature importance analysis
- Proactive customer retention strategies

#### Sales Forecasting (Time Series)
- Multi-period forecasting (1-6 months ahead)
- **2.9% MAPE** (Mean Absolute Percentage Error)
- Trend analysis and growth metrics
- Revenue planning and resource optimization

---

## 🎯 Model Performance

| Model | Metric | Performance |
|-------|--------|-------------|
| **Customer Segmentation** | Silhouette Score | 0.68 |
| **Churn Prediction** | Accuracy | 83.3% |
| **Churn Prediction** | Precision (Churned) | 67% |
| **Churn Prediction** | Recall (Churned) | 100% |
| **Sales Forecasting** | MAPE | 2.9% |
| **Sales Forecasting** | RMSE | £4,349 |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Ollama (for local LLM features)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/kingsley-123/retail-ai-assistant.git
cd retail-ai-assistant
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Install Ollama and pull model** (for full features)
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2
```

**5. Run the application**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 💡 Usage Examples

### Document Q&A
```
Q: "What was our Q1 2024 revenue?"

A: Based on the Q1 2024 report, total revenue was £2,450,000, 
   representing a 15% increase compared to Q1 2023.
```

### Customer Segmentation
```
Champions: 5 customers
  - Avg Recency: 14 days
  - Avg Frequency: 18.2 purchases
  - Avg Monetary: £5,460

At Risk: 7 customers  
  - Avg Recency: 39 days
  - Avg Frequency: 10.0 purchases
  - Avg Monetary: £3,000
```

### Churn Prediction
```
Customer Profile:
  - Recency: 150 days
  - Frequency: 2 purchases
  - Monetary: £600

Prediction Result:
  ⚠️ Churn Probability: 100%
  ⚠️ Risk Level: HIGH
  💡 Action: Immediate retention campaign required
```

### Sales Forecasting
```
Next 3 Months Forecast:
  📊 October 2024: £142,344
  📊 November 2024: £145,376
  📊 December 2024: £148,408

Trend: ↗️ Upward (65.8% annual growth rate)
Monthly Change: +£3,032
```

---

## 📁 Project Structure

```
retail-ai-assistant/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── data/
│   ├── sample_report.txt          # Sample business document
│   ├── customer_data.csv          # Customer dataset
│   └── sales_data.csv             # Sales time series data
│
├── src/
│   ├── rag/
│   │   ├── document_processor.py  # Text chunking & preprocessing
│   │   ├── embeddings.py          # Sentence embeddings generation
│   │   ├── vector_store.py        # FAISS database operations
│   │   ├── llm_client.py          # Ollama interface
│   │   └── rag_pipeline.py        # Complete RAG system
│   │
│   ├── agents/
│   │   ├── tools.py               # Agent tools (search, calculate, format)
│   │   ├── simple_agent.py        # Planning & execution agent
│   │   ├── ml_tools.py            # ML model integration
│   │   └── rag_agent.py           # RAG-enabled agent
│   │
│   └── ml/
│       ├── customer_segmentation.py   # K-means clustering
│       ├── churn_prediction.py        # Random Forest classifier
│       └── sales_forecasting.py       # Time series forecasting
│
└── models/                        # Saved model artifacts
```

---

## 📊 Business Impact

This system enables organizations to:

| Impact Area | Benefit |
|------------|---------|
| ⚡ **Speed** | Instant insights from business documents (vs. hours of manual review) |
| 💰 **Revenue** | Proactive churn prevention saves customer lifetime value |
| 📈 **Planning** | Accurate sales forecasts enable optimal resource allocation |
| 🎯 **Marketing** | Targeted campaigns based on customer segments increase ROI |
| ⏱️ **Efficiency** | Automated analysis reduces manual effort by 80% |

---

## 🔮 Future Enhancements

- [ ] Real-time data pipeline integration with streaming data
- [ ] Multi-language support for international documents
- [ ] Advanced NLP: Named Entity Recognition, sentiment analysis
- [ ] Deep learning models (LSTM/Transformer) for forecasting
- [ ] A/B testing framework for business experiments
- [ ] REST API endpoints for programmatic access
- [ ] Database integration (PostgreSQL/MongoDB)
- [ ] User authentication and role-based access control
- [ ] Automated model retraining pipeline with MLOps
- [ ] Custom dashboard builder for business users

---

## 👤 Author

**Kingsley Okonkwo**  
MSc Data Science | University of Surrey

[![GitHub](https://img.shields.io/badge/GitHub-kingsley--123-181717?logo=github)](https://github.com/kingsley-123)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-kingsleyokonkwo-0077B5?logo=linkedin)](https://linkedin.com/in/kingsleyokonkwo)

---

## 📄 License

This project is created as part of a job application portfolio for Storm Technologies.

---

## 🙏 Acknowledgments

Built for **Storm Technologies' AI & Data Science position**. This project demonstrates proficiency in:

✅ Production-ready ML systems  
✅ RAG architecture and LLM integration  
✅ Cloud deployment and DevOps  
✅ Clean code and comprehensive documentation  
✅ End-to-end AI product development  


