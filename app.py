"""
Storm Technologies AI Business Intelligence Assistant
Web interface for RAG and ML capabilities
"""
import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from rag.rag_pipeline import RAGPipeline
from agents.ml_tools import MLToolKit

# Page config
st.set_page_config(
    page_title="Storm Technologies AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    with st.spinner("Loading AI models..."):
        st.session_state.rag_pipeline = RAGPipeline()
        st.session_state.rag_pipeline.load_document('data/sample_report.txt')

if 'ml_toolkit' not in st.session_state:
    with st.spinner("Loading ML models..."):
        st.session_state.ml_toolkit = MLToolKit()

# Sidebar
st.sidebar.title("🤖 Storm Technologies")
st.sidebar.markdown("### AI Business Intelligence Assistant")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "💬 Document Q&A", "📊 ML Analytics"]
)

# HOME PAGE
if page == "🏠 Home":
    st.title("AI-Powered Business Intelligence System")
    st.markdown("### Built for Storm Technologies")
    
    st.markdown("""
    This intelligent system combines **Retrieval-Augmented Generation (RAG)**, 
    **Large Language Models**, and **Machine Learning** to provide comprehensive 
    business analytics and insights.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📄 Document Intelligence")
        st.markdown("""
        - Upload business documents
        - Semantic search with FAISS
        - Natural language Q&A
        - Context-aware responses
        """)
    
    with col2:
        st.markdown("### 🤖 AI Agent")
        st.markdown("""
        - Multi-step reasoning
        - Tool selection
        - Autonomous planning
        - Complex query handling
        """)
    
    with col3:
        st.markdown("### 📈 ML Analytics")
        st.markdown("""
        - Customer segmentation
        - Churn prediction
        - Sales forecasting
        - Trend analysis
        """)
    
    st.markdown("---")
    st.markdown("### Technology Stack")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        **AI & ML:**
        - LangChain (Agent framework)
        - Ollama/Llama 3.2 (LLM)
        - FAISS (Vector database)
        - Sentence Transformers (Embeddings)
        - Scikit-learn (ML models)
        """)
    
    with tech_col2:
        st.markdown("""
        **Data & Deployment:**
        - Python 3.12
        - Pandas, NumPy
        - Streamlit (Web interface)
        - GitHub (Version control)
        - Free cloud hosting
        """)

# DOCUMENT Q&A PAGE
elif page == "💬 Document Q&A":
    st.title("📄 Document Question & Answer")
    st.markdown("Ask questions about your business documents using RAG + LLM")
    
    # Show loaded documents
    st.info("📁 Loaded: Storm Technologies Q1 2024 Business Report")
    
    # Sample questions
    with st.expander("💡 Try these sample questions"):
        st.markdown("""
        - What was the total revenue in Q1 2024?
        - How many new customers did we acquire?
        - What is our customer satisfaction score?
        - Tell me about inventory management performance
        - What are the top product categories by revenue?
        """)
    
    # Question input
    question = st.text_input(
        "Ask a question:",
        placeholder="e.g., What was our Q1 revenue?"
    )
    
    if st.button("Get Answer", type="primary"):
        if question:
            with st.spinner("Searching documents and generating answer..."):
                answer = st.session_state.rag_pipeline.answer_question(question, k=2)
                
                st.markdown("### Answer:")
                st.success(answer)
        else:
            st.warning("Please enter a question")

# ML ANALYTICS PAGE
elif page == "📊 ML Analytics":
    st.title("📊 Machine Learning Analytics")
    
    tab1, tab2, tab3 = st.tabs([
        "👥 Customer Segmentation", 
        "⚠️ Churn Prediction", 
        "📈 Sales Forecasting"
    ])
    
    # Customer Segmentation Tab
    with tab1:
        st.markdown("### Customer Segmentation Analysis")
        st.markdown("Customers grouped by RFM (Recency, Frequency, Monetary) behavior")
        
        if st.button("Run Segmentation Analysis"):
            with st.spinner("Analyzing customer segments..."):
                result = st.session_state.ml_toolkit.analyze_customer_segments()
                st.text(result)
    
    # Churn Prediction Tab
    with tab2:
        st.markdown("### Churn Risk Prediction")
        st.markdown("Predict which customers are at risk of leaving")
        
        col1, col2 = st.columns(2)
        
        with col1:
            recency = st.number_input("Days since last purchase", min_value=1, value=150)
            frequency = st.number_input("Total purchases", min_value=1, value=2)
            monetary = st.number_input("Total spent (£)", min_value=0, value=600)
        
        with col2:
            tenure = st.number_input("Months as customer", min_value=1, value=10)
            avg_order = st.number_input("Average order value (£)", min_value=0, value=300)
        
        if st.button("Predict Churn Risk"):
            with st.spinner("Analyzing churn risk..."):
                result = st.session_state.ml_toolkit.predict_customer_churn(
                    recency, frequency, monetary, tenure, avg_order
                )
                st.text(result)
    
    # Sales Forecasting Tab
    with tab3:
        st.markdown("### Sales Forecasting")
        st.markdown("Predict future sales based on historical trends")
        
        periods = st.slider("Forecast periods (months)", min_value=1, max_value=6, value=3)
        
        if st.button("Generate Forecast"):
            with st.spinner("Generating sales forecast..."):
                result = st.session_state.ml_toolkit.forecast_sales(periods=periods)
                st.text(result)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Project by:** Kingsley Okonkwo")
st.sidebar.markdown("[GitHub Repository](https://github.com/kingsley-123/retail-ai-assistant)")