import streamlit as st

# Define dictionaries for hyperparameters
vector_store_params = {
    "chunk_size": "How large each document chunk is before embedding. Affects recall vs. precision.",
    "chunk_overlap": "Overlap between chunks to preserve context continuity.",
    "text_splitter_type": "Type of text splitter to use (e.g., Sentence-aware (NLTK) or Character-based (Recursive)).",
    "embedding_source": "Source for embeddings (e.g., OpenAI or Hugging Face).",
    "embedding_model": "Which model to use for creating document embeddings."
}

model_config_params = {
    "llm_model_name": "Which GPT model to use for generation (`gpt-4-turbo`, `gpt-3.5-turbo`, etc.).",
    "k (top-k retrieval)": "Number of relevant chunks to pull from the vector store (higher = more context).",
    "temperature": "Controls randomness of output (lower = more factual, higher = more diverse)."
}

generation_params = {
    "prompt_template": "The system prompt structure (can be edited for consistent tone)."
}

# Must be the first Streamlit command
st.set_page_config(
    page_title="RPEC Annual Mortality Dashboard",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None 
)

st.title("📘 RPEC Annual Mortality Dashboard")

st.markdown("""
Welcome to the **RPEC Annual Mortality Dashboard**, a dynamic tool powered by AI to help actuaries and analysts generate narrative sections of the SOA's Retirement Plans Experience Committee (RPEC) report.
""")

st.info("""
*This app is designed to specifically generate analysis on **mortality experience during the COVID-19 pandemic**.*
""")

st.markdown("""
This application uses **RAG (Retrieval-Augmented Generation)** techniques with **LangChain** and **OpenAI** models to synthesize structured insights from:
- Prior RPEC PDF reports (embedded in a vector store)
- A synthetic dataset of mortality metrics for the year

---
""")

st.markdown(
"""
### 🧠 What is RAG (Retrieval-Augmented Generation)?
RAG enhances Large Language Models (LLMs) by allowing them to retrieve relevant knowledge from external sources before generating responses. This is critical for:
- Citing accurate information from proprietary or domain-specific documents
- Generating structured outputs that reference embedded content (like figures or tables)

We use **LangChain**, a powerful Python framework for building RAG workflows. It connects document loaders, embeddings, retrievers, LLMs, and outputs.

---
""")    

st.markdown(
"""
### 🔧 Hyperparameters You Can Control
Below is a quick summary of the key RAG components you can tune throughout the app:

#### **Vector Store**
""" + 
("| Hyperparameter       | Description                                                                 |\n" +
"|----------------------|-----------------------------------------------------------------------------|\n" + 
"\n".join([f"| `{key}` | {value} |" for key, value in vector_store_params.items()])) + 
"""

#### **Model Configuration**
""" + 
("| Hyperparameter       | Description                                                                 |\n" +
"|----------------------|-----------------------------------------------------------------------------|\n" + 
"\n".join([f"| `{key}` | {value} |" for key, value in model_config_params.items()])) + 
"""

#### **Generation**
""" + 
("| Hyperparameter       | Description                                                                 |\n" +
"|----------------------|-----------------------------------------------------------------------------|\n" + 
"\n".join([f"| `{key}` | {value} |" for key, value in generation_params.items()])) + 
"""

---
""")

st.markdown(
"""

### 📂 App Structure
- **Set API Keys**: Set your OpenAI API key or Google API key to run this project.
- **Vector Store**: Build a vector store with configurable hyperparameters for downstream retrieval.
- **Mortality Data**: Upload an annual mortality dataset that will be used in the Pandemic Mortality Section from the report using the vector store. 
- **Model Configuration**: Configure the model and set hyperparameters that shape the generation process.
- **Generate Sections**: After configuring the model, generate the mortality analysis section.

> ⚠️ Tip: Try regenerating the Pandemic Mortality Section after changing the model settings to see how the output changes.
""")
