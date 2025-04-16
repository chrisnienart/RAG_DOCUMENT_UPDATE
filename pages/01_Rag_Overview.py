import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="RPEC 2024 Mortality Dashboard",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None 
)

st.title("📘 RPEC 2024 Mortality Dashboard")

st.markdown("""
Welcome to the **RPEC 2024 Mortality Dashboard**, a dynamic tool powered by AI to help actuaries and analysts generate narrative sections of the SOA's Retirement Plans Experience Committee (RPEC) report.

This application uses **RAG (Retrieval-Augmented Generation)** techniques with **LangChain** and **OpenAI** models to synthesize structured insights from:
- Prior RPEC PDF reports (embedded in a vector store)
- A synthetic dataset of mortality metrics for 2024

---

### 🧠 What is RAG (Retrieval-Augmented Generation)?
RAG enhances Large Language Models (LLMs) by allowing them to retrieve relevant knowledge from external sources before generating responses. This is critical for:
- Citing accurate information from proprietary or domain-specific documents
- Generating structured outputs that reference embedded content (like figures or tables)

We use **LangChain**, a powerful Python framework for building RAG workflows. It connects document loaders, embeddings, retrievers, LLMs, and outputs.

---

### 🔧 Hyperparameters You Can Control
Below is a quick summary of the key RAG components you can tune throughout the app:

#### **RAG Builder (Vector Store Tab)**
| Hyperparameter       | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `chunk_size`         | How large each document chunk is before embedding. Affects recall vs. precision. |
| `chunk_overlap`      | Overlap between chunks to preserve context continuity.                        |
| `embedding_model`    | Which model to use for creating document embeddings (`text-embedding-3-small` vs. `large`). |
| `vectorstore_name`   | Name for storing the built vector store (can swap different versions later).  |

#### **Section Generator (Generate Section Tab)**
| Hyperparameter       | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `k` (top-k retrieval)| Number of relevant chunks to pull from the vector store (higher = more context). |
| `llm_model_name`     | Which GPT model to use for generation (`gpt-4-turbo`, `gpt-3.5-turbo`, etc.). |
| `temperature`        | Controls randomness of output (lower = more factual, higher = more diverse). |
| `prompt_template`    | The system prompt structure (currently fixed for consistent tone).           |

---

### 📂 App Structure
- **Set API Keys**: Set your OpenAI API key or Google API key to run this project.
- **Vector Store**: Build a vector store with configurable hyperparameters for downstream retrieval.
- **Mortality Data**: Upload a 2024 mortality dataset that will be used in Section 3.1 from the report using the vector store. 
- **Model Configuration**: Configure the model and set hyperparameters that shape the generation process.
- **Generate Section 3.1**: After configuring the model, generate the mortality analysis section.

> ⚠️ Tip: Try regenerating Section 3.1 after changing the model settings to see how the output changes.
""")
