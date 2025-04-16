import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import traceback

# Load environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Page config
st.set_page_config(page_title="RAG Builder - RPEC Vector Store", layout="wide")
st.title("🛠️ RAG Builder for RPEC Reports")
st.markdown("Build a vector store with configurable hyperparameters for downstream retrieval.")

# --- File Uploader ---
uploaded_files = st.file_uploader("Upload source PDFs (e.g., RPEC 2022/2023 Reports)", type="pdf", accept_multiple_files=True)

# --- Hyperparameter Inputs ---
chunk_size = st.number_input("Chunk Size", value=1000, min_value=100, max_value=5000, step=100)
chunk_overlap = st.number_input("Chunk Overlap", value=150, min_value=0, max_value=500, step=50)
embedding_model = st.selectbox("Embedding Model", ["text-embedding-3-small", "text-embedding-3-large"])

# --- Output name ---
store_name = "vector_store"  # Generic name used downstream

if st.button("🚀 Build Vector Store"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF document.")
    else:
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

            all_docs = []
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                file_path = os.path.join("temp_docs", filename)
                os.makedirs("temp_docs", exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                loader = PyMuPDFLoader(file_path)
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = filename

                chunks = text_splitter.split_documents(docs)
                all_docs.extend(chunks)
                st.success(f"✅ Loaded and split {filename} into {len(chunks)} chunks")

            embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=openai_api_key)
            vectorstore = FAISS.from_documents(all_docs, embeddings)
            vectorstore.save_local(store_name)
            st.session_state.vector_store_exists = True

            # Save embedding model metadata
            os.makedirs(store_name, exist_ok=True)
            with open(os.path.join(store_name, "embedding_model.txt"), "w") as f:
                f.write(embedding_model)

            st.success(f"🎉 Vector store saved as '{store_name}' and ready for use in downstream tasks")

        except Exception as e:
            st.error(f"❌ Failed to build vector store: {e}")
            st.text(traceback.format_exc())

# Navigation button
# try:
#     with open("vector_store/embedding_model.txt", "r") as f:
#         st.session_state['embedding_model'] = f.read().strip()
# except Exception as e:
#     st.stop()

# if uploaded_files:
st.divider()
st.page_link(
    "pages/02a_Upload_Dataset.py", 
    label="Continue to Mortality Data Upload →",
    icon="📤",
    use_container_width=True
)
