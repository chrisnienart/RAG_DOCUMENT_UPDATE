import streamlit as st
import os
from dotenv import load_dotenv
import uuid
import re
import json
import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
# import nltk
import traceback

# # Ensure NLTK tokenizer is available
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     with st.spinner("📥 Downloading NLTK tokenizers (first-time setup)..."):
#         nltk.download('punkt')
#     try:
#         nltk.data.find('tokenizers/punkt')
#     except LookupError:
#         st.error("❌ NLTK tokenizer download failed! Check:")
#         st.markdown("- Internet connection\n- Proxy/firewall settings\n- Disk permissions")
#         st.stop()

# Load environment
load_dotenv()
openai_api_key = st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
if not openai_api_key:
    st.error("⚠️ No OpenAI API key found! Please set your API key in the API Keys page.")
    st.stop()

# Page config
st.set_page_config(page_title="RAG Builder - RPEC Vector Store", layout="wide")
st.title("🛠️ RAG Builder for RPEC Reports")
st.markdown("Build a vector store with configurable hyperparameters for downstream retrieval.")

# --- File Uploader ---
uploaded_files = st.file_uploader("Upload source PDFs (e.g., RPEC 2022/2023 Reports)", type="pdf", accept_multiple_files=True)
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

# --- Previous Uploads Info ---
if 'vector_store_exists' in st.session_state:
    st.success("📁 Previously uploaded files")
elif 'uploaded_files' in st.session_state:
    st.info("Set hyperparameters and click 'Build Vector Store'.")
else:
    st.info("Please upload at least one PDF document.")

# --- 🧠 RAG Tuning Tips Helper ---
with st.expander("🧠 How to choose Chunk Size, Overlap, and Splitter Type?", expanded=False):
    st.markdown("""
**📏 Chunk Size**
- Typical range: `500 – 1000 characters`
- Smaller = precise retrieval, more chunks
- Larger = fewer chunks, better context, risk of drift

**🔁 Chunk Overlap**
- Use ~15–20% of chunk size (e.g., `150` if chunk is `800`)
- Helps preserve context between chunks

**✂️ Splitter Type**
- **Sentence-aware (NLTK)**: Preserves sentence meaning. Great for reports or dense content.
- **Character-based (Recursive)**: Fast fallback. Good for unstructured data.

👉 Default recommendation:
- Chunk Size: `800`
- Overlap: `150`
- Splitter: `Sentence-aware`
""")

# --- Hyperparameter Inputs ---
chunk_size = st.number_input("Chunk Size", value=1000, min_value=100, max_value=5000, step=100)
chunk_overlap = st.number_input("Chunk Overlap", value=150, min_value=0, max_value=500, step=50)
splitter_type = st.selectbox("Text Splitter Type", ["Sentence-aware (NLTK)", "Character-based (Recursive)"])

# --- Embedding Source + Model Selection ---
embedding_source = st.selectbox("Embedding Source", ["OpenAI", "Hugging Face (local)"])

if embedding_source == "OpenAI":
    embedding_model = st.selectbox("Embedding Model", ["text-embedding-3-small", "text-embedding-3-large"])
elif embedding_source == "Hugging Face (local)":
    embedding_model = st.selectbox(
        "Embedding Model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-MiniLM-L12-v2"
        ]
    )
st.session_state['embedding_model'] = embedding_model

# --- Auto Vector Dimension Detection ---
if "text-embedding-3-small" in embedding_model:
    vector_dim = 1536
elif "text-embedding-3-large" in embedding_model:
    vector_dim = 3072
elif "all-MiniLM-L6" in embedding_model:
    vector_dim = 384
elif "paraphrase-MiniLM-L12" in embedding_model:
    vector_dim = 768
else:
    vector_dim = 768  # fallback

# --- Output names ---
collection_base_name = "rpec"
store_path = "vector_store"
st.session_state['store_path'] = store_path

# --- Generate Unique Collection Name ---
unique_id = uuid.uuid4().hex[:6]
collection_name = f"{collection_base_name}_{embedding_source.lower().replace(' ', '_')}_{vector_dim}_{unique_id}"


if st.button("🚀 Build Vector Store"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF document.")
    else:
        try:
            # Text splitter logic
            if splitter_type == "Sentence-aware (NLTK)":
                text_splitter = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            all_chunks = []
            metadata_log = []
            file_stats = []

            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                file_path = os.path.join("temp_docs", filename)
                os.makedirs("temp_docs", exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                loader = PyMuPDFLoader(file_path)
                docs = loader.load()

                for doc in docs:
                    doc.metadata.update({
                        "source": filename,
                        "doc_type": "mortality_research",
                        "committee": "RPEC",
                        "organization": "Society of Actuaries Research Institute",
                        "topic": "mortality_improvement",
                        "region": "US",
                        "audience": "retirement_programs",
                        "doc_purpose": "assumption_development",
                        "source_type": "research_report"
                    })

                    year_match = re.search(r"(20[1-2][0-9])", filename)
                    if year_match:
                        doc.metadata["year"] = int(year_match.group(1))

                chunks = text_splitter.split_documents(docs)
                chunk_count = len(chunks)
                all_chunks.extend(chunks)

                file_stats.append({
                    "Filename": filename,
                    "Pages": len(docs),
                    "Chunks": chunk_count
                })

                st.info(f"📄 {filename} → {chunk_count} chunks")

                if chunk_count == 0:
                    st.warning(f"⚠️ No chunks were generated for {filename}. This file may be empty or improperly formatted.")

                metadata_log.extend([doc.metadata for doc in docs])

            # 📊 Chunk summary table
            if file_stats:
                chunk_df = pd.DataFrame(file_stats)
                st.markdown("### 📊 Chunk Generation Summary")
                st.dataframe(chunk_df)

            # Save metadata to disk
            os.makedirs(store_path, exist_ok=True)
            with open(os.path.join(store_path, f"{collection_name}_document_metadata.json"), "w") as f:
                json.dump(metadata_log, f, indent=2)

            # Embedding model init
            if embedding_source == "OpenAI":
                embeddings = OpenAIEmbeddings(
                    model=embedding_model,
                    openai_api_key=openai_api_key
                )
            elif embedding_source == "Hugging Face (local)":
                embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model
                )

            # Ensure vector store directory exists
            os.makedirs("vector_store", exist_ok=True)
            
            # Initialize Qdrant with persistent storage
            qdrant_client = QdrantClient(
                path="vector_store",
                prefer_grpc=True
            )

            # Create new unique collection
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )

            # Build the vector store
            vectorstore = Qdrant(
                client=qdrant_client,
                collection_name=collection_name,
                embeddings=embeddings,
            )
            vectorstore.add_documents(all_chunks)
            st.session_state.vector_store_exists = True
            st.session_state['qdrant_collection'] = collection_name  # Store collection name

            # ✅ Phase 2 Validation: Compare vector count
            stored_vectors = qdrant_client.count(collection_name=collection_name).count
            if stored_vectors != len(all_chunks):
                st.warning(f"⚠️ Mismatch: {len(all_chunks)} chunks processed but Qdrant has {stored_vectors} vectors stored.")
            else:
                st.success(f"✅ Verified: {stored_vectors} vectors stored in Qdrant — all chunks embedded successfully.")

            # Save embedding metadata
            with open(os.path.join(store_path, f"{collection_name}_embedding_model.txt"), "w") as f:
                f.write(f"{embedding_source} | {embedding_model} | dim: {vector_dim}")

            # Summary of uploads
            st.info(f"""
📚 **Upload Summary**
- Total files uploaded: {len(file_stats)}
- Total chunks created: {len(all_chunks)}
- Vector store collection: '{collection_name}'
- Embedding: {embedding_source} | {embedding_model} | Dim: {vector_dim}
""")

        except Exception as e:
            st.error(f"❌ Failed to build vector store: {e}")
            st.text(traceback.format_exc())

# Navigation button
if 'vector_store_exists' in st.session_state:
    st.divider()
    st.page_link(
        "pages/Upload_Dataset.py", 
        label="Continue to Mortality Data Upload →",
        icon="📤",
        use_container_width=True
    )
