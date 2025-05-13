import streamlit as st
import os
from dotenv import load_dotenv
import uuid
import re
import json
import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
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
chunk_size = st.number_input(
    "Chunk Size",
    value=st.session_state.get('chunk_size', 1000),
    min_value=100,
    max_value=5000,
    step=100
)
chunk_overlap = st.number_input(
    "Chunk Overlap",
    value=st.session_state.get('chunk_overlap', 150),
    min_value=0,
    max_value=500,
    step=50
)
splitter_type = st.selectbox(
    "Text Splitter Type",
    ["Sentence-aware (NLTK)", "Character-based (Recursive)"],
    index=0 if st.session_state.get('splitter_type', "Sentence-aware (NLTK)") == "Sentence-aware (NLTK)" else 1
)

# --- Embedding Source + Model Selection ---
embedding_source = st.selectbox(
    "Embedding Source",
    ["OpenAI", "Hugging Face (local)"],
    index=0 if st.session_state.get('embedding_source', "OpenAI") == "OpenAI" else 1
)

if embedding_source == "OpenAI":
    embedding_model = st.selectbox(
        "Embedding Model",
        ["text-embedding-3-small", "text-embedding-3-large"],
        index=0 if st.session_state.get('embedding_model', "text-embedding-3-small") == "text-embedding-3-small" else 1
    )
elif embedding_source == "Hugging Face (local)":
    embedding_model = st.selectbox(
        "Embedding Model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-MiniLM-L12-v2"
        ],
        index=0 if st.session_state.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2") == "sentence-transformers/all-MiniLM-L6-v2" else 1
    )
st.session_state['embedding_model'] = embedding_model

# --- Force Recreate Option ---
force_recreate = st.checkbox(
    "Force recreate collection (required when changing embedding models)",
    value=st.session_state.get('force_recreate', False),
    help="Essential when switching embedding models with different dimensional outputs"
)


# --- Output names ---
collection_base_name = "rpec"
store_path = "vector_store"
st.session_state['store_path'] = store_path

# --- Generate Unique ID ---
unique_id = uuid.uuid4().hex[:6]

if st.button("🚀 Rebuild Vector Store" if 'vector_store_exists' in st.session_state else "🚀 Build Vector Store"):
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
            
            # # Close any existing client before creating new one
            # if 'qdrant_client' in st.session_state:
            #     try:
            #         st.session_state.qdrant_client.close()
            #     except Exception:
            #         pass  # Ignore errors if already closed
            
            # # Initialize fresh Qdrant client
            # st.session_state.qdrant_client = QdrantClient(
            #     path="vector_store",
            #     prefer_grpc=True
            # )

            # Initialize Qdrant client
            if 'qdrant_client' not in st.session_state:
                st.session_state.qdrant_client = QdrantClient(
                    path="vector_store",
                    prefer_grpc=True
                )
            
            # Get vector dimensions from actual embeddings
            test_embedding = embeddings.embed_query("test")
            vector_dim = len(test_embedding)

            # Generate collection name with known dimensions
            collection_name = f"{collection_base_name}_{embedding_source.lower().replace(' ', '_')}_{vector_dim}_{unique_id}"

            # Save metadata to disk
            os.makedirs(store_path, exist_ok=True)
            with open(os.path.join(store_path, f"{collection_name}_document_metadata.json"), "w") as f:
                json.dump(metadata_log, f, indent=2)

            # Check existing collection and dimensions
            try:
                existing_collection = st.session_state.qdrant_client.get_collection(collection_name)
                
                # Dimension check logic
                if existing_collection.config.params.vectors.size != vector_dim and not force_recreate:
                    st.error(f"""Dimension mismatch! Existing: {existing_collection.config.params.vectors.size}D, """
                             f"""Current: {vector_dim}D. Enable 'Force recreate collection' to resolve.""")
                    st.stop()
                    
            except Exception:
                pass  # Collection doesn't exist yet

            # Force delete if requested
            if force_recreate:
                try:
                    st.session_state.qdrant_client.delete_collection(collection_name)
                    st.session_state.force_recreate = False  # Reset after use
                except Exception:
                    pass

            # Create/recreate collection with explicit parameters
            st.session_state.qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_dim,
                    distance=Distance.COSINE
                )
            )

            # Build the vector store
            st.info(f"📊 Creating new collection with {vector_dim}-dimensional vectors")
            vectorstore = QdrantVectorStore(
                client=st.session_state.qdrant_client,
                collection_name=collection_name,
                embedding=embeddings,
            )
            vectorstore.add_documents(all_chunks)
            st.session_state.vector_store_exists = True
            st.session_state['qdrant_collection'] = collection_name  # Store collection name

            # ✅ Phase 2 Validation: Compare vector count
            stored_vectors = st.session_state.qdrant_client.count(collection_name=collection_name).count
            if stored_vectors != len(all_chunks):
                st.warning(f"⚠️ Mismatch: {len(all_chunks)} chunks processed but Qdrant has {stored_vectors} vectors stored.")
            else:
                st.success(f"✅ Verified: {stored_vectors} vectors stored in Qdrant — all chunks embedded successfully.")

            # Save parameters to session state
            st.session_state['chunk_size'] = chunk_size
            st.session_state['chunk_overlap'] = chunk_overlap
            st.session_state['splitter_type'] = splitter_type
            st.session_state['embedding_source'] = embedding_source
            st.session_state['embedding_model'] = embedding_model

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
