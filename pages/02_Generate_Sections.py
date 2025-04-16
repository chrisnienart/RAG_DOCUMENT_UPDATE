# pages/02_Generate_Section.py
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import traceback
import matplotlib.pyplot as plt
import re
import tabulate

# Load environment and checks
load_dotenv()
openai_api_key = st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
if not openai_api_key:
    st.error("⚠️ No OpenAI API key found! Please set your API key in the API Keys page.")
    st.stop()

# Page config
st.set_page_config(page_title="Generate Section - RPEC", layout="wide")
st.title("🚀 Generate Section 3.1")
st.markdown("Configure model settings and generate the mortality analysis section")

# Dataset check
if 'mortality_data' not in st.session_state:
    st.error("⚠️ No dataset found! Please upload your data first.")
    st.stop()

# Model Configuration Section
st.header("⚙️ Model Settings")
with st.container(border=True):
    st.markdown("### 🧩 Configuration Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state['model_k'] = st.slider("🔍 Top K Chunks to Retrieve", 5, 50, 20)
        st.session_state['temperature'] = st.slider("🌡️ Temperature (Creativity)", 0.0, 1.0, 0.2)
    
    with col2:
        st.session_state['model_name'] = st.selectbox("🧠 LLM Model", ["gpt-4-turbo", "gpt-3.5-turbo"])
        try:
            with open("vector_store/embedding_model.txt", "r") as f:
                st.session_state['embedding_model'] = f.read().strip()
            st.markdown(f"**Embedding Model:** `{st.session_state['embedding_model']}`")
        except Exception as e:
            st.error("❌ Failed to load embedding model name.")
            st.stop()

# Generation Section
st.header("📄 Report Generation")
try:
    # Get configured values
    k = st.session_state['model_k']
    model_name = st.session_state['model_name']
    temperature = st.session_state['temperature']
    embedding_model = st.session_state['embedding_model']

    # Load vector store
    embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=openai_api_key)
    vectorstore = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Initialize LLM components
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a mortality analyst preparing the SOA RPEC 2024 Report. "
            "Use the provided context and dataset to generate a response in the same style and structure "
            "as Section 3 of the 2023 RPEC Report. Focus only on updated data from the uploaded dataset.\n\n"
            "Use the markdown format from the 2023 report to format tables and figures. If 'Figure X.X' is referenced, "
            "generate a chart using the dataset and render it inline below that reference. Graphs should appear immediately below reference.\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}"
        )
    )

    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=openai_api_key
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt_template},
        input_key="question"
    )

    # Generation button and logic
    if st.button("🚀 Generate Section 3.1", type="primary", use_container_width=True):
        df = st.session_state.mortality_data
        dataset_summary = df.head(20).to_markdown(index=False)
        query = f"""
Using the uploaded dataset, write Section 3 of the RPEC 2024 report. Place the figure and table at the appropriate location within the narrative.

Here is the dataset sample:
{dataset_summary}
"""
        result = qa_chain({"question": query})
        section_text = result["result"]

        # Figure rendering logic
        pattern = r"(Figure\s+(\d+\.\d+)\s*[:\-–—]\s*(.*?)(\n|$))"
        match = re.search(pattern, section_text, re.IGNORECASE)

        st.subheader("Generated Content")
        if match:
            fig_full, fig_id, fig_desc, _ = match.groups()
            pre_fig = section_text[:match.end()]
            post_fig = section_text[match.end():]

            st.markdown(pre_fig)
            try:
                fig_df = pd.read_csv("data/FIG_3_1.csv")
                fig, ax = plt.subplots()
                ax.plot(fig_df["Year"], fig_df["Rate"], marker="o")
                ax.set_xlabel("Year")
                ax.set_ylabel("Rate per 100,000")
                ax.set_ylim(fig_df["Rate"].min() * 0.95, fig_df["Rate"].max() * 1.05)
                ax.grid(True)
                st.markdown(f"#### 📊 Figure {fig_id} – {fig_desc.strip()}")
                st.pyplot(fig)
            except Exception as fig_err:
                st.warning(f"⚠️ Unable to render Figure {fig_id}: {fig_err}")

            st.markdown(post_fig)
        else:
            st.markdown(section_text)

except Exception as e:
    st.error(f"❌ Initialization error: {e}")
    st.text(traceback.format_exc())
