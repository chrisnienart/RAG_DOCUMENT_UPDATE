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
import tabulate  # <-- Added for pandas to_markdown support
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Page config
st.set_page_config(page_title="Generate Section 3.1 - RPEC", layout="wide")
st.title("🧐 Generate Section 3.1 for RPEC 2024")
st.markdown("Upload the dataset and use the latest vector store from the RAG Builder to generate an updated Section 3.1.")

# File upload
df = None
uploaded_file = st.file_uploader("📂 Upload Updated Mortality Dataset (CSV)", type="csv")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded and read successfully")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        st.stop()

# Prompt for model and retrieval params (after upload)
if df is not None:
    st.markdown("### 🧩 Model Configuration")
    k = st.slider("🔍 Top K Chunks to Retrieve", 5, 50, 20)
    model_choice = st.selectbox("🧠 LLM Model", ["gpt-4-turbo", "gpt-3.5-turbo", "gemini-1.5-pro"])
    temperature = st.slider("🌡️ Temperature (Creativity)", 0.0, 1.0, 0.2)

    try:
        with open("vector_store/embedding_model.txt", "r") as f:
            embedding_model = f.read().strip()
    except Exception as e:
        st.error("❌ Failed to load embedding model name.")
        st.stop()

    try:
        # Preloaded vector store
        embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=openai_api_key)
        vectorstore = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)

        retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        if model_choice.startswith("gpt"):
            llm = ChatOpenAI(
                model_name=model_choice,
                temperature=temperature,
                openai_api_key=openai_api_key
            )
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
        else:
            llm = ChatGoogleGenerativeAI(
                model=model_choice,
                temperature=temperature,
                google_api_key=google_api_key
            )
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template=(
                    "You are a professional actuarial report writer. Based on the context from previous reports and a provided dataset, "
                    "generate a fully written markdown document for Section 3.1 of the SOA RPEC 2024 Report. "
                    "The tone should be formal, structured, and human-written. DO NOT include Python code or placeholders like '[Insert table]'. "
                    "Instead, describe the data as if it were already in the report. "
                    "Structure your response exactly like a completed report section.\n\n"
                    "Context:\n{context}\n\nQuestion:\n{question}"
                )
            )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt_template},
            input_key="question"
        )

        if st.button("🚀 Generate Section 3.1"):
            try:
                dataset_summary = df.head(20).to_markdown(index=False)
                query = f"""
Using the uploaded dataset, write Section 3 of the RPEC 2024 report. Place the figure and table at the appropriate location within the narrative.

Here is the dataset sample:
{dataset_summary}
"""
                result = qa_chain({"question": query})

                section_text = result["result"]

                # Detect and split at figure reference
                pattern = r"(Figure\s+(\d+\.\d+)\s*[:\-–—]\s*(.*?)(\n|$))"
                match = re.search(pattern, section_text, re.IGNORECASE)

                st.subheader("📄 Generated Section 3.1")

                if match:
                    fig_full, fig_id, fig_desc, _ = match.groups()
                    pre_fig = section_text[:match.end()]
                    post_fig = section_text[match.end():]

                    st.markdown(pre_fig)
                    try:
                        fig_df = pd.read_csv("FIG_3_1.csv")
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
                st.error(f"❌ Failed to process file: {e}")
                st.text(traceback.format_exc())

    except Exception as e:
        st.error(f"❌ Failed to initialize GPT or RAG: {e}")
        st.text(traceback.format_exc())

else:
    st.info("⬆️ Please upload a synthetic mortality dataset to begin.")
