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

# Load environment and checks
load_dotenv()
openai_api_key = st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
if not openai_api_key:
    st.error("⚠️ No OpenAI API key found! Please set your API key in the API Keys page.")
    st.stop()

# Page config
st.set_page_config(page_title="Generate Section - RPEC", layout="wide")
st.title("🚀 Generate Section 3.1")
st.markdown("Generate the mortality analysis section using configured parameters")

# Check for config
required_keys = ['model_k', 'model_name', 'temperature', 'embedding_model']
if any(key not in st.session_state for key in required_keys):
    st.error("⚠️ Model not configured! Please set parameters first.")
    st.stop()

# Get configured values
k = st.session_state['model_k']
model_name = st.session_state['model_name']
temperature = st.session_state['temperature']
embedding_model = st.session_state['embedding_model']

# Get data from session state
if 'mortality_data' not in st.session_state:
    st.error("⚠️ No dataset found! Please upload your data first.")
    st.stop()

df = st.session_state.mortality_data

    try:
        # Preloaded vector store
        embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=openai_api_key)
        vectorstore = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)

        retriever = vectorstore.as_retriever(search_kwargs={"k": k})

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
                st.error(f"❌ Failed to process file: {e}")
                st.text(traceback.format_exc())

    except Exception as e:
        st.error(f"❌ Failed to initialize GPT or RAG: {e}")
        st.text(traceback.format_exc())

else:
    st.info("⬆️ Please upload a synthetic mortality dataset to begin.")
