import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import traceback
import matplotlib.pyplot as plt
import re
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Page config
st.set_page_config(page_title="Generate Section - RPEC", layout="wide")
st.title("🚀 Generate Pandemic Mortality Section")

# Load environment and checks
load_dotenv()
openai_api_key = st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
openrouter_api_key = st.session_state.get("openrouter_api_key", os.getenv("OPENROUTER_API_KEY"))
google_api_key = st.session_state.get("google_api_key", os.getenv("GOOGLE_API_KEY"))

if not (openai_api_key or openrouter_api_key or google_api_key):
    st.error("⚠️ No API key found! Please set at least one API key (OpenAI/OpenRouter/Google) in the API Keys page.")
    st.stop()

# Embedding model check
if 'embedding_model' not in st.session_state:
    st.error("⚠️ No embedding model found! Please build a vector store first.")
    st.stop()

# Dataset check
if 'mortality_data' not in st.session_state:
    st.error("⚠️ No dataset found! Please upload your mortality data first.")
    st.stop()

# Generation Section
try:
    # Get configured values
    k = st.session_state['model_k']
    model_name = st.session_state['model_name']
    temperature = st.session_state['temperature']
    embedding_model = st.session_state['embedding_model']
    store_path = st.session_state['store_path']

    # Load vector store
    collection_name = st.session_state.get('qdrant_collection')
    if not collection_name:
        st.error("⚠️ No Qdrant collection found! Please build a vector store first.")
        st.stop()

    if "text-embedding" in embedding_model:
        if not openai_api_key:
            st.error("⚠️ OpenAI API key required for OpenAI embeddings!")
            st.stop()
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key
        )
    else:  # Hugging Face case
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )

    try:
        embeddings.embed_query("test")  # Verify embeddings work
    except Exception as e:
        st.error(f"❌ Failed to initialize embeddings: {e}")
        st.stop()

    qdrant_client = QdrantClient(
        path=store_path,
        prefer_grpc=True
    )

    # Verify collection exists
    try:
        qdrant_client.get_collection(collection_name)
    except Exception:
        st.error(f"❌ Collection '{collection_name}' not found in vector_store directory")
        st.stop()

    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Define prompt templates in a dictionary
    prompt_templates = {
        "openai": (
            "You are a mortality analyst preparing the SOA RPEC 2024 Report. "
            "Use the provided context and dataset to generate a response in the same style and structure "
            "as Section 3 of the 2023 RPEC Report. Focus only on updated data from the uploaded dataset.\n\n"
            "Use the markdown format from the 2023 report to format tables and figures. If 'Figure X.X' is referenced, "
            "generate a chart using the dataset and render it inline below that reference. Graphs should appear immediately below reference."
        ),
        "google": (
            "You are a professional actuarial report writer. Based on the context from previous reports and a provided dataset, "
            "generate a fully written markdown document for the Pandemic Mortality Section of the SOA RPEC 2024 Report. "
            "The tone should be formal, structured, and human-written. DO NOT include Python code or placeholders like '[Insert table]'. "
            "Instead, describe the data as if it were already in the report. "
            "Structure your response exactly like a completed report section."
        ),
        "weak": (
            "Write something about mortality data. "
            "Include some numbers and trends from the dataset."
        ),
        "poor": (
            "Talk about the data. "
            "Mention any interesting patterns you see."
        )
    }

    # Initialize LLM components
    if model_name.startswith("gpt"):
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=openai_api_key
        )
        template_key = "openai"
    else: # Assuming Google model if not OpenAI
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=google_api_key
        )
        template_key = "google"

    # Generation button and logic
    st.markdown(
        "If all prior steps are completed, choose a prompt template and then click the button below to generate the Pandemic Mortality Section of the report."
    )

    with st.expander("🧠 What is a prompt?", expanded=False):
        st.markdown(
            """
            A prompt is a string of text that guides the model to generate a response. 
            It can be a question, a statement, or a combination of both. \n\n 
            The model uses the prompt to understand the context and generate a relevant response.
            """)

    # Create user-friendly names for the prompt templates
    prompt_options = {
        "OpenAI Style": "openai",
        "Google Style": "google",
        "Weak Prompt": "weak",
        "Poor Prompt": "poor"
    }
    
    selected_prompt_name = st.selectbox(
        "Prompt Template",
        options=list(prompt_options.keys())
    )
    
    # Show the selected prompt template with edit option
    selected_template_key = prompt_options[selected_prompt_name]
    
    # Store prompt in session state
    if 'prompt' not in st.session_state:
        st.session_state.prompt = prompt_templates[selected_template_key]
    
    # Update prompt when template changes
    if st.session_state.get('last_template_key') != selected_template_key:
        st.session_state.prompt = prompt_templates[selected_template_key]
        st.session_state.last_template_key = selected_template_key
    
    # Expandable text area for editing the prompt
    with st.expander("✏️ Edit Prompt Template", expanded=True):
        # Calculate height based on number of lines in prompt
        line_count = len(st.session_state.prompt.split('\n'))
        height = min(max(100, (line_count + 3) * 20), 500)  # Between 100-500px based on content
        
        prompt = st.text_area(
            "Edit the prompt template below:",
            value=st.session_state.prompt,
            height=height,
            label_visibility="collapsed"
        )
        
        if prompt != prompt_templates[selected_template_key]:
            st.session_state.prompt = prompt
            st.success("✅ Using edited prompt template")
        else:
            st.info("ℹ️ Using default prompt template")

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=f"{st.session_state.prompt}\n\nContext:\n{{context}}\n\nQuestion:\n{{question}}"
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt_template},
        input_key="question"
    )

    # Save generated content to session state
    if 'generated_content' not in st.session_state:
        st.session_state.generated_content = None

    # Changed button label and moved generation logic
    button_label = "Regenerate Pandemic Mortality Section" if st.session_state.generated_content else "Generate Pandemic Mortality Section"
    
    if st.button(f"🚀 {button_label}"):
        df = st.session_state.mortality_data
        dataset_summary = df.head(20).to_markdown(index=False)
        query = f"""
Using the uploaded dataset, write Section 3 of the RPEC 2024 report in the {selected_prompt_name} style. 
Place the figure and table at the appropriate location within the narrative.

Here is the dataset sample:
{dataset_summary}
"""
        result = qa_chain({"question": query})
        st.session_state.generated_content = result["result"]
        st.rerun()  # Refresh to show updated content

    # Display existing content if available
    if st.session_state.generated_content:
        st.subheader("Generated Content")
        section_text = st.session_state.generated_content
        
        # Existing figure rendering logic
        pattern = r"(Figure\s+(\d+\.\d+)\s*[:\-–—]\s*(.*?)(\n|$))"
        match = re.search(pattern, section_text, re.IGNORECASE)
        
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

        # Keep download button visible
        st.download_button(
            label="💾 Save Generated Content",
            data=st.session_state.generated_content,
            file_name="RPEC_2024_Section3.md",
            mime="text/markdown",
            help="Save the generated content as a Markdown file"
        )

except Exception as e:
    st.error(f"❌ Initialization error: {e}")
    st.text(traceback.format_exc())
