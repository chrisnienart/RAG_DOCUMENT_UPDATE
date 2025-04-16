import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv


# Page config
st.set_page_config(page_title="Model Config - RPEC", layout="wide")
st.title("⚙️ Model Configuration")
st.markdown("Configure the model and hyperparameters for Section 3.1 generation")

# Load environment and checks
load_dotenv()
openai_api_key = st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
if not openai_api_key:
    st.error("⚠️ No OpenAI API key found! Please set your API key in the API Keys page.")
    st.stop()

# Embedding model load
try:
    with open("vector_store/embedding_model.txt", "r") as f:
        st.session_state['embedding_model'] = f.read().strip()
except Exception as e:
    st.error("❌ Failed to load embedding model name.")
    st.stop()
    
# Dataset check
if 'mortality_data' not in st.session_state:
    st.error("⚠️ No dataset found! Please upload your mortality data first.")
    st.stop()

# Model parameters
st.markdown("### 🧩 Model Settings")
st.session_state['model_k'] = st.slider("🔍 Top K Chunks to Retrieve", 5, 50, 20)
st.session_state['model_name'] = st.selectbox("🧠 LLM Model", ["gpt-4-turbo", "gpt-3.5-turbo"])
st.session_state['temperature'] = st.slider("🌡️ Temperature (Creativity)", 0.0, 1.0, 0.2)

# Navigation button
st.divider()
st.page_link(
    "pages/02_Generate_Sections.py", 
    label="Continue to Section Generation →",
    icon="🚀",
    use_container_width=True
)