import streamlit as st

# Page config
st.set_page_config(page_title="Model Config - RPEC", layout="wide")
st.title("⚙️ Model Configuration")
st.markdown("Configure the model and hyperparameters for Section 3.1 generation")

# Embedding model check
if 'embedding_model' not in st.session_state:
    st.error("⚠️ No embedding model found! Please build a vector store first.")
    st.stop()
    
# Dataset check
if 'mortality_data' not in st.session_state:
    st.error("⚠️ No dataset found! Please upload your mortality data first.")
    st.stop()

# Model parameters
st.markdown("### 🧩 Model Settings")
st.session_state['model_k'] = st.slider("🔍 Top K Chunks to Retrieve", 5, 50, 20)
st.session_state['model_name'] = st.selectbox("🧠 LLM Model", ["gpt-4.1-mini", "gpt-4-turbo", "gpt-3.5-turbo","gemini-1.5-pro"])
st.session_state['temperature'] = st.slider("🌡️ Temperature (Creativity)", 0.0, 1.0, 0.2)

# Navigation button
st.divider()
st.page_link(
    "pages/Generate_Sections.py", 
    label="Continue to Section Generation →",
    icon="🚀",
    use_container_width=True
)