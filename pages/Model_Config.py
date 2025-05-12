import streamlit as st

# Page config
st.set_page_config(page_title="Model Config - RPEC", layout="wide")
st.title("⚙️ Model Configuration")
st.markdown("Configure the model and hyperparameters for the Pandemic Mortality Section generation")

# Embedding model check
if 'embedding_model' not in st.session_state:
    st.error("⚠️ No embedding model found! Please build a vector store first.")
    st.stop()
    
# Dataset check
if 'mortality_data' not in st.session_state:
    st.error("⚠️ No dataset found! Please upload your mortality data first.")
    st.stop()

with st.expander("🧠 How to pick an LLM and hyper parameters", expanded=False):
    st.markdown(
        """
        **🧠 LLM Model**
        - GPT-4-Turbo: Best for tasks that require long context, but can be expensive
        - GPT-3.5-Turbo: Best for tasks that require long context, but can be expensive
        - GPT-4.1-Mini: Best for tasks that require short context, but can be expensive
        - Gemini-1.5-Pro: Best for tasks that require short context, but can be expensive
        """)
    st.markdown(
        """
        **🔍 Top K Chunks to Retrieve**
        - Higher = more context, but risk of drift
        - Lower = fewer chunks, but better recall
        """)
    st.markdown(
        """
        **🌡️ Temperature (Creativity)**
        - Higher = more randomness, but risk of hallucinations
        - Lower = more factual, but risk of hallucinations
        """)
    
# Model parameters
st.markdown("### 🧩 Model Settings")

# Initialize model parameters with defaults if not set
if 'model_name' not in st.session_state:
    st.session_state.model_name = "gpt-4-turbo"
if 'model_k' not in st.session_state:
    st.session_state.model_k = 20
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.2

# Create widgets with current session state values
st.session_state['model_name'] = st.selectbox(
    "🧠 LLM Model", 
    ["gpt-4.1-mini", "gpt-4-turbo", "gpt-3.5-turbo","gemini-1.5-pro"],
    index=["gpt-4.1-mini", "gpt-4-turbo", "gpt-3.5-turbo","gemini-1.5-pro"].index(st.session_state.model_name)
)
st.session_state['model_k'] = st.slider(
    "🔍 Top K Chunks to Retrieve", 
    5, 50, 
    value=st.session_state.model_k
)
st.session_state['temperature'] = st.slider(
    "🌡️ Temperature (Creativity)", 
    0.0, 1.0, 
    value=st.session_state.temperature
)

# Navigation button
st.divider()
st.page_link(
    "pages/Generate_Sections.py", 
    label="Continue to Section Generation →",
    icon="🚀",
    use_container_width=True
)
