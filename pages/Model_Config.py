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
        - GPT-4.1-Mini: OpenAI's low-cost and low latency model with comparable performance to GPT-4o
        - GPT-4o: OpenAI's flagship model with high performance and expanded capabilities
        - GPT-4-Turbo: OpenAI's older generation model with high performance but high cost
        - Gemini-2.5-Pro: Google's flagship "thinking model" with state of the art intelligence
        - Gemini-2.5-Flash: A variant of Gemini-2.5-Pro with faster inference and lower cost
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
    st.markdown(
        """
        **👉  Default Recommendation**
        - model: GPT-4.1-Mini
        - top k chunks = 20 
        - temperature = 0.2
        """
    )
    
# Model parameters
st.markdown("### 🧩 Model Settings")

# Available models list
MODELS = ["gpt-4.1-mini", "gpt-4o", "gpt-4-turbo", "gemini-2.5-pro-preview-03-25", "gemini-2.5-flash-preview-04-17"]

# Initialize model parameters with defaults if not set
if 'model_name' not in st.session_state:
    st.session_state.model_name = MODELS[0]  # First model is default
if 'model_k' not in st.session_state:
    st.session_state.model_k = 20
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.2

# Create widgets with current session state values
st.session_state['model_name'] = st.selectbox(
    "🧠 LLM Model", 
    MODELS,
    index=MODELS.index(st.session_state.model_name)
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
