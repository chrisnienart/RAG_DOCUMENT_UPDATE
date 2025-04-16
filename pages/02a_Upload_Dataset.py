import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()
openai_api_key = st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY"))

# Page config
st.set_page_config(page_title="Upload Dataset - RPEC", layout="wide")
st.title("📤 Upload Mortality Dataset")
st.markdown("Upload the dataset needed for Section 3.1 generation")

# File upload
uploaded_file = st.file_uploader("📂 Upload Updated Mortality Dataset (CSV)", type="csv")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.mortality_data = df  # Store in session state
        st.success("✅ File uploaded and read successfully")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        st.stop()
else:
    st.info("⬆️ Please upload a synthetic mortality dataset to continue.")
