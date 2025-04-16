import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="API Key Management", layout="wide")
st.title("🔑 API Key Management")

# Initialize session state for API keys if not exists
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")

# API Key input section
st.markdown("### OpenAI API Key Configuration")
st.markdown("""
Enter your OpenAI API key below. The key will be stored in your session state
and used for all API calls while the app is running.
""")

with st.form("api_key_form"):
    # API Key input with placeholder only
    new_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-... (enter new key to update)",
        help="Enter your OpenAI API key here. Get one at https://platform.openai.com/api-keys",
        key="api_key_input"
    )
    
    # Form submit button
    submitted = st.form_submit_button("Update API Key")
    if submitted:
        if new_api_key:
            if new_api_key != st.session_state.openai_api_key:
                st.session_state.openai_api_key = new_api_key
                st.success("✅ API key updated successfully!")
            else:
                st.info("🔑 API key unchanged")
        else:
            st.session_state.openai_api_key = ""
            st.error("❌ API key cleared from session state")

# Display current key status
st.markdown("### Current API Key Status")
if st.session_state.openai_api_key:
    st.info("🔐 OpenAI API key is set in session state")
else:
    st.error("❌ No OpenAI API key is currently set")
