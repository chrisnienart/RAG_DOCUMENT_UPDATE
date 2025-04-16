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

# Initialize Google API key in session state
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY", "")

st.markdown("### API Key Configuration")

# OpenAI Row
with st.form("openai_key_form"):
    cols = st.columns([4, 1])
    with cols[0]:
        new_openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Get your key from https://platform.openai.com/account/api-keys",
            key="openai_input"
        )
    with cols[1]:
        submitted_openai = st.form_submit_button("Update OpenAI Key")

# Google Row
with st.form("google_key_form"):
    cols = st.columns([4, 1])
    with cols[0]:
        new_google_key = st.text_input(
            "Google API Key",
            type="password",
            placeholder="AIza...",
            help="Get your key from https://console.cloud.google.com/apis/credentials",
            key="google_input"
        )
    with cols[1]:
        submitted_google = st.form_submit_button("Update Google Key")

# Handle OpenAI submissions
if submitted_openai:
    if new_openai_key:
        st.session_state.openai_api_key = new_openai_key
        st.success("✅ OpenAI key updated!")
    else:
        st.session_state.openai_api_key = ""
        st.error("❌ OpenAI key cleared")

# Handle Google submissions
if submitted_google:
    if new_google_key:
        st.session_state.google_api_key = new_google_key
        st.success("✅ Google key updated!")
    else:
        st.session_state.google_api_key = ""
        st.error("❌ Google key cleared")

# Display current key status
st.markdown("### Current Key Status")
status_cols = st.columns(2)
with status_cols[0]:
    if st.session_state.openai_api_key:
        st.info("🔐 OpenAI: Key set")
    else:
        st.error("❌ OpenAI: No key set")
with status_cols[1]:
    if st.session_state.google_api_key:
        st.info("🔐 Google: Key set")
    else:
        st.error("❌ Google: No key set")
