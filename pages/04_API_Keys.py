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
    cols = st.columns([3, 5, 2])
    with cols[0]:
        st.markdown("**OpenAI API Key**  ")
    with cols[1]:
        new_openai_key = st.text_input(
            label=" ",
            type="password",
            placeholder="sk-...",
            key="openai_input",
            label_visibility="collapsed"
        )
    with cols[2]:
        submitted_openai = st.form_submit_button("🔁 Update", use_container_width=True)
    cols[1].caption("Get key: [platform.openai.com](https://platform.openai.com/account/api-keys)")

# Google Row
with st.form("google_key_form"):
    cols = st.columns([3, 5, 2])
    with cols[0]:
        st.markdown("**Google API Key**  ")
    with cols[1]:
        new_google_key = st.text_input(
            label=" ",
            type="password",
            placeholder="AIza...",
            key="google_input",
            label_visibility="collapsed"
        )
    with cols[2]:
        submitted_google = st.form_submit_button("🔁 Update", use_container_width=True)
    cols[1].caption("Get key: [console.cloud.google.com](https://console.cloud.google.com/apis/credentials)")

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
