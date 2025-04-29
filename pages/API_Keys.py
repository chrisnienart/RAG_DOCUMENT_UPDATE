import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="API Key Management", layout="wide")
st.title("🔑 API Key Management")

st.markdown(
    """
    Set your OpenAI API, OpenRouter, Google API key to run this project. 
    You can choose which model to use later on the Generation page.
    """)

# User help information
with st.expander("🧠 What is an API key?", expanded=False): 
    st.markdown(
        """
        An API key is a unique identifier used to authenticate and authorize 
        a user, developer, or calling program to an API. \n\n It acts as a secret 
        token that signifies the connecting API has a defined set of access 
        rights and is necessary for software applications to send and receive 
        data or connect one program to another.
        """)
    
# Initialize session state for API keys if not exists
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY", "")
if "openrouter_api_key" not in st.session_state:
    st.session_state.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
# if "qdrant_api_key" not in st.session_state:
#     st.session_state.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")

# Informational message about setting API keys
st.session_state.api_keys = st.session_state.openai_api_key or st.session_state.google_api_key or st.session_state.openrouter_api_key
if not st.session_state.api_keys:
    st.info("⬆️ Please set at least one API key (OpenAI/OpenRouter/Google) to continue.")

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
            placeholder="sk-proj-...",
            key="openai_input",
            label_visibility="collapsed"
        )
    with cols[2]:
        submitted_openai = st.form_submit_button("🔁 Update", use_container_width=True)
    cols[1].caption("Get key: [platform.openai.com](https://platform.openai.com/account/api-keys)")

# OpenRouter Row
with st.form("openrouter_key_form"):
    cols = st.columns([3, 5, 2])
    with cols[0]:
        st.markdown("**OpenRouter API Key**  ")
    with cols[1]:
        new_openrouter_key = st.text_input(
            label=" ",
            type="password",
            placeholder="sk-or-....",
            key="openrouter_input",
            label_visibility="collapsed"
        )
    with cols[2]:
        submitted_openrouter = st.form_submit_button("🔁 Update", use_container_width=True)
    cols[1].caption("Get key: [openrouter.ai](https://openrouter.ai/settings/keys)")

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

# Handle OpenRouter submissions
if submitted_openrouter:
    if new_openrouter_key:
        st.session_state.openrouter_api_key = new_openrouter_key
        st.success("✅ OpenRouter key updated!")
    else:
        st.session_state.openrouter_api_key = ""
        st.error("❌ OpenRouter key cleared")

# Handle Google submissions
if submitted_google:
    if new_google_key:
        st.session_state.google_api_key = new_google_key
        st.success("✅ Google key updated!")
    else:
        st.session_state.google_api_key = ""
        st.error("❌ Google key cleared")

# # Qdrant Row
# with st.form("qdrant_key_form"):
#     cols = st.columns([3, 5, 2])
#     with cols[0]:
#         st.markdown("**Qdrant API Key**  ")
#     with cols[1]:
#         new_qdrant_key = st.text_input(
#             label=" ",
#             type="password",
#             placeholder="Enter Qdrant cloud key...",
#             key="qdrant_input",
#             label_visibility="collapsed"
#         )
#     with cols[2]:
#         submitted_qdrant = st.form_submit_button("🔁 Update", use_container_width=True)
#     cols[1].caption("Cloud keys: [cloud.qdrant.io](https://cloud.qdrant.io)")

# # Handle Qdrant submissions
# if submitted_qdrant:
#     if new_qdrant_key:
#         st.session_state.qdrant_api_key = new_qdrant_key
#         st.success("✅ Qdrant key updated!")
#     else:
#         st.session_state.qdrant_api_key = ""
#         st.error("❌ Qdrant key cleared")

# Display current key status
st.markdown("### Current Key Status")
# status_cols = st.columns(4)
status_cols = st.columns(3)
with status_cols[0]:
    if st.session_state.openai_api_key:
        st.info("🔐 OpenAI: Key set")
    else:
        st.error("❌ OpenAI: No key set")
with status_cols[1]:
    if st.session_state.openrouter_api_key:
        st.info("🔐 OpenRouter: Key set")
    else:
        st.error("❌ OpenRouter: No key set")
with status_cols[2]:
    if st.session_state.google_api_key:
        st.info("🔐 Google: Key set")
    else:
        st.error("❌ Google: No key set")
# with status_cols[3]:
#     if st.session_state.qdrant_api_key:
#         st.info("🔐 Qdrant: Key set")
#     else:
#         st.warning("⚠️ Qdrant: Using local/no key")


# Navigation button
st.session_state.api_keys = st.session_state.openai_api_key or st.session_state.google_api_key or st.session_state.openrouter_api_key
if st.session_state.api_keys:
    st.divider()
    st.page_link(
        "pages/Vector_Store.py", 
        label="Continue to Vector Store Creation →", 
        icon="🛠️",
        use_container_width=True
    )
