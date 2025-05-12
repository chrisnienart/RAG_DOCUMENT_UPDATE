import streamlit as st

st.title("Settings")

# Initialize session state for show_api_keys if not exists
if "show_api_keys" not in st.session_state:
    st.session_state.show_api_keys = True

# Toggle for API Keys page visibility
st.toggle(
    "Show API Keys Page in Navigation",
    value=st.session_state.show_api_keys,
    key="show_api_keys_toggle",
    help="Controls whether the API Keys configuration page appears in the app navigation"
)

# Update session state when toggle changes
if "show_api_keys_toggle" in st.session_state:
    st.session_state.show_api_keys = st.session_state.show_api_keys_toggle
