import streamlit as st

st.set_page_config(
    page_title="RPEC RAG App Overview", 
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Always open the overview page by default
st.switch_page("pages/01_RAG_App_Overview.py")
