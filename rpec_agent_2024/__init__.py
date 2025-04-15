import streamlit as st

st.set_page_config(
    page_title="RPEC RAG App Overview", 
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

if page == "RPEC RAG App Overview":
    st.switch_page("pages/00_Rag_App_Overview.py")
elif page == "RAG Builder":
    st.switch_page("pages/01_Rag_Build.py")
elif page == "Generate Section":
    st.switch_page("pages/02_Generate_Sections.py")
