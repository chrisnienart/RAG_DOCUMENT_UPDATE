"""
Utility functions for the RPEC RAG application.
"""
import streamlit as st
import os

@st.cache_resource
def download_nltk_data():
    """
    Download required NLTK data for NLTKTextSplitter.
    Uses @st.cache_resource to only download once per session.
    """
    import nltk
    
    try:
        # Check if punkt tokenizer is available
        nltk.data.find('tokenizers/punkt')
        return True
    except LookupError:
        try:
            # Download required NLTK data
            with st.spinner("Downloading NLTK data for sentence tokenization..."):
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)  # For newer NLTK versions
            return True
        except Exception as e:
            st.error(f"Failed to download NLTK data: {e}")
            return False

def initialize_nltk():
    """
    Initialize NLTK data downloads.
    Returns True if successful, False otherwise.
    """
    return download_nltk_data()