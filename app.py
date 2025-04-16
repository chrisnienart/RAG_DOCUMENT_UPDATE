import streamlit as st

pages = [
    st.Page("pages/01_Rag_Overview.py", title = "Overview"),
    st.Page("pages/04_API_Keys.py", title = "Set API Keys"),
    st.Page("pages/03_Rag_Build.py", title = "Create Vector Store"),
    st.Page("pages/02a_Upload_Dataset.py", title = "Upload Mortality Data"),
    st.Page("pages/02a_Model_Config.py", title = "Model Configuration"),
    st.Page("pages/02_Generate_Sections.py", title = "Generate Section 3.1"),
]

pg = st.navigation(pages)
pg.run()