import streamlit as st

def main():
    pages = [
        st.Page("pages/Rag_Overview.py", title = "Overview"),
        st.Page("pages/API_Keys.py", title = "Set API Keys"),
        st.Page("pages/Vector_Store.py", title = "Create Vector Store"),
        st.Page("pages/Upload_Dataset.py", title = "Upload Mortality Data"),
        st.Page("pages/Model_Config.py", title = "Model Configuration"),
        st.Page("pages/Generate_Sections.py", title = "Generate Section 3.1"),
        st.Page("pages/Settings.py", title = "Settings"),
    ]
    pg = st.navigation(pages)
    pg.run()

if __name__ == "__main__":
    main()
