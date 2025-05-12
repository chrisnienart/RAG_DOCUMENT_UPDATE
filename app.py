import streamlit as st

def main():
    # Initialize pages list with always-visible pages
    pages = [
        st.Page("pages/Rag_Overview.py", title = "Overview"),
        st.Page("pages/Vector_Store.py", title = "Create Vector Store"),
        st.Page("pages/Upload_Dataset.py", title = "Upload Mortality Data"),
        st.Page("pages/Model_Config.py", title = "Model Configuration"),
        st.Page("pages/Generate_Sections.py", title = "Generate Sections"),
        st.Page("pages/Settings.py", title = "Settings"),
    ]
    
    # Conditionally add API Keys page if enabled in settings
    if st.session_state.get("show_api_keys", True):
        pages.insert(1, st.Page("pages/API_Keys.py", title = "Set API Keys"))
    pg = st.navigation(pages)
    pg.run()

if __name__ == "__main__":
    main()
