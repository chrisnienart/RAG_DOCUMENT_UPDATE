import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="Upload Dataset - RPEC", layout="wide")
st.title("📤 Upload Mortality Dataset")
st.markdown("Upload the dataset needed for Section 3.1 generation")

# File upload
uploaded_file = st.file_uploader("📂 Upload Updated Mortality Dataset (CSV)", type="csv")

# Check whether the dataset is already uploaded in the session state.
if 'mortality_data' in st.session_state and not uploaded_file:
    st.success("✅ Dataset already uploaded.")
elif uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.mortality_data = df  # Store in session state
        st.success("✅ File uploaded and read successfully")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        st.stop()
else:
    st.info("⬆️ Please upload a synthetic mortality dataset to continue.")

# Navigation button
if 'mortality_data' in st.session_state:
    st.divider()
    st.page_link(
        "pages/Model_Config.py", 
        label="Continue to Model Configuration →",
        icon="⚙️",
        use_container_width=True
    )