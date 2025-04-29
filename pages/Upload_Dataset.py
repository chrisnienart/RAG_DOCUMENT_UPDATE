import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="Upload Dataset - RPEC", layout="wide")
st.title("📤 Upload Mortality Dataset")
st.markdown("Upload the dataset needed for Section 3.1 generation")

# User help information
with st.expander("🧠 How to create synthetic data", expanded=False): 
    st.markdown(
        """
        Here are some basic ways you can generate CSV data for a RAG
        (Retrieval Augmented Generation) pipeline.\n\n
        """)
    
    st.markdown(
"""
**1. Manual Creation**

- Text Editor/Spreadsheet Software: The simplest way for very small datasets or 
generating specific test cases. Open a text editor or spreadsheet (like Excel, 
Google Sheets, or LibreOffice Calc) and type in your data following the CSV format:
  - Each row is a record.
  - Values within a row are separated by a delimiter (usually a comma, but can be
    a semicolon or tab).
  - The first row is typically the header row, defining the column names.
- Pros: Easy for small, simple datasets. You have complete control over the data.
- Cons: Impractical for large datasets. Prone to manual errors. Limited scalability.

**2. Copy and Paste**

- Copy and paste data from a website or another source into a text editor or spreadsheet.
- Pros: Easy for small, simple datasets. You have complete control over the data.    
- Cons: Limited scalability. Data may not be in the correct format or structure.

**3. Web Scraping** 
- Use a web scraping tool to extract data from a website and format it
- Libraries like Beautiful Soup or Scrapy: If the data you need is available on websites, 
you can use web scraping libraries to extract it programmatically.
- Structure the Extracted Data: Once you've extracted the data, structure it into a list 
of lists or a list of dictionaries before writing it to a CSV file using the csv module.
- Pros: Can gather large amounts of specific data from the web.
- Cons: Requires more advanced programming skills. Legal and ethical considerations (respecting website terms of service, robots.txt). Websites can change their structure, breaking your scraper.

**4. Generate with Scripts**

- Generating Data from Lists/Dictionaries: If you have data already stored in Python 
lists or dictionaries, you can easily write a script to format it into a CSV.
- Basic Loops and String Formatting: Use loops to iterate through your data and string 
formatting to create each row with commas as separators.
- Use a Python library like `Faker` to generate synthetic data.
- Pros: Good for testing your pipeline's functionality without relying on real data.
- Cons: The data isn't real and may not capture the nuances of your actual data.
""")

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