# RPEC Mortality Analysis Application

## Project Description

This is a Streamlit-based application designed for analyzing and generating sections of the Society of Actuaries' (SOA) Retirement Plans Experience Committee (RPEC) reports. It leverages Retrieval-Augmented Generation (RAG) techniques with LangChain and various AI models (such as OpenAI or Google) to process mortality data from PDFs and datasets. The app allows users to upload documents, configure hyperparameters, and generate narrative reports on topics like mortality trends during events such as the COVID-19 pandemic.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.

## Installation

This project uses `uv` for faster and more reliable Python package management.

### Quick Install

```bash
# Install uv globally
pip install uv

# Install dependencies using uv (from project root)
uv pip install .  # Base dependencies only
```

### Development Installation

For development, including test dependencies:

```bash
# Install with dev dependencies
uv pip install .[dev]
```

### Using Virtual Environment (Recommended)

```bash
# Create and activate virtual environment
uv venv  # Creates .venv by default

# Activate the environment
source .venv/bin/activate  # Linux/MacOS
# OR
.venv\Scripts\activate  # Windows

# Install dependencies in virtual environment
uv pip install .[dev]
```

### Local Install

If `uv` cannot be installed globally or you prefer to install locally, you can install in the project directory.

```bash
# Windows
python -m venv venv
venv\Scripts\activate
pip install uv
uv pip install .[dev]
```

```bash
# Linux/MacOS
python -m venv venv
source venv/bin/activate
pip install uv
uv pip install .[dev]
```

## Running the App

To run the app, run the following command:

```bash
streamlit run app.py
```

This will start the app in your default browser.
