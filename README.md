# RPEC Mortality Analysis Application

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

## Running the App

To run the app, run the following command:

```bash
streamlit run app.py
```

This will start the app in your default browser.
