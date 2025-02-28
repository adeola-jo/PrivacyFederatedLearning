
#!/usr/bin/env python3
"""
Main entry point for the Privacy-Preserving Federated Learning Framework.
This script launches the Streamlit web application.
"""

import streamlit as st
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Run the Streamlit app directly
if __name__ == "__main__":
    import subprocess
    subprocess.run(["streamlit", "run", "src/ui/app.py", "--server.headless=true", "--server.address=0.0.0.0", "--server.port=5000"])
