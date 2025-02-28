
#!/usr/bin/env python3
"""
Main entry point for the Privacy-Preserving Federated Learning Framework.
This script launches the Streamlit web application.
"""

import os
import sys
import streamlit.web.cli as stcli
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Running the app with Streamlit's CLI to properly handle command line args
    sys.argv = ["streamlit", "run", "src/ui/app.py", "--server.headless=true", 
                "--server.address=0.0.0.0", "--server.port=5000"]
    sys.exit(stcli.main())
