
#!/usr/bin/env python3
"""
Main entry point for the Privacy-Preserving Federated Learning Framework.
This script launches the Streamlit web application.
"""

import subprocess
import os
import sys

if __name__ == "__main__":
    # Add the src directory to the Python path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
    
    # Run the Streamlit app
    subprocess.run(["streamlit", "run", "src/ui/app.py"])
