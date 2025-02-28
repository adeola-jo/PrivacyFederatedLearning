#!/usr/bin/env python3
"""
Main entry point for the Privacy-Preserving Federated Learning Framework.
This script launches the Streamlit web application.
"""

if __name__ == "__main__":
    import subprocess
    subprocess.run(["streamlit", "run", "src/ui/app.py", "--server.headless=true", 
                    "--server.address=0.0.0.0", "--server.port=5000"])