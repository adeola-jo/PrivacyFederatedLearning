
#!/bin/bash
# Set Python path to include project root
export PYTHONPATH=$PYTHONPATH:$(pwd)
streamlit run src/ui/app.py --server.headless=true --server.address=0.0.0.0 --server.port=5000
