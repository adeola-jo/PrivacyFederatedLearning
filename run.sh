
#!/bin/bash

# Set Python path to include the project root
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the Streamlit app
streamlit run src/ui/app.py --server.headless=true --server.address=0.0.0.0 --server.port=5000
