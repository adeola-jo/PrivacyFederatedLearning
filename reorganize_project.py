
"""
Script to reorganize project files into a more structured layout.
"""

import os
import shutil
from pathlib import Path

def create_directory(dir_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

def move_file(source, destination):
    """Move a file from source to destination."""
    if os.path.exists(source):
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Move the file
        shutil.move(source, destination)
        print(f"Moved: {source} -> {destination}")
    else:
        print(f"Warning: Source file not found: {source}")

def main():
    # Define the new directory structure
    dirs = [
        'src',
        'src/core',
        'src/data',
        'src/models',
        'src/privacy',
        'src/ui',
        'src/utils',
        'docs',
        'tests',
        'examples',
        'config'
    ]
    
    # Create the directories
    for dir_path in dirs:
        create_directory(dir_path)
    
    # Define file mappings (source -> destination)
    file_mappings = {
        # Core federated learning files
        'federated_learning.py': 'src/core/federated_learning.py',
        'federated_utils.py': 'src/utils/federated_utils.py',
        
        # Data handling
        'data_handler.py': 'src/data/data_handler.py',
        
        # Models
        'model.py': 'src/models/model.py',
        '_model.py': 'src/models/legacy_model.py',
        
        # Privacy
        'differential_privacy.py': 'src/privacy/differential_privacy.py',
        
        # Database
        'database.py': 'src/utils/database.py',
        
        # Visualization
        'visualization.py': 'src/ui/visualization.py',
        
        # UI
        'main.py': 'src/ui/app.py',
        
        # Tests
        'tests/test_db_operations.py': 'tests/test_db_operations.py',
        'tests/test_differential_privacy.py': 'tests/test_differential_privacy.py',
        'tests/test_federated_integration.py': 'tests/test_federated_integration.py',
        'tests/test_new_features.py': 'tests/test_new_features.py',
        'tests/README.md': 'tests/README.md',
        
        # Documentation
        'README.md': 'README.md',
        'TECHNICAL_REPORT.md': 'docs/TECHNICAL_REPORT.md'
    }
    
    # Move files according to the mappings
    for source, destination in file_mappings.items():
        move_file(source, destination)
    
    # Create __init__.py files in all directories
    for dir_path in dirs:
        init_file = os.path.join(dir_path, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f'"""\nInitialization file for {dir_path} package.\n"""\n')
            print(f"Created: {init_file}")
    
    # Create a main.py file in the root directory to run the application
    with open('main.py', 'w') as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
Main entry point for the Privacy-Preserving Federated Learning Framework.
This script launches the Streamlit web application.
\"\"\"

import subprocess
import os
import sys

if __name__ == "__main__":
    # Add the src directory to the Python path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
    
    # Run the Streamlit app
    subprocess.run(["streamlit", "run", "src/ui/app.py"])
""")
    print("Created new main.py launcher")
    
    # Create a requirements.txt file
    with open('requirements.txt', 'w') as f:
        f.write("""# Core dependencies
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
plotly>=5.3.0
streamlit>=1.8.0

# Database
sqlalchemy>=1.4.0

# Testing
pytest>=6.2.5

# Documentation
sphinx>=4.0.0
""")
    print("Created requirements.txt")
    
    print("\nProject reorganization complete!")
    print("\nTo run the application after reorganization:")
    print("python main.py")

if __name__ == "__main__":
    main()
