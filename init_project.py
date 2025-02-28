
"""
Script to initialize the reorganized project structure.
This sets up the necessary environment and creates configuration files.
"""

import os
import json
import subprocess
from pathlib import Path

def create_config_files():
    """Create configuration files for the project."""
    
    # Create streamlit config
    os.makedirs('.streamlit', exist_ok=True)
    with open('.streamlit/config.toml', 'w') as f:
        f.write("""[server]
port = 5000
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = true
""")
    
    # Create project config
    os.makedirs('config', exist_ok=True)
    with open('config/default_config.json', 'w') as f:
        config = {
            "federated": {
                "num_clients": 5,
                "client_fraction": 0.6,
                "num_rounds": 10,
                "local_epochs": 2
            },
            "privacy": {
                "enabled": True,
                "noise_scale": 0.1,
                "privacy_budget": 1.0
            },
            "non_iid": {
                "enabled": False,
                "alpha": 0.5
            },
            "compression": {
                "enabled": False,
                "ratio": 0.5
            },
            "training": {
                "batch_size": 64,
                "optimizer": "sgd",
                "learning_rate": 0.01,
                "weight_decay": 1e-4
            },
            "system": {
                "seed": 42,
                "device": "auto",  # "auto", "cpu", or "cuda"
                "verbose": True
            }
        }
        json.dump(config, f, indent=2)
    
    print("Created configuration files")

def create_doc_templates():
    """Create documentation templates."""
    
    os.makedirs('docs/api', exist_ok=True)
    
    # Create a basic Sphinx conf.py
    with open('docs/conf.py', 'w') as f:
        f.write("""# Configuration file for the Sphinx documentation builder.

project = 'Privacy-Preserving Federated Learning'
copyright = '2025'
author = 'PPFL Team'

# The full version, including alpha/beta/rc tags
release = '1.0.0'

# Add any Sphinx extension module names here
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

# Add any paths that contain templates here
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files
html_static_path = ['_static']
""")
    
    # Create index.rst
    with open('docs/index.rst', 'w') as f:
        f.write("""Privacy-Preserving Federated Learning Documentation
================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   usage
   api/index
   examples
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
""")
    
    # Create some basic rst files
    for filename in ['introduction.rst', 'installation.rst', 'usage.rst', 'examples.rst', 'contributing.rst']:
        with open(f'docs/{filename}', 'w') as f:
            title = filename.replace('.rst', '').capitalize()
            f.write(f"""{title}
{'=' * len(title)}

This is a placeholder for the {title} documentation.
""")
    
    # Create API index
    with open('docs/api/index.rst', 'w') as f:
        f.write("""API Reference
============

.. toctree::
   :maxdepth: 2

   core
   data
   models
   privacy
   ui
   utils
""")
    
    # Create API module files
    for module in ['core', 'data', 'models', 'privacy', 'ui', 'utils']:
        with open(f'docs/api/{module}.rst', 'w') as f:
            title = f"{module.capitalize()} Module"
            f.write(f"""{title}
{'=' * len(title)}

.. automodule:: src.{module}
   :members:
   :undoc-members:
   :show-inheritance:

""")
            
            # Add submodule references if applicable
            if os.path.isdir(f'src/{module}'):
                for py_file in os.listdir(f'src/{module}'):
                    if py_file.endswith('.py') and py_file != '__init__.py':
                        submodule = py_file[:-3]
                        f.write(f""".. automodule:: src.{module}.{submodule}
   :members:
   :undoc-members:
   :show-inheritance:

""")
    
    print("Created documentation templates")

def create_examples():
    """Create example notebooks and scripts."""
    
    os.makedirs('examples', exist_ok=True)
    
    # Create a basic example script
    with open('examples/basic_federated_learning.py', 'w') as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
Basic example of using the Privacy-Preserving Federated Learning framework.
This example shows how to set up and run a federated learning experiment
programmatically, without using the Streamlit UI.
\"\"\"

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models.model import SimpleConvNet
from core.federated_learning import FederatedLearning
from data.data_handler import load_mnist_data
import torch

def main():
    # Load data
    print("Loading MNIST dataset...")
    train_data, val_data, test_data = load_mnist_data(iid=True)
    
    # Initialize model
    model = SimpleConvNet()
    
    # Configure federated learning
    config = {
        'privacy': {
            'enabled': True,
            'noise_scale': 0.1,
            'privacy_budget': 1.0
        },
        'compression': {
            'enabled': False,
            'ratio': 0.5
        },
        'non_iid': {
            'enabled': False,
            'alpha': 0.5
        },
        'batch_size': 64,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'verbose': True
    }
    
    # Initialize federated learning system
    fl_system = FederatedLearning(
        model, 
        num_clients=5,
        config=config
    )
    
    # Run federated learning for 5 rounds
    print("Starting federated learning...")
    for round_idx in range(5):
        # Perform one round of federated learning
        round_accuracy, privacy_loss = fl_system.train_round(
            train_data,
            val_data,
            test_data,
            local_epochs=2,
            client_fraction=0.6
        )
        
        print(f"Round {round_idx+1}/5 - Accuracy: {round_accuracy:.2f}%, Privacy Loss: {privacy_loss:.4f}")
    
    # Evaluate final model
    final_accuracy = fl_system.evaluate(test_data)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")

if __name__ == "__main__":
    main()
""")
    
    print("Created example scripts")

def main():
    """Initialize the reorganized project structure."""
    print("Initializing project structure...")
    
    # Create configuration files
    create_config_files()
    
    # Create documentation templates
    create_doc_templates()
    
    # Create example scripts
    create_examples()
    
    print("\nProject initialization complete!")
    print("\nYou can now:")
    print("1. Run the reorganization script: python reorganize_project.py")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Start the application: python main.py")

if __name__ == "__main__":
    main()
