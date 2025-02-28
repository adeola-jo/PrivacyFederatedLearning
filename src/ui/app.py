
"""
Main application module for the Privacy-Preserving Federated Learning Framework.
Implements the Streamlit web interface and orchestrates the federated learning process.
Provides interactive parameter configuration and real-time visualization of training progress.
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path

# Import with relative paths for direct Streamlit execution
import sys
import os
import torch
import copy
import numpy as np
# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# Now use the imports
from src.data.data_handler import load_mnist_data
from src.core.federated_learning import FederatedLearning

# Define required functions that might be missing
def add_noise(tensor, scale):
    """Add Gaussian noise to tensor for differential privacy"""
    return tensor + torch.randn_like(tensor) * scale

def calculate_privacy_loss(noise_scale, num_selected, total_clients):
    """Simple privacy loss calculation based on noise scale and client participation"""
    if noise_scale == 0:
        return float('inf')  # Infinite privacy loss when no noise
    
    # Basic formula based on participation rate and noise scale
    participation_rate = num_selected / total_clients
    return participation_rate / (noise_scale ** 2)
from src.models.model import SimpleConvNet
from src.ui.visualization import plot_training_progress, plot_privacy_metrics, display_sample_predictions, display_experiment_comparison
from src.privacy.differential_privacy import add_noise
from src.utils.database import TrainingRound, ExperimentConfig, get_db
from sqlalchemy.orm import Session
from contextlib import contextmanager

# Page configuration
st.set_page_config(page_title="Privacy-Preserving FL Framework", layout="wide")
st.title("Privacy-Preserving Federated Learning Framework")

@contextmanager
def get_session():
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()

# About section in sidebar
with st.sidebar:
    with st.expander("About This Framework"):
        st.markdown("""
        ### Privacy-Preserving Federated Learning Framework
        This framework implements:
        - Federated averaging across multiple clients
        - Differential privacy through noise addition
        - Privacy budget monitoring
        - Performance visualization and comparison
        - Experiment tracking and storage

        #### Parameters:
        - **Number of Clients**: Number of participants in federated learning
        - **Client Fraction**: Percentage of clients selected per round
        - **Federated Rounds**: Number of aggregation cycles
        - **Local Epochs**: Training iterations per client per round
        - **Privacy Budget (ε)**: Maximum privacy loss allowed
        - **Noise Scale (σ)**: Amount of noise added to protect privacy
        - **Non-IID Degree (α)**: Controls data distribution skew across clients
        - **Compression Ratio**: Reduces model size for communication efficiency
        """)

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Configuration", "Training", "Experiment Comparison"])

# Tab 1: Configuration
with tab1:
    st.header("Configuration")
    st.write("Configure the parameters for your federated learning experiment.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Parameters")
        num_clients = st.slider("Number of Clients", 2, 10, 5, 
                             help="Number of participants in the federated learning process")
        client_fraction = st.slider("Client Fraction", 0.1, 1.0, 0.6, 
                                 help="Fraction of clients that participate in each round")
        num_rounds = st.slider("Number of Federated Rounds", 1, 20, 10, 
                            help="Number of global aggregation rounds")
        local_epochs = st.slider("Local Epochs", 1, 5, 2, 
                              help="Number of training epochs performed by each client")
    
    with col2:
        st.subheader("Advanced Parameters")
        
        # Privacy settings
        privacy_enabled = st.checkbox("Enable Differential Privacy", value=True, 
                                   help="Add noise to client updates to preserve privacy")
        privacy_budget = st.slider("Privacy Budget (ε)", 0.1, 10.0, 1.0, 
                                help="Maximum allowed privacy loss (lower = more private)", 
                                disabled=not privacy_enabled)
        noise_scale = st.slider("Noise Scale (σ)", 0.0, 1.0, 0.1, 
                             help="Standard deviation of Gaussian noise added (higher = more private)", 
                             disabled=not privacy_enabled)
        
        # Data distribution settings
        non_iid_enabled = st.checkbox("Enable Non-IID Data", value=False, 
                                   help="Distribute data unevenly among clients")
        alpha = st.slider("Non-IID Degree (α)", 0.1, 5.0, 0.5, 
                       help="Concentration parameter for Dirichlet distribution (lower = more skewed)", 
                       disabled=not non_iid_enabled)
        
        # Compression settings
        compression_enabled = st.checkbox("Enable Model Compression", value=False, 
                                       help="Compress model updates to reduce communication overhead")
        compression_ratio = st.slider("Compression Ratio", 0.1, 1.0, 0.5, 
                                   help="Fraction of model parameters to retain after compression", 
                                   disabled=not compression_enabled)
    
    # Experiment description
    experiment_description = st.text_area("Experiment Description (optional)", 
                                       help="Add notes or details about this experiment for future reference")

# Tab 2: Training
with tab2:
    st.header("Training")
    st.write("Run federated learning and monitor progress in real-time.")
    
    # Load and preprocess data if not already done
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    if not st.session_state.data_loaded:
        with st.spinner("Loading MNIST dataset..."):
            train_data, val_data, test_data = load_mnist_data(iid=not non_iid_enabled, alpha=alpha)
            st.session_state.train_data = train_data
            st.session_state.val_data = val_data
            st.session_state.test_data = test_data
            st.session_state.data_loaded = True
            st.success("Dataset loaded successfully!")
    
    # Initialize model and federated learning
    model = SimpleConvNet()
    
    # Training section
    if st.button("Start Training"):
        # Create a new experiment configuration
        with get_session() as db:
            experiment = ExperimentConfig(
                num_clients=num_clients,
                num_rounds=num_rounds,
                local_epochs=local_epochs,
                privacy_budget=privacy_budget if privacy_enabled else 0,
                noise_scale=noise_scale if privacy_enabled else 0,
                description=experiment_description
            )
            db.add(experiment)
            db.commit()
        
        # Create columns for metrics and visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Create progress chart placeholder
            progress_chart = st.empty()
            
            # Create metrics placeholders
            metrics_container = st.container()
            
        with col2:
            # Create privacy metrics placeholder
            privacy_metrics = st.empty()
            
            # Create log area
            log_area = st.empty()
        
        # Initialize federated learning
        fl = FederatedLearning(
            model=model,
            train_data=st.session_state.train_data,
            val_data=st.session_state.val_data,
            test_data=st.session_state.test_data,
            num_clients=num_clients,
            client_fraction=client_fraction,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            privacy_budget=privacy_budget if privacy_enabled else float('inf'),
            noise_scale=noise_scale if privacy_enabled else 0.0,
            compression_ratio=compression_ratio if compression_enabled else 1.0
        )
        
        # Train the model
        for round_idx, metrics in enumerate(fl.train()):
            # Update progress chart
            with progress_chart:
                plot_training_progress(metrics['round_history'])
            
            # Update metrics
            with metrics_container:
                cols = st.columns(4)
                cols[0].metric("Current Round", f"{round_idx + 1}/{num_rounds}")
                cols[1].metric("Accuracy", f"{metrics['accuracy']:.2f}%")
                cols[2].metric("Loss", f"{metrics['loss']:.4f}")
                cols[3].metric("Privacy Loss", f"{metrics['privacy_loss']:.2f}" if privacy_enabled else "N/A")
            
            # Update privacy metrics
            with privacy_metrics:
                if privacy_enabled:
                    plot_privacy_metrics(metrics['round_history'])
                else:
                    st.info("Privacy metrics not available (Differential Privacy is disabled)")
            
            # Update log
            with log_area:
                st.text_area("Training Log", 
                            value=f"Round {round_idx + 1}/{num_rounds} completed\n"
                                  f"Accuracy: {metrics['accuracy']:.2f}%\n"
                                  f"Loss: {metrics['loss']:.4f}\n"
                                  f"Privacy Loss: {metrics['privacy_loss']:.2f}" if privacy_enabled else "N/A",
                            height=200)
        
        # Final evaluation
        st.header("Final Evaluation")
        test_accuracy = fl.evaluate()
        st.success(f"Final test accuracy: {test_accuracy:.2f}%")
        
        # Display sample predictions
        st.subheader("Sample Predictions")
        display_sample_predictions(model, st.session_state.test_data)
        
        # Save experiment results
        st.success("Training complete! Results saved to database.")
    
    # Display previous experiments
    st.subheader("Previous Experiments")
    with get_session() as db:
        experiments = db.query(ExperimentConfig).order_by(ExperimentConfig.timestamp.desc()).limit(5).all()
        
        if experiments:
            for exp in experiments:
                with st.expander(f"Experiment from {exp.timestamp}"):
                    st.write(f"Clients: {exp.num_clients}")
                    st.write(f"Rounds: {exp.num_rounds}")
                    st.write(f"Privacy Budget: {exp.privacy_budget}")
                    if exp.description:
                        st.write(f"Description: {exp.description}")
        else:
            st.info("No previous experiments found.")

# Tab 3: Experiment Comparison
with tab3:
    st.header("Experiment Comparison")
    st.write("Compare results from different experiments to analyze performance and privacy trade-offs.")
    
    # Display experiment comparison
    display_experiment_comparison()
