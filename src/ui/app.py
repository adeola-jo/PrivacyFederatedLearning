"""
Main application module for the Privacy-Preserving Federated Learning Framework.
Implements the Streamlit web interface and orchestrates the federated learning process.
Provides interactive parameter configuration and real-time visualization of training progress.
"""

import streamlit as st
import torch
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Import project modules
from src.data.data_handler import load_mnist_data
from src.core.federated_learning import FederatedLearning
from src.models.model import SimpleConvNet
from src.ui.visualization import plot_training_progress, plot_privacy_metrics, display_sample_predictions
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

# Create info expansion section instead of static text at bottom
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

# Sidebar parameters
st.sidebar.header("Parameters")
num_clients = st.sidebar.slider("Number of Clients", 2, 10, 5)
client_fraction = st.sidebar.slider("Client Fraction", 0.1, 1.0, 0.6)
num_rounds = st.sidebar.slider("Number of Federated Rounds", 1, 20, 10)
local_epochs = st.sidebar.slider("Local Epochs", 1, 5, 2)

# Create collapsible sections for advanced parameters
with st.sidebar.expander("Privacy Settings"):
    privacy_enabled = st.checkbox("Enable Differential Privacy", value=True)
    privacy_budget = st.slider("Privacy Budget (ε)", 0.1, 10.0, 1.0, disabled=not privacy_enabled)
    noise_scale = st.slider("Noise Scale (σ)", 0.0, 1.0, 0.1, disabled=not privacy_enabled)

with st.sidebar.expander("Data Distribution"):
    non_iid_enabled = st.checkbox("Enable Non-IID Data", value=False)
    alpha = st.slider("Non-IID Degree (α)", 0.1, 5.0, 0.5, disabled=not non_iid_enabled)

with st.sidebar.expander("Compression Settings"):
    compression_enabled = st.checkbox("Enable Model Compression", value=False)
    compression_ratio = st.slider("Compression Ratio", 0.1, 1.0, 0.5, disabled=not compression_enabled)

# Optional experiment description
experiment_description = st.sidebar.text_area("Experiment Description (optional)")

# Load and preprocess data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if not st.session_state.data_loaded:
    with st.spinner("Loading MNIST dataset..."):
        train_data, val_data, test_data = load_mnist_data(iid=not non_iid_enabled, alpha=alpha)
        st.session_state.train_data = train_data
        st.session_state.val_data = val_data
        st.session_state.test_data = test_data
        st.session_state.data_loaded = True

# Initialize model and federated learning
model = SimpleConvNet()

# Create config dictionary
config = {
    'privacy': {
        'enabled': privacy_enabled,
        'noise_scale': noise_scale,
        'privacy_budget': privacy_budget
    },
    'compression': {
        'enabled': compression_enabled,
        'ratio': compression_ratio
    },
    'non_iid': {
        'enabled': non_iid_enabled,
        'alpha': alpha
    },
    'batch_size': 64,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'verbose': True
}

fl_system = FederatedLearning(
    model, 
    num_clients=num_clients,
    config=config
)

# Training
col1, col2 = st.columns(2)
start_button = col1.button("Start Training")
test_button = col2.button("Test Aggregated Model")

if start_button:
    # Save experiment configuration
    with get_session() as db:
        config_record = ExperimentConfig(
            num_clients=num_clients,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            privacy_budget=privacy_budget if privacy_enabled else 0,
            noise_scale=noise_scale if privacy_enabled else 0,
            description=experiment_description
        )
        db.add(config_record)
        db.commit()

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Metrics storage
    federated_accuracies = []
    privacy_losses = []

    for round_idx in range(num_rounds):
        # Perform one round of federated learning
        round_accuracy, privacy_loss = fl_system.train_round(
            st.session_state.train_data,
            st.session_state.val_data,
            st.session_state.test_data,
            local_epochs=local_epochs,
            client_fraction=client_fraction
        )

        federated_accuracies.append(round_accuracy)
        privacy_losses.append(privacy_loss)

        # Save round results
        with get_session() as db:
            round_data = TrainingRound(
                round_number=round_idx + 1,
                accuracy=round_accuracy,
                privacy_loss=privacy_loss,
                num_clients=num_clients,
                privacy_budget=privacy_budget if privacy_enabled else 0,
                noise_scale=noise_scale if privacy_enabled else 0
            )
            db.add(round_data)
            db.commit()

        # Update progress
        progress = (round_idx + 1) / num_rounds
        progress_bar.progress(progress)
        status_text.text(f"Round {round_idx+1}/{num_rounds} - Accuracy: {round_accuracy:.2f}%")

        # Plot progress
        vis_col1, vis_col2 = st.columns(2)
        with vis_col1:
            plot_training_progress(federated_accuracies)
        with vis_col2:
            plot_privacy_metrics(privacy_losses)

    st.success("Training completed!")

    # Store the model in session state for testing
    st.session_state.trained_model = fl_system.global_model

    # Final evaluation
    final_accuracy = fl_system.evaluate(st.session_state.test_data)
    st.metric("Final Test Accuracy", f"{final_accuracy:.2f}%")

# Test the model and show sample predictions
if test_button or ('trained_model' in st.session_state and start_button):
    if 'trained_model' in st.session_state:
        st.subheader("Test Results")

        # Evaluate the model
        if 'test_data' in st.session_state:
            final_accuracy = fl_system.evaluate(st.session_state.test_data)
            st.metric("Test Accuracy", f"{final_accuracy:.2f}%")

            # Control for number of samples to display
            num_samples = st.slider("Number of test samples to display", min_value=1, max_value=10, value=5)
            st.write("Sample Predictions:")

            # Display sample predictions
            display_sample_predictions(fl_system.global_model, st.session_state.test_data, num_samples)
        else:
            st.warning("Test data not loaded. Please start training first.")
    else:
        st.warning("No trained model available. Please complete training first.")

# Display previous experiments
with get_session() as db:
    st.subheader("Previous Experiments")
    experiments = db.query(ExperimentConfig).order_by(ExperimentConfig.timestamp.desc()).limit(5).all()

    if experiments:
        for exp in experiments:
            with st.expander(f"Experiment from {exp.timestamp}"):
                st.write(f"Clients: {exp.num_clients}")
                st.write(f"Rounds: {exp.num_rounds}")
                st.write(f"Privacy Budget: {exp.privacy_budget}")
                if exp.description:
                    st.write(f"Description: {exp.description}")