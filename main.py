"""
Main application module for the Privacy-Preserving Federated Learning Framework.
Implements the Streamlit web interface and orchestrates the federated learning process.
Provides interactive parameter configuration and real-time visualization of training progress.
"""

import streamlit as st
import torch
import numpy as np
from data_handler import load_mnist_data
from federated_learning import FederatedLearning
from model import SimpleConvNet
from visualization import plot_training_progress, plot_privacy_metrics
from differential_privacy import add_noise
from database import TrainingRound, ExperimentConfig, get_db
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

# Sidebar parameters
st.sidebar.header("Parameters")
num_clients = st.sidebar.slider("Number of Clients", 2, 10, 5)
num_rounds = st.sidebar.slider("Number of Federated Rounds", 1, 20, 10)
local_epochs = st.sidebar.slider("Local Epochs", 1, 5, 2)
privacy_budget = st.sidebar.slider("Privacy Budget (ε)", 0.1, 10.0, 1.0)
noise_scale = st.sidebar.slider("Noise Scale (σ)", 0.0, 1.0, 0.1)

# Advanced parameters
st.sidebar.header("Advanced Parameters")
client_fraction = st.sidebar.slider("Client Participation Fraction", 0.1, 1.0, 0.6, 0.1)
use_non_iid = st.sidebar.checkbox("Use Non-IID Data Distribution", False)
alpha = st.sidebar.slider("Non-IID Concentration (α)", 0.1, 5.0, 0.5, 
                         help="Lower values create more skewed distributions across clients")
use_compression = st.sidebar.checkbox("Use Model Compression", False)
compression_ratio = st.sidebar.slider("Compression Ratio", 0.1, 1.0, 0.5, 0.1,
                                     help="Fraction of model weights to keep")

# Optional experiment description
experiment_description = st.sidebar.text_area("Experiment Description (optional)")

# Load and preprocess data
if 'data_loaded' not in st.session_state or st.session_state.use_non_iid != use_non_iid:
    st.session_state.data_loaded = False
    st.session_state.use_non_iid = use_non_iid

if not st.session_state.data_loaded:
    with st.spinner(f"Loading MNIST dataset with {'non-IID' if use_non_iid else 'IID'} distribution..."):
        train_data, val_data, test_data = load_mnist_data(iid=not use_non_iid)
        st.session_state.train_data = train_data
        st.session_state.val_data = val_data
        st.session_state.test_data = test_data
        st.session_state.data_loaded = True

# Initialize model and federated learning
model = SimpleConvNet()

# Custom configuration with new parameters
config = {
    'privacy': {
        'enabled': True,
        'noise_scale': noise_scale,
        'privacy_budget': privacy_budget
    },
    'compression': {
        'enabled': use_compression,
        'ratio': compression_ratio
    },
    'non_iid': {
        'enabled': use_non_iid,
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

# Display which features are enabled
feature_status = st.empty()
feature_text = []
if use_non_iid:
    feature_text.append(f"✅ Non-IID Data (α={alpha})")
else:
    feature_text.append("❌ Non-IID Data (using IID)")

if client_fraction < 1.0:
    feature_text.append(f"✅ Partial Client Participation ({int(client_fraction*100)}%)")
else:
    feature_text.append("❌ Partial Client Participation (using all clients)")

if use_compression:
    feature_text.append(f"✅ Model Compression (ratio={compression_ratio})")
else:
    feature_text.append("❌ Model Compression")

if noise_scale > 0:
    feature_text.append(f"✅ Differential Privacy (σ={noise_scale})")
else:
    feature_text.append("❌ Differential Privacy (no noise)")

feature_status.markdown("### Enabled Features:\n" + "\n".join(feature_text))

# Training
if st.button("Start Training"):
    # Save experiment configuration
    with get_session() as db:
        config = ExperimentConfig(
            num_clients=num_clients,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            privacy_budget=privacy_budget,
            noise_scale=noise_scale,
            description=experiment_description
        )
        db.add(config)
        db.commit()

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Metrics storage
    federated_accuracies = []
    privacy_losses = []

    for round_idx in range(num_rounds):
        # Perform one round of federated learning with new features
        round_accuracy, privacy_loss = fl_system.train_round(
            st.session_state.train_data,
            st.session_state.val_data,
            st.session_state.test_data,
            local_epochs,
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
                privacy_budget=privacy_budget,
                noise_scale=noise_scale
            )
            db.add(round_data)
            db.commit()

        # Update progress
        progress = (round_idx + 1) / num_rounds
        progress_bar.progress(progress)
        status_text.text(f"Round {round_idx+1}/{num_rounds} - Accuracy: {round_accuracy:.2f}%")

        # Plot progress
        col1, col2 = st.columns(2)
        with col1:
            plot_training_progress(federated_accuracies)
        with col2:
            plot_privacy_metrics(privacy_losses)

    st.success("Training completed!")

    # Final evaluation
    final_accuracy = fl_system.evaluate(st.session_state.test_data)
    st.metric("Final Test Accuracy", f"{final_accuracy:.2f}%")

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

# Add information about the framework
st.markdown("""
## About this Framework
This privacy-preserving federated learning framework implements:
- Federated averaging across multiple clients
- Differential privacy through noise addition
- Privacy budget monitoring
- Performance visualization and comparison
- Experiment tracking and storage
""")