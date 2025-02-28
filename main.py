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
from visualization import (
    create_training_progress_chart, create_privacy_metrics_chart,
    update_training_progress, update_privacy_metrics,
    display_experiment_comparison
)
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
    
    # Add visualization of non-IID distribution
    if st.checkbox("Show Client Data Distribution"):
        st.write("Preparing visualization of data distribution across clients...")
        # Distribute data among clients
        client_datasets = fl_system.distribute_data(
            st.session_state.train_data, 
            use_non_iid=True, 
            alpha=alpha
        )
        
        # Collect label distributions for visualization
        import plotly.graph_objects as go
        import numpy as np
        
        fig = go.Figure()
        client_distributions = []
        
        for i, dataset in enumerate(client_datasets):
            # Get labels for this client's data
            if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'tensors'):
                indices = dataset.indices
                labels = torch.argmax(dataset.dataset.tensors[1][indices], dim=1).numpy()
            else:
                # Handle other dataset types
                try:
                    indices = dataset.indices
                    labels = []
                    for idx in indices:
                        if hasattr(dataset.dataset, 'targets'):
                            if torch.is_tensor(dataset.dataset.targets):
                                labels.append(dataset.dataset.targets[idx].item())
                            else:
                                labels.append(dataset.dataset.targets[idx])
                        else:
                            labels.append(torch.argmax(dataset.dataset.tensors[1][idx]).item())
                    labels = np.array(labels)
                except:
                    st.warning(f"Could not extract labels for client {i}")
                    continue
            
            # Count occurrences of each label
            unique, counts = np.unique(labels, return_counts=True)
            distribution = np.zeros(10)  # Assuming 10 classes for MNIST
            distribution[unique] = counts / counts.sum()
            client_distributions.append(distribution)
            
            # Plot the distribution with Plotly
            fig.add_trace(go.Bar(
                x=np.arange(10),
                y=distribution,
                name=f'Client {i+1}',
                opacity=0.7,
                width=0.1,
                offset=i * 0.1 - 0.5  # Center the bars
            ))
        
        fig.update_layout(
            title='Data Distribution Across Clients',
            xaxis_title='Class Label',
            yaxis_title='Proportion',
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
else:
    feature_text.append("❌ Non-IID Data (using IID)")

if client_fraction < 1.0:
    feature_text.append(f"✅ Partial Client Participation ({int(client_fraction*100)}%)")
else:
    feature_text.append("❌ Partial Client Participation (using all clients)")

if use_compression:
    feature_text.append(f"✅ Model Compression (ratio={compression_ratio})")
    
    # Add visualization of model compression effect
    if st.checkbox("Show Model Compression Impact"):
        st.write("Visualizing model compression impact...")
        
        # Create a sample model and compress it
        sample_model = SimpleConvNet()
        state_dict = sample_model.state_dict()
        
        # Calculate original model size
        original_size = sum(param.nelement() * param.element_size() 
                           for param in sample_model.parameters()) / 1024
        
        # Compress the model
        compressed_state = fl_system.compress_model(state_dict, compression_ratio)
        
        # Calculate number of non-zero parameters
        original_nonzeros = sum(torch.count_nonzero(tensor).item() 
                               for name, tensor in state_dict.items() 
                               if isinstance(tensor, torch.Tensor))
        
        compressed_nonzeros = sum(torch.count_nonzero(tensor).item() 
                                 for name, tensor in compressed_state.items() 
                                 if isinstance(tensor, torch.Tensor))
        
        # Create comparison metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Model Parameters", f"{original_nonzeros:,}")
        with col2:
            st.metric("After Compression", f"{compressed_nonzeros:,}", 
                     delta=f"{-100 * (1 - compressed_nonzeros/original_nonzeros):.1f}%")
            
        # Show size comparison bar chart with Plotly
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Original', 'Compressed'],
                y=[original_nonzeros, compressed_nonzeros],
                marker_color=['royalblue', 'limegreen'],
                text=[f'{original_nonzeros:,}', f'{compressed_nonzeros:,}'],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title='Effect of Model Compression',
            yaxis_title='Number of Non-Zero Parameters',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
else:
    feature_text.append("❌ Model Compression")

if noise_scale > 0:
    feature_text.append(f"✅ Differential Privacy (σ={noise_scale})")
else:
    feature_text.append("❌ Differential Privacy (no noise)")

feature_status.markdown("### Enabled Features:\n" + "\n".join(feature_text))

# Create tabs for training and comparison
tab1, tab2 = st.tabs(["Training", "Experiment Comparison"])

with tab1:
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
        
        # Create static charts that will be updated
        col1, col2 = st.columns(2)
        with col1:
            acc_chart_placeholder = st.empty()
            acc_fig = create_training_progress_chart()
            acc_chart_placeholder.plotly_chart(acc_fig, use_container_width=True)
            
        with col2:
            priv_chart_placeholder = st.empty()
            priv_fig = create_privacy_metrics_chart()
            priv_chart_placeholder.plotly_chart(priv_fig, use_container_width=True)

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

            # Update the static charts with new data
            acc_fig = update_training_progress(acc_fig, federated_accuracies)
            acc_chart_placeholder.plotly_chart(acc_fig, use_container_width=True)
            
            priv_fig = update_privacy_metrics(priv_fig, privacy_losses)
            priv_chart_placeholder.plotly_chart(priv_fig, use_container_width=True)

        st.success("Training completed!")

        # Final evaluation
        final_accuracy = fl_system.evaluate(st.session_state.test_data)
        st.metric("Final Test Accuracy", f"{final_accuracy:.2f}%")

    # Display previous experiments
    with get_session() as db:
        st.subheader("Recent Experiments")
        experiments = db.query(ExperimentConfig).order_by(ExperimentConfig.timestamp.desc()).limit(5).all()

        if experiments:
            for exp in experiments:
                with st.expander(f"Experiment from {exp.timestamp}"):
                    st.write(f"Clients: {exp.num_clients}")
                    st.write(f"Rounds: {exp.num_rounds}")
                    st.write(f"Local Epochs: {exp.local_epochs}")
                    st.write(f"Privacy Budget: {exp.privacy_budget}")
                    st.write(f"Noise Scale: {exp.noise_scale}")
                    if exp.description:
                        st.write(f"Description: {exp.description}")
        else:
            st.info("No recent experiments found. Start a training session to see results.")

with tab2:
    # Use the comparison function from visualization.py
    display_experiment_comparison()

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