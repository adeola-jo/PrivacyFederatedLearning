
"""
Main application module for the Privacy-Preserving Federated Learning Framework.
Implements the Streamlit web interface and orchestrates the federated learning process.
Provides interactive parameter configuration and real-time visualization of training progress.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
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
import plotly.express as px
import plotly.graph_objects as go
from torchvision import transforms
from torch.utils.data import DataLoader

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

# Helper function to create parameter tooltips
def param_help(title, description):
    return f"{description}"

# Create three main tabs for the application
tab_config, tab_train, tab_compare = st.tabs(["Configuration", "Training", "Experiment Comparison"])

with tab_config:
    # Two columns for parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Federated Learning Parameters")
        
        with st.expander("About these parameters", expanded=False):
            st.markdown("""
            **Number of Clients**: Total number of clients participating in federated learning.
            
            **Client Participation Fraction**: Proportion of clients that participate in each round.
            
            **Number of Federated Rounds**: Total training rounds to perform.
            
            **Local Epochs**: Number of training epochs each client performs locally.
            """)
        
        num_clients = st.slider("Number of Clients", 2, 10, 5, 
                               help=param_help("Clients", "Number of clients participating in federated learning"))
        
        client_fraction = st.slider("Client Participation Fraction", 0.1, 1.0, 0.6, 0.1, 
                                  help=param_help("Participation", "Fraction of clients selected in each round"))
        
        num_rounds = st.slider("Number of Federated Rounds", 1, 20, 10, 
                              help=param_help("Rounds", "Total rounds of federated training"))
        
        local_epochs = st.slider("Local Epochs", 1, 5, 2, 
                               help=param_help("Epochs", "Number of epochs each client trains locally"))
        
    with col2:
        st.header("Privacy & Distribution Parameters")
        
        with st.expander("About these parameters", expanded=False):
            st.markdown("""
            **Privacy Budget (ε)**: Controls the privacy-utility tradeoff. Lower values provide stronger privacy.
            
            **Noise Scale (σ)**: Amount of noise added for differential privacy. Higher values increase privacy.
            
            **Non-IID Distribution**: When enabled, clients receive data that is not identically distributed.
            
            **Non-IID Concentration (α)**: Controls how skewed the data distribution is across clients.
            
            **Model Compression**: Reduces communication overhead by pruning model weights.
            
            **Compression Ratio**: Fraction of model weights to keep during compression.
            """)
        
        privacy_budget = st.slider("Privacy Budget (ε)", 0.1, 10.0, 1.0, 
                                 help=param_help("Privacy Budget", "Controls privacy-utility tradeoff (lower is more private)"))
        
        noise_scale = st.slider("Noise Scale (σ)", 0.0, 1.0, 0.1, 
                              help=param_help("Noise Scale", "Amount of noise added to model updates"))
        
        use_non_iid = st.checkbox("Use Non-IID Data Distribution", False, 
                                help=param_help("Non-IID", "Creates a more realistic scenario where data is not identically distributed"))
        
        alpha = st.slider("Non-IID Concentration (α)", 0.1, 5.0, 0.5, 
                         help=param_help("Alpha", "Lower values create more skewed distributions across clients"))
        
        use_compression = st.checkbox("Use Model Compression", False, 
                                    help=param_help("Compression", "Reduces communication overhead by pruning model weights"))
        
        compression_ratio = st.slider("Compression Ratio", 0.1, 1.0, 0.5, 0.1, 
                                     help=param_help("Ratio", "Fraction of model weights to keep during compression"))
    
    # Optional experiment description
    st.header("Experiment Description")
    experiment_description = st.text_area("Add a description for this experiment (optional)", 
                                         placeholder="e.g., Testing impact of noise scale on model accuracy")

# Global state for data loading
if 'data_loaded' not in st.session_state or st.session_state.use_non_iid != use_non_iid:
    st.session_state.data_loaded = False
    st.session_state.use_non_iid = use_non_iid

# Training tab content
with tab_train:
    # Load data if needed
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
    
    # Display enabled features
    st.header("Current Configuration")
    
    # Create a nicer feature summary with icons
    feature_col1, feature_col2 = st.columns(2)
    with feature_col1:
        st.subheader("Client Setup")
        st.markdown(f"**Clients:** {num_clients}")
        st.markdown(f"**Client Participation:** {int(client_fraction*100)}%")
        st.markdown(f"**Training Rounds:** {num_rounds}")
        st.markdown(f"**Local Epochs:** {local_epochs}")
    
    with feature_col2:
        st.subheader("Privacy & Data")
        st.markdown(f"**Privacy Budget (ε):** {privacy_budget}")
        st.markdown(f"**Noise Scale (σ):** {noise_scale}")
        st.markdown(f"**Data Distribution:** {'Non-IID (α=' + str(alpha) + ')' if use_non_iid else 'IID'}")
        st.markdown(f"**Model Compression:** {'Enabled (' + str(int(compression_ratio*100)) + '%)' if use_compression else 'Disabled'}")
    
    # Show non-IID distribution visualization if enabled
    if use_non_iid and st.checkbox("Show Client Data Distribution"):
        st.write("Visualizing data distribution across clients...")
        # Distribute data among clients
        client_datasets = fl_system.distribute_data(
            st.session_state.train_data, 
            use_non_iid=True, 
            alpha=alpha
        )
        
        # Collect label distributions for visualization
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
    
    # Show model compression effect if enabled
    if use_compression and st.checkbox("Show Model Compression Impact"):
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
    
    # Training section
    st.header("Training")
    
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
        
        # Store the model in session state for testing
        st.session_state.trained_model = model
        st.session_state.has_trained_model = True

        st.success("Training completed!")

        # Final evaluation
        final_accuracy = fl_system.evaluate(st.session_state.test_data)
        st.metric("Final Test Accuracy", f"{final_accuracy:.2f}%")
    
    # Test model section
    st.header("Test Global Model")
    
    # Check if we have a trained model
    has_model = st.session_state.get('has_trained_model', False)
    
    if not has_model:
        st.info("Train a model first or load a previously trained model to test it.")
    else:
        test_button = st.button("Run Test on Global Model")
        
        if test_button:
            # Get the trained model from session state
            test_model = st.session_state.trained_model
            test_model.eval()
            
            # Evaluate on test data
            test_loader = DataLoader(st.session_state.test_data, batch_size=64, shuffle=False)
            
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            all_images = []
            
            # Get device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(test_loader):
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = test_model(images)
                    _, predicted = torch.max(outputs, 1)
                    
                    total += labels.size(0)
                    # Get the indices of the highest values in the target one-hot encoding
                    target_indices = torch.argmax(labels, dim=1)
                    batch_correct = predicted.eq(target_indices).sum().item()
                    correct += batch_correct
                    
                    # Store results for display
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(target_indices.cpu().numpy())
                    all_images.extend(images.cpu())
                    
                    # Only collect a subset for visualization
                    if len(all_images) >= 100:
                        break
            
            # Calculate accuracy
            accuracy = 100. * correct / total
            
            # Display accuracy
            st.metric("Test Accuracy", f"{accuracy:.2f}%", 
                     delta=f"{accuracy - st.session_state.get('previous_accuracy', accuracy):.2f}%" 
                     if 'previous_accuracy' in st.session_state else None)
            
            st.session_state.previous_accuracy = accuracy
            
            # Let user select number of examples to view
            num_examples = st.slider("Number of examples to display", 1, min(10, len(all_images)), 5)
            
            # Display examples in a grid
            if num_examples > 0:
                st.subheader(f"Sample Test Images and Predictions")
                
                # Create columns for the grid
                cols = st.columns(num_examples)
                
                # Choose random examples to display
                indices = np.random.choice(len(all_images), num_examples, replace=False)
                
                for i, idx in enumerate(indices):
                    img = all_images[idx]
                    true_label = all_labels[idx]
                    pred_label = all_preds[idx]
                    
                    # Convert tensor to numpy and remove channel dimension for grayscale images
                    if img.dim() == 3:  # [C, H, W]
                        img_np = img.squeeze(0).numpy() if img.shape[0] == 1 else img.numpy().transpose(1, 2, 0)
                    else:
                        img_np = img.numpy()
                    
                    with cols[i]:
                        # Use Plotly for the visualization
                        fig = px.imshow(img_np, color_continuous_scale='gray')
                        fig.update_layout(
                            coloraxis_showscale=False,
                            margin=dict(l=0, r=0, t=30, b=0),
                            height=200, width=200
                        )
                        
                        # Add title showing prediction and true label
                        status = "✓" if pred_label == true_label else "✗"
                        fig.update_layout(
                            title=f"Pred: {pred_label} {status}<br>True: {true_label}"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    # Display previous experiments
    st.header("Recent Experiments")
    with get_session() as db:
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

# Comparison tab
with tab_compare:
    # Use the comparison function from visualization.py
    display_experiment_comparison()

# Replace the footer
st.sidebar.markdown("---")
with st.sidebar.expander("About this Framework", expanded=False):
    st.markdown("""
    ### Privacy-Preserving Federated Learning Framework
    
    This framework implements:
    - **Federated Averaging (FedAvg)** across multiple clients
    - **Differential Privacy** through noise addition
    - **Privacy Budget Monitoring** to track privacy loss
    - **Non-IID Data Distribution** to simulate realistic scenarios
    - **Model Compression** to reduce communication costs
    - **Experiment Tracking** for comparing results
    
    The framework is designed for research and educational purposes to demonstrate privacy-preserving machine learning techniques.
    """)
