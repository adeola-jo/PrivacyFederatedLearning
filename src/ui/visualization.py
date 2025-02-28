"""
Visualization module for the federated learning framework.
Provides functions for plotting training progress and privacy metrics
using Plotly and Streamlit.
"""

import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from src.utils.database import TrainingRound, ExperimentConfig, get_db
from sqlalchemy import desc
import numpy as np
from contextlib import contextmanager
import matplotlib.pyplot as plt
import plotly.express as px
import torch
from torchvision import transforms
import io
from PIL import Image

@contextmanager
def get_session():
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()

def create_training_progress_chart():
    """Create a static chart for training progress that can be updated"""
    # Create a placeholder figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[], y=[],
        mode='lines+markers',
        name='Federated Learning Accuracy'
    ))

    fig.update_layout(
        title='Training Progress',
        xaxis_title='Round',
        yaxis_title='Accuracy (%)',
        yaxis_range=[0, 100],
        height=400,
    )

    return fig

def create_privacy_metrics_chart():
    """Create a static chart for privacy metrics that can be updated"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[], y=[],
        mode='lines+markers',
        name='Privacy Loss'
    ))

    fig.update_layout(
        title='Privacy Loss over Training',
        xaxis_title='Round',
        yaxis_title='Privacy Loss (ε)',
        height=400,
    )

    return fig

def update_training_progress(fig, accuracies):
    """Update the training progress chart with new data"""
    fig.data[0].x = list(range(1, len(accuracies) + 1))
    fig.data[0].y = accuracies
    return fig

def update_privacy_metrics(fig, privacy_losses):
    """Update the privacy metrics chart with new data"""
    fig.data[0].x = list(range(1, len(privacy_losses) + 1))
    fig.data[0].y = privacy_losses
    return fig

def display_experiment_comparison():
    """Display a comparison of all experiments"""
    st.title("Experiment Comparison")

    with get_session() as db:
        # Get all experiments
        experiments = db.query(ExperimentConfig).order_by(desc(ExperimentConfig.timestamp)).all()

        if not experiments:
            st.info("No experiments found. Run a training session to see results here.")
            return

        # Create dataframe with experiment metadata
        experiment_df = pd.DataFrame([
            {
                'id': exp.id,
                'timestamp': exp.timestamp,
                'num_clients': exp.num_clients,
                'num_rounds': exp.num_rounds,
                'local_epochs': exp.local_epochs,
                'privacy_budget': exp.privacy_budget,
                'noise_scale': exp.noise_scale,
                'description': exp.description or "No description"
            } for exp in experiments
        ])

        # Display experiment selector
        st.subheader("Select Experiments to Compare")
        cols = st.columns(3)

        with cols[0]:
            selected_experiments = st.multiselect(
                "Choose experiments",
                options=experiment_df.index,
                format_func=lambda i: f"{experiment_df.iloc[i]['timestamp'].strftime('%Y-%m-%d %H:%M')} - {experiment_df.iloc[i]['description'][:20]}"
            )

        with cols[1]:
            chart_type = st.selectbox("Chart type", ["Accuracy", "Privacy Loss", "Both"])

        with cols[2]:
            normalize_rounds = st.checkbox("Normalize rounds", True, 
                help="Scale x-axis to show percentage of total rounds for each experiment")

        if not selected_experiments:
            st.info("Select at least one experiment to compare")
            return

        # Get training data for selected experiments
        all_experiment_data = {}
        max_rounds = 0

        for idx in selected_experiments:
            exp_id = experiment_df.iloc[idx]['id']
            exp_rounds = db.query(TrainingRound).filter(
                TrainingRound.num_clients == experiment_df.iloc[idx]['num_clients'],
                TrainingRound.privacy_budget == experiment_df.iloc[idx]['privacy_budget'],
                TrainingRound.noise_scale == experiment_df.iloc[idx]['noise_scale'],
                TrainingRound.timestamp >= experiment_df.iloc[idx]['timestamp']
            ).order_by(TrainingRound.round_number).limit(experiment_df.iloc[idx]['num_rounds']).all()

            exp_data = {
                'rounds': [r.round_number for r in exp_rounds],
                'accuracy': [r.accuracy for r in exp_rounds],
                'privacy_loss': [r.privacy_loss for r in exp_rounds],
                'metadata': experiment_df.iloc[idx]
            }

            if len(exp_data['rounds']) > max_rounds:
                max_rounds = len(exp_data['rounds'])

            all_experiment_data[exp_id] = exp_data

        # Create comparison charts
        if chart_type in ["Accuracy", "Both"]:
            acc_fig = go.Figure()

            for exp_id, exp_data in all_experiment_data.items():
                if normalize_rounds and len(exp_data['rounds']) > 0:
                    x_values = [r / exp_data['metadata']['num_rounds'] * 100 for r in exp_data['rounds']]
                    x_title = "Training Progress (%)"
                else:
                    x_values = exp_data['rounds']
                    x_title = "Round"

                exp_name = f"{exp_data['metadata']['timestamp'].strftime('%m-%d %H:%M')} - "
                exp_name += f"{exp_data['metadata']['description'][:15]}..." if len(exp_data['metadata']['description']) > 15 else exp_data['metadata']['description']
                exp_name += f" (C:{exp_data['metadata']['num_clients']}, E:{exp_data['metadata']['local_epochs']}, ε:{exp_data['metadata']['privacy_budget']})"

                acc_fig.add_trace(go.Scatter(
                    x=x_values,
                    y=exp_data['accuracy'],
                    mode='lines+markers',
                    name=exp_name
                ))

            acc_fig.update_layout(
                title='Accuracy Comparison',
                xaxis_title=x_title,
                yaxis_title='Accuracy (%)',
                yaxis_range=[0, 100],
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.5,
                    xanchor="center",
                    x=0.5
                )
            )

            st.plotly_chart(acc_fig, use_container_width=True)

        if chart_type in ["Privacy Loss", "Both"]:
            priv_fig = go.Figure()

            for exp_id, exp_data in all_experiment_data.items():
                if normalize_rounds and len(exp_data['rounds']) > 0:
                    x_values = [r / exp_data['metadata']['num_rounds'] * 100 for r in exp_data['rounds']]
                    x_title = "Training Progress (%)"
                else:
                    x_values = exp_data['rounds']
                    x_title = "Round"

                exp_name = f"{exp_data['metadata']['timestamp'].strftime('%m-%d %H:%M')} - "
                exp_name += f"{exp_data['metadata']['description'][:15]}..." if len(exp_data['metadata']['description']) > 15 else exp_data['metadata']['description']
                exp_name += f" (C:{exp_data['metadata']['num_clients']}, E:{exp_data['metadata']['local_epochs']}, σ:{exp_data['metadata']['noise_scale']})"

                priv_fig.add_trace(go.Scatter(
                    x=x_values,
                    y=exp_data['privacy_loss'],
                    mode='lines+markers',
                    name=exp_name
                ))

            priv_fig.update_layout(
                title='Privacy Loss Comparison',
                xaxis_title=x_title,
                yaxis_title='Privacy Loss (ε)',
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.5,
                    xanchor="center",
                    x=0.5
                )
            )

            st.plotly_chart(priv_fig, use_container_width=True)

        # Display experiment metrics table
        st.subheader("Experiment Final Metrics")

        metrics_data = []
        for exp_id, exp_data in all_experiment_data.items():
            if exp_data['accuracy']:
                metrics_data.append({
                    'Experiment': exp_data['metadata']['description'] or f"Experiment {exp_id}",
                    'Date': exp_data['metadata']['timestamp'].strftime('%Y-%m-%d %H:%M'),
                    'Clients': exp_data['metadata']['num_clients'],
                    'Rounds': exp_data['metadata']['num_rounds'],
                    'Local Epochs': exp_data['metadata']['local_epochs'],
                    'Privacy Budget': exp_data['metadata']['privacy_budget'],
                    'Noise Scale': exp_data['metadata']['noise_scale'],
                    'Final Accuracy': exp_data['accuracy'][-1] if exp_data['accuracy'] else 'N/A',
                    'Final Privacy Loss': exp_data['privacy_loss'][-1] if exp_data['privacy_loss'] else 'N/A'
                })

        if metrics_data:
            st.dataframe(pd.DataFrame(metrics_data))


def plot_training_progress(accuracies):
    """Plot training progress using Plotly."""
    fig = px.line(
        x=list(range(1, len(accuracies) + 1)),
        y=accuracies,
        labels={'x': 'Round', 'y': 'Accuracy (%)'},
        title='Federated Learning Progress'
    )
    fig.update_layout(
        xaxis_title="Round",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 100]
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_privacy_metrics(privacy_losses):
    """Plot privacy metrics using Plotly."""
    fig = px.line(
        x=list(range(1, len(privacy_losses) + 1)),
        y=privacy_losses,
        labels={'x': 'Round', 'y': 'Privacy Loss (ε)'},
        title='Privacy Budget Usage'
    )
    fig.update_layout(
        xaxis_title="Round",
        yaxis_title="Privacy Loss (ε)"
    )
    st.plotly_chart(fig, use_container_width=True)

def display_sample_predictions(model, test_data, num_samples=5):
    """
    Display sample predictions from the test set in a grid format.

    Args:
        model: The trained model
        test_data: Test dataset
        num_samples: Number of samples to display
    """
    # Set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device

    # Get samples from test data
    dataloader = torch.utils.data.DataLoader(test_data, batch_size=num_samples, shuffle=True)
    images, labels = next(iter(dataloader))

    # Make predictions
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Create a grid with Plotly
    cols = min(5, num_samples)  # Max 5 columns
    rows = (num_samples + cols - 1) // cols

    fig = plt.figure(figsize=(2*cols, 2*rows))

    for i in range(num_samples):
        # Get the current image
        image = images[i].cpu().numpy().squeeze()
        true_label = labels[i].item()
        pred_label = predicted[i].item()

        # Add subplot
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(image, cmap='gray')
        ax.set_title(f'True: {true_label}\nPred: {pred_label}')
        ax.axis('off')

    plt.tight_layout()

    # Display in Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    st.image(buf, use_column_width=True)
    plt.close(fig)