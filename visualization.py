import plotly.graph_objects as go
import streamlit as st

def plot_training_progress(accuracies):
    """Plot training accuracy progress"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=accuracies,
        mode='lines+markers',
        name='Federated Learning Accuracy'
    ))
    
    fig.update_layout(
        title='Training Progress',
        xaxis_title='Round',
        yaxis_title='Accuracy (%)',
        yaxis_range=[0, 100]
    )
    
    st.plotly_chart(fig)

def plot_privacy_metrics(privacy_losses):
    """Plot privacy loss metrics"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=privacy_losses,
        mode='lines+markers',
        name='Privacy Loss'
    ))
    
    fig.update_layout(
        title='Privacy Loss over Training',
        xaxis_title='Round',
        yaxis_title='Privacy Loss (ε)',
    )
    
    st.plotly_chart(fig)
