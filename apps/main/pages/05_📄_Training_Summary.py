"""
Predictions & Residuals Page
Model predictions analysis and residual visualization
Migrated from report_v2.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import json
import torch
from pathlib import Path
import re
from datetime import datetime

from utils.initialization import atmospheric_model_chooser

# Add src to path for imports
# Path structure: apps/main_platform/pages/05_... -> project_root
project_root = Path(__file__).resolve().parent.parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from utils.model_eval import ModelEvaluator
from utils.loss_functions import combined_loss
from utils.dataset_manager import DatasetManager
from utils.colors import apply_theme, get_color, colored_metric, status_badge

# Apply OneSun theme
apply_theme()

st.title("📄 Training Summary")
#st.markdown("Analyze model predictions, residuals, and performance across different spectral measurements.")

# Get experiment runs directory
experiment_runs_dir = st.session_state.experiment_runs_dir


# === USE EXPERIMENT SELECTION FROM 04_EXPERIMENT_OVERVIEW ===
# Get selected experiment from session state (set in 04_Experiment_Overview)
selected_experiment = st.session_state.get('selected_experiment')

if not selected_experiment:
    st.info("🔍 **No experiment selected.** Please select an experiment first in **📈 Analysis → Experiment Overview**.")
    st.stop()

# Display current selection
st.markdown(f"Current Experiment: {selected_experiment}")

experiment_dir = experiment_runs_dir / selected_experiment

# Load saved models (with experiment-specific caching to fix update bug)
@st.cache_resource
@st.cache_resource
def load_saved_models(experiment_dir_str):
    """
    Load the model architecture, weights, and both train/val DataLoaders based on saved config.
    Returns:
        model_i (nn.Module) or None on error
        model_a (nn.Module) or None on error
        train_loader (DataLoader) or None on error
        val_loader (DataLoader) or None on error
    """
    from utils.initialization import load_constants
    from utils.initialization import instrument_model_chooser, atmospheric_model_chooser
    from pathlib import Path
    import torch

    experiment_dir = Path(experiment_dir_str)

    # 1) Load configuration
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        st.error("No config.json found in experiment directory.")
        return None, None, None, None
    with open(config_path, "r") as f:
        config = json.load(f)

    # 2) Dataset ID
    dataset_id = config.get('dataset_name')
    if not dataset_id:
        st.error("Dataset ID not found in config.json.")
        return None, None, None, None

    # 3) Load train dataset
    dataset_manager = st.session_state.dataset_manager
    train_dataset = dataset_manager.load_dataset(dataset_id=dataset_id, dataset_type='train')
    if train_dataset is None:
        st.error(f"Training dataset '{dataset_id}' not found.")
        return None, None, None, None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 1),
        shuffle=False
    )

    # 4) Load validation dataset
    val_dataset = dataset_manager.load_dataset(dataset_id=dataset_id, dataset_type='val')
    if val_dataset is None:
        st.error(f"Validation dataset '{dataset_id}' not found.")
        return None, None, None, None
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 1),
        shuffle=False
    )

    # 5) Model configs
    if 'model_i_config' not in config or 'model_a_config' not in config:
        st.error("Model configuration not found in config.json.")
        return None, None, None, None
    model_i_config = config['model_i_config']
    model_a_config = config['model_a_config']

    # 6) Load constants for atmospheric model
    try:
        possible_paths = [
            project_root / "data" / "constants",
            Path("../data/constants"),
            Path("../../data/constants"),
            Path("../../../data/constants"),
            Path("/mnt/lb_onesun_data/constants")
        ]
        data_dir = next(p for p in possible_paths if p.exists())
        gas_ods = config.get('gas_ods', 'gas_ods_conv05.pkl')
        solar_conv = config.get('solar_conv', 'solar_conv05.pkl')
        xsec_gas, le_solar, tau_rayleigh, _, _ = load_constants(
            data_dir=str(data_dir),
            gas_ods=gas_ods,
            solar_conv=solar_conv
        )
    except Exception as e:
        st.error(f"Error loading model constants: {e}")
        return None, None, None, None

    # 7) Initialize instrument model
    try:
        input_dim = model_i_config.get('input_dim', 2000)
        fwpos_dim = model_i_config.get('fwpos_dim', 4)
        output_dim = model_a_config.get('output_dim', 2000)
        model_i = instrument_model_chooser(config.get('instrument_model'), input_dim, fwpos_dim, output_dim)
    except Exception as e:
        st.error(f"Error initializing instrument model: {e}")
        return None, None, None, None

    # 8) Initialize atmospheric model
    try:
        model_a = atmospheric_model_chooser(
            config.get('atmospheric_model'),
            model_a_config.get('input_dim', 2000),
            model_a_config.get('latent_dim', 4),
            model_a_config.get('output_dim', 2000),
            xsec_gas.iloc[:, 1:], le_solar.iloc[:, 1:], tau_rayleigh
        )
    except Exception as e:
        st.error(f"Error initializing atmospheric model: {e}")
        return None, None, None, None

    # 9) Load weights
    models_dir = experiment_dir / "models"
    model_i_path = models_dir / "model_i_final.pth"
    model_a_path = models_dir / "model_a_final.pth"
    if not model_i_path.exists() or not model_a_path.exists():
        st.error("Model weight files not found in experiment directory.")
        return None, None, None, None
    try:
        model_i.load_state_dict(torch.load(model_i_path, map_location='cpu'))
        model_a.load_state_dict(torch.load(model_a_path, map_location='cpu'))
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None, None, None, None

    # 10) Return models + both loaders
    return model_i, model_a, val_loader, train_loader


# Model evaluation (Fixed: Use experiment dir as cache key)
@st.cache_data
def get_model_predictions(experiment_dir_str, dataset_type='val'):  # Add dataset_type parameter
    """Get model predictions for analysis"""
    temp_model_i, temp_model_a, temp_val_loader, temp_train_loader = load_saved_models(experiment_dir_str)
    if temp_model_i is None:
        return None, None
    
    # Choose the appropriate loader based on dataset_type
    if dataset_type == 'val':
        data_loader = temp_val_loader
    elif dataset_type == 'train':
        data_loader = temp_train_loader
    else:
        st.error(f"Invalid dataset_type: {dataset_type}")
        return None, None
    
    evaluator = ModelEvaluator(experiment_dir=Path(experiment_dir_str))
    results, avg_loss = evaluator.evaluate_models(
        temp_model_i, temp_model_a, data_loader
    )
    return results, avg_loss

# Clear cache when experiment changes to ensure correct data is loaded
def clear_model_cache_on_experiment_change():
    if 'last_selected_experiment' not in st.session_state:
        st.session_state['last_selected_experiment'] = str(experiment_dir)
    elif st.session_state['last_selected_experiment'] != str(experiment_dir):
        load_saved_models.clear()  # Clear the cache for the model loader
        get_model_predictions.clear()  # Also clear predictions cache
        st.session_state['last_selected_experiment'] = str(experiment_dir)

clear_model_cache_on_experiment_change()

# Load models with spinner (Fixed: Pass experiment dir for proper cache invalidation)
with st.spinner("🔄 Loading trained models..."):
    model_i, model_a, val_loader, train_loader = load_saved_models(str(experiment_dir))

if model_i is None or model_a is None or val_loader is None or train_loader is None:
    st.stop()

#st.success("✅ Models loaded successfully!")



# === TRAINING LOSS CURVES ===
st.header("Training Progress")

# Load experiment metrics for loss curves
metrics_path = experiment_dir / "experiment_metrics.json"

if metrics_path.exists():
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Create loss curves visualization
    if 'train_losses' in metrics and 'valid_losses' in metrics:
        train_losses = metrics['train_losses']
        valid_losses = metrics['valid_losses']
        
        if train_losses and valid_losses:
            # Create dataframe for plotting
            epochs = list(range(1, len(train_losses) + 1))
            loss_df = pd.DataFrame({
                'epoch': epochs + epochs,
                'loss': train_losses + valid_losses,
                'type': ['Training'] * len(epochs) + ['Validation'] * len(epochs)
            })
            
            # Create loss curve plot with OneSun theme
            fig_loss = px.line(
                loss_df, x='epoch', y='loss', color='type',
                markers=True,
                title=f'',
                labels={'epoch': 'Epoch', 'loss': 'Loss', 'type': 'Dataset'},
                color_discrete_map={
                    'Training': get_color('primary'),
                    'Validation': get_color('secondary')
                }
            )
            
            fig_loss.update_layout(
                height=400,
                xaxis=dict(dtick=1),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=get_color('text_primary'))
            )
            
            st.plotly_chart(fig_loss, use_container_width=True, key=f"loss_curves_{selected_experiment}")
            
        
            
        else:
            st.warning("No loss data available in experiment metrics.")
    else:
        st.warning("Training and validation losses not found in experiment metrics.")
else:
    st.warning("No experiment metrics file found. Train a model first in **🚀 Training → Training & Monitoring**.")

#st.markdown("---")

st.header("Pixel Overview")

col1, col2 = st.columns(2)

with col1:
    # Dataset selection for analysis
    dataset_choice = st.radio(
        "",
        options=['Validation Dataset', 'Training Dataset'],
        horizontal=True,
        help="Choose which dataset to analyze in the spectral overview below"
    )

    dataset_type = 'val' if dataset_choice == 'Validation Dataset' else 'train'

with col2:
    # Pixel selection
    pixel_idx = st.number_input(
        "Select pixel index:",
        min_value=0,
        max_value=2047,  # Assuming typical spectrometer size
        value=1000,
        help="Enter pixel index to analyze (typically 0-2047)"
    )

with st.spinner(f"🔄 Computing model predictions for {dataset_choice.lower()}..."):
    results, avg_loss = get_model_predictions(str(experiment_dir), dataset_type=dataset_type)

if not results:
    st.error(f"❌ No evaluation results available for {dataset_choice.lower()}.")
    st.stop()

# Create single pixel spectral plot
@st.cache_data
def create_single_pixel_spectral_plot(_results, pixel_idx, dataset_type, _experiment_name=None):
    """Create spectral plot for a single pixel showing L0, L1_i, and L1_a"""
    
    # Collect data for the selected pixel across all measurements
    all_L0 = []
    all_L1_i = []
    all_L1_a = []
    measurement_labels = []
    
    for sample_idx, r in enumerate(_results):
        L0_data = r['L0']
        L1_i_data = r['L1_i']
        L1_a_data = r['L1_a']
        
        # Convert to numpy if needed
        if torch.is_tensor(L0_data):
            L0_data = L0_data.detach().cpu().numpy()
        if torch.is_tensor(L1_i_data):
            L1_i_data = L1_i_data.detach().cpu().numpy()
        if torch.is_tensor(L1_a_data):
            L1_a_data = L1_a_data.detach().cpu().numpy()
        
        # Extract pixel values for all spectra in this sample
        for spectrum_idx in range(L0_data.shape[0]):
            if pixel_idx < L0_data.shape[1]:
                all_L0.append(L0_data[spectrum_idx, pixel_idx])
                all_L1_i.append(L1_i_data[spectrum_idx, pixel_idx])
                all_L1_a.append(L1_a_data[spectrum_idx, pixel_idx])
                measurement_labels.append(f"S{sample_idx}_M{spectrum_idx}")
    
    # Create the plot
    fig = go.Figure()
    
    # Add traces for L0, L1_i, and L1_a
    measurement_indices = list(range(len(all_L0)))
    
    fig.add_trace(
        go.Scatter(
            x=measurement_indices,
            y=all_L0,
            mode='lines+markers',
            name='L0 (Original)',
            line=dict(color=get_color('primary'), width=2),
            marker=dict(size=4)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=measurement_indices,
            y=all_L1_i,
            mode='lines+markers',
            name='L1_i (Instrument)',
            line=dict(color=get_color('secondary'), width=2),
            marker=dict(size=4)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=measurement_indices,
            y=all_L1_a,
            mode='lines+markers',
            name='L1_a (Atmospheric)',
            line=dict(color=get_color('accent'), width=2),
            marker=dict(size=4)
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"Spectral Analysis for Pixel {pixel_idx} ({dataset_type.upper()} Dataset)",
        xaxis_title="Measurement Index",
        yaxis_title="Intensity",
        height=500,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=get_color('text_primary')),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add hover information
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>" +
                      "Measurement: %{x}<br>" +
                      "Intensity: %{y:.6f}<extra></extra>"
    )
    
    return fig

# Generate and display the plot
fig_spectral = create_single_pixel_spectral_plot(
    results, pixel_idx, dataset_type, _experiment_name=selected_experiment
)
st.plotly_chart(fig_spectral, use_container_width=True, key=f"pixel_spectral_{selected_experiment}_{dataset_type}")


# === DETAILED SAMPLE ANALYSIS ===
st.header("Detailed Sample Analysis")

# Sample selection
num_samples = len(results)
if num_samples > 1:
    sample_idx = st.slider("Select Sample", 0, num_samples-1, 0)
else:
    sample_idx = 0

sample_results = results[sample_idx]

# Spectrum selection within sample
num_spectra = sample_results['L0'].shape[0]

# Select starting spectrum index for single plot display
if num_spectra > 1:
    spectrum_idx = st.slider("Select Spectrum Index", 0, num_spectra-1, int(num_spectra/2))
else:
    spectrum_idx = 0

# Extract spectrum data for the selected spectrum
L0_example = sample_results['L0'][spectrum_idx]
L1_i_example = sample_results['L1_i'][spectrum_idx]
L1_a_example = sample_results['L1_a'][spectrum_idx]

# Convert to numpy if needed
if torch.is_tensor(L0_example):
    L0_example = L0_example.detach().cpu().numpy()
if torch.is_tensor(L1_i_example):
    L1_i_example = L1_i_example.detach().cpu().numpy()
if torch.is_tensor(L1_a_example):
    L1_a_example = L1_a_example.detach().cpu().numpy()

# Create single plot like in Individual Spectrum Detailed Analysis
pixels = np.arange(len(L1_i_example))
residual = L1_a_example - L1_i_example

# Create the detailed analysis plot
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=[f'Sample {sample_idx} - Spectrum {spectrum_idx} ({dataset_type.upper()} Dataset)', 'Residuals'],
    vertical_spacing=0.12,
    row_heights=[0.7, 0.3]
)

# Top plot: Spectra
fig.add_trace(
    go.Scatter(
        x=pixels, y=L0_example,
        mode='lines', name='L0 (Original)', 
        line=dict(color=get_color('primary'), width=2)
    ), row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=pixels, y=L1_i_example,
        mode='lines', name='L1_i (Instrument)', 
        line=dict(color=get_color('secondary'), width=2)
    ), row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=pixels, y=L1_a_example,
        mode='lines', name='L1_a (Atmospheric)', 
        line=dict(color=get_color('accent'), width=2)
    ), row=1, col=1
)

# Bottom plot: Residuals
fig.add_trace(
    go.Scatter(
        x=pixels, y=residual,
        mode='lines', name='Residual (L1_a - L1_i)', 
        line=dict(color=get_color('info'), width=2)
    ), row=2, col=1
)

# Update layout
fig.update_layout(
    height=600,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color=get_color('text_primary')),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# Update axes
fig.update_xaxes(title_text="Pixel Index", row=2, col=1)
fig.update_yaxes(title_text="Intensity", row=1, col=1)
fig.update_yaxes(title_text="Residual", row=2, col=1)

# Display the plot
st.plotly_chart(fig, use_container_width=True, key=f"simplified_detailed_spectrum_{spectrum_idx}_{selected_experiment}_{dataset_type}")



# === LATENT SPACE ANALYSIS ===
st.header("Latent Space Analysis")

# Get latent space variables for the current sample
with st.spinner("Computing latent space variables..."):
    model_a.eval()
    with torch.no_grad():
        # Get L1_i for all spectra in current sample
        L1_i_sample = sample_results['L1_i']
        latent_space = model_a.encoder(L1_i_sample)

    latent_space_np = latent_space.detach().cpu().numpy()
    amf_sample = sample_results['amf'].cpu().numpy()

# === LATENT VARIABLES OVER MEASUREMENTS ===

# Plot latent space variables with OneSun theme colors
@st.cache_data
def create_latent_space_plot(latent_space_np, _experiment_name=None):
    """Create latent space visualization with OneSun theme colors"""
    n_dims = latent_space_np.shape[1]
    n_measurements = latent_space_np.shape[0]
    measurement_indices = np.arange(n_measurements)

    if n_dims == 1:
        # Single plot for 1D latent space
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=measurement_indices,
            y=latent_space_np[:, 0],
            mode='lines+markers',
            marker=dict(color=get_color('primary'), size=6),
            line=dict(color=get_color('primary')),
            name='Latent Variable 1'
        ))
        fig.update_layout(
            title="",
            xaxis_title="Measurement Index",
            yaxis_title="Latent Variable 1",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=get_color('text_primary'))
        )
    else:
        # Multiple subplots for multi-dimensional latent space - max 5 plots side by side
        n_cols = min(5, n_dims)
        n_rows = (n_dims + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"Latent Variable {i+1}" for i in range(n_dims)],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )

        # OneSun theme colors for different latent variables
        latent_colors = [
            get_color('primary'),    # Teal blue
            get_color('secondary'),  # Golden yellow
            get_color('accent'),     # Orange
            get_color('info'),       # Blue
            get_color('success'),    # Green
            get_color('warning')     # Yellow/Orange
        ]

        for i in range(n_dims):
            row = i // n_cols + 1
            col = i % n_cols + 1
            color = latent_colors[i % len(latent_colors)]

            fig.add_trace(
                go.Scatter(
                    x=measurement_indices,
                    y=latent_space_np[:, i],
                    mode='lines+markers',
                    marker=dict(color=color, size=4),
                    line=dict(color=color),
                    name=f'Latent {i+1}',
                    showlegend=False
                ),
                row=row, col=col
            )

        fig.update_layout(
            title="",
            height=300 * n_rows,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=get_color('text_primary'))
        )

        # Update all x-axes
        for i in range(n_dims):
            row = i // n_cols + 1
            col = i % n_cols + 1
            fig.update_xaxes(title_text="Measurement Index", row=row, col=col)
            fig.update_yaxes(title_text=f"Latent {i+1}", row=row, col=col)

    return fig

fig_latent = create_latent_space_plot(latent_space_np, _experiment_name=selected_experiment)
st.plotly_chart(fig_latent, use_container_width=True, key=f"latent_space_{selected_experiment}_{sample_idx}")

# === LATENT SPACE VARIABLE MAPPING ===

with st.expander("🔍 **Latent Variable Mapping**", expanded=False):
    #st.subheader("Latent Space Variable Mapping")

    experiment_dir = Path(str(experiment_dir))

    # 1) Load configuration
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        st.error("No config.json found in experiment directory.")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load the gas cross-section dataframe to get the gas species names
    try:
        # Get the gas_ods filename from config
        gas_ods = config.get('gas_ods', 'gas_ods_conv05.pkl')
        # Find the data directory
        possible_paths = [
            project_root / "data" / "constants",
            Path("../data/constants"),
            Path("../../data/constants"),
            Path("../../../data/constants"),
            Path("/mnt/lb_onesun_data/constants")
        ]
        data_dir = next(p for p in possible_paths if p.exists())
        
        # Load the dataframe
        ods_conv_df = pd.read_pickle(data_dir / gas_ods)
        # Extract gas species names (excluding 'wl' column)
        gas_species = [col for col in ods_conv_df.columns if col != 'wl']
        
        # Create mapping table
        st.markdown("**Latent Variable → Gas Species Mapping:**")
        
        mapping_data = []
        for i, gas in enumerate(gas_species, 1):
            mapping_data.append({
                'Latent Variable': f'Variable {i}',
                'Gas Species': gas,
            })
        
        mapping_df = pd.DataFrame(mapping_data)
        # Display with custom column configuration
        st.dataframe(
            mapping_df, 
            use_container_width=False,
            column_config={
                "Latent Var": st.column_config.TextColumn(
                    "Latent Variable",
                    width="small"
                ),
                "Gas Species": st.column_config.TextColumn(
                    "Gas Species", 
                    width="medium"
                ),
                "Index": st.column_config.NumberColumn(
                    "Index",
                    width="small"
                )
            },
            hide_index=True
        )
            
        
    except Exception as e:
        st.error(f"Error loading gas species mapping: {e}")
        st.stop()