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

st.title("🎯 Predictions & Residuals")
st.markdown("Analyze model predictions, residuals, and performance across different spectral measurements.")

# Custom color scheme - named colors
CUSTOM_COLORS = {
    'goldenrod': 'palegoldenrod',               # Goldenrod
    'dark_orange': 'goldenrod',            # Dark orange
    'salmon_red': 'orangered',                 # Salmon red
    'dodger_blue': 'dodgerblue',            # Dodger blue
    'secondary': ['palegoldenrod', 'goldenrod', 'orangered', 'dodgerblue']
}

# Get experiment runs directory
experiment_runs_dir = st.session_state.experiment_runs_dir

def _parse_experiment_datetime(name: str) -> datetime:
    """Extract datetime from experiment folder name for sorting"""
    m = re.search(r'(\d{8}(?:[_-]?\d{6})?)', name)
    if not m:
        return datetime.min
    s = m.group(1).replace('_','').replace('-','')
    try:
        if len(s) == 8:
            return datetime.strptime(s, "%Y%m%d")
        elif len(s) == 14:
            return datetime.strptime(s, "%Y%m%d%H%M%S")
    except ValueError:
        pass
    return datetime.min

# Get experiments
if experiment_runs_dir.exists():
    experiments = [
        d.name for d in experiment_runs_dir.iterdir() 
        if d.is_dir()
    ]
    experiments.sort(key=_parse_experiment_datetime, reverse=True)
else:
    experiments = []

if not experiments:
    st.info("🔍 **No experiments found.** Complete a training run first in **🚀 Training → Training & Monitoring**.")
    st.stop()

# Experiment selection
selected_experiment = st.selectbox(
    "select experiment:",
    experiments,
    help="Select experiment for predictions analysis",
    key="experiment_selector"  # Fixed: Add key to ensure proper state management
)

if not selected_experiment:
    st.stop()

experiment_dir = experiment_runs_dir / selected_experiment

# Load saved models (with experiment-specific caching to fix update bug)
@st.cache_resource
def load_saved_models(experiment_dir_str):  # Use only experiment_dir_str as cache key
    """
    Load the model architecture and weights based on the saved configuration
    """
    from utils.initialization import load_constants
    from models.atmospheric_model import AtmosphericModel
    from utils.initialization import instrument_model_chooser
    
    experiment_dir = Path(experiment_dir_str)
    
    # Load configuration
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        st.error("No config.json found in experiment directory.")
        return None, None, None
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Get dataset ID from config
    dataset_id = config.get('dataset_name')
    if not dataset_id:
        st.error("Dataset ID not found in config.json.")
        return None, None, None
    
    # Load validation dataset
    dataset_manager = st.session_state.dataset_manager
    val_dataset = dataset_manager.load_dataset(dataset_id=dataset_id, dataset_type='val')
    if val_dataset is None:
        st.error(f"Validation dataset '{dataset_id}' not found.")
        return None, None, None
    
    # Create data loader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 1),
        shuffle=False
    )
    
    # Load model configs
    if 'model_i_config' not in config or 'model_a_config' not in config:
        st.error("Model configuration not found in config.json.")
        return None, None, val_loader
    
    model_i_config = config['model_i_config']
    model_a_config = config['model_a_config']
    
    # Load constants for atmospheric model
    try:
        possible_paths = [
            project_root / "data" / "constants",
            Path("../data/constants"),
            Path("../../data/constants"),
            Path("../../../data/constants"),
            Path("/mnt/lb_onesun_data/constants")
        ]
        
        data_dir = None
        for path in possible_paths:
            if path.exists():
                data_dir = path
                break
                
        if data_dir is None:
            st.error("Could not find the constants directory.")
            return None, None, val_loader

        gas_ods = config.get('gas_ods', 'gas_ods_conv05.pkl')
        solar_conv = config.get('solar_conv', 'solar_conv05.pkl')
        xsec_gas, le_solar, tau_rayleigh, _, _ = load_constants(
            data_dir=str(data_dir), 
            gas_ods=gas_ods, 
            solar_conv=solar_conv
        )
    except Exception as e:
        st.error(f"Error loading model constants: {e}")
        return None, None, val_loader
    
    # Initialize instrument model
    input_dim = model_i_config.get('input_dim', 2000)
    fwpos_dim = model_i_config.get('fwpos_dim', 4)
    output_dim = model_a_config.get('output_dim', 2000)
    instrument_model = config.get('instrument_model', 'Standard Instrument Model')
    atmospheric_model = config.get('atmospheric_model', 'Atmospheric Model')
    
    try:
        model_i = instrument_model_chooser(instrument_model, input_dim, fwpos_dim, output_dim)
    except Exception as e:
        st.error(f"Error initializing instrument model: {e}")
        return None, None, val_loader

    # Initialize atmospheric model
    try:
        model_a = atmospheric_model_chooser(atmospheric_model, model_a_config.get('input_dim', 2000), model_a_config.get('latent_dim', 4), model_a_config.get('output_dim', 2000),
                                        xsec_gas.iloc[:, 1:], le_solar.iloc[:, 1:], tau_rayleigh)
    except Exception as e:
        st.error(f"Error initializing atmospheric model: {e}")
        return None, None, val_loader
    
    # Load weights
    models_dir = experiment_dir / "models"
    model_i_path = models_dir / "model_i_final.pth"
    model_a_path = models_dir / "model_a_final.pth"
    
    if not model_i_path.exists() or not model_a_path.exists():
        st.error("Model weight files not found in experiment directory.")
        return None, None, val_loader
    
    try:
        model_i.load_state_dict(torch.load(model_i_path, map_location='cpu'))
        model_a.load_state_dict(torch.load(model_a_path, map_location='cpu'))
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None, None, val_loader
    
    return model_i, model_a, val_loader

# Model evaluation (Fixed: Use experiment dir as cache key)
@st.cache_data
def get_model_predictions(experiment_dir_str):  # Use experiment_dir_str as cache key
    """Get model predictions for analysis"""
    temp_model_i, temp_model_a, temp_val_loader = load_saved_models(experiment_dir_str)
    if temp_model_i is None:
        return None, None
    evaluator = ModelEvaluator(experiment_dir=Path(experiment_dir_str))
    results, avg_val_loss = evaluator.evaluate_models(
        temp_model_i, temp_model_a, temp_val_loader
    )
    return results, avg_val_loss

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
    model_i, model_a, val_loader = load_saved_models(str(experiment_dir))

if model_i is None or model_a is None or val_loader is None:
    st.stop()

st.success("✅ Models loaded successfully!")

with st.spinner("🔄 Computing model predictions..."):
    results, avg_val_loss = get_model_predictions(str(experiment_dir))

if not results:
    st.error("❌ No evaluation results available.")
    st.stop()

# === OVERVIEW SECTION ===
st.header("📊 Prediction Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🎯 Average Val Loss", f"{avg_val_loss:.6f}")

with col2:
    total_samples = len(results)
    st.metric("📊 Total Samples", str(total_samples))

with col3:
    # Calculate total spectra across all samples
    total_spectra = sum(r['L0'].shape[0] for r in results)
    st.metric("Total Spectra", str(total_spectra))

with col4:
    # Calculate average RMSE across all predictions
    rmse_values = []
    for r in results:
        L1_i = r['L1_i'].detach().cpu().numpy()
        L1_a = r['L1_a'].detach().cpu().numpy()
        rmse = np.sqrt(np.mean((L1_a - L1_i)**2))
        rmse_values.append(rmse)
    avg_rmse = np.mean(rmse_values)
    st.metric("📈 Average RMSE", f"{avg_rmse:.6f}")

# Complete fixes for 05_🎯_Predictions.py

# === SPECTRAL OVERVIEW BY PIXEL (Updated colors) ===
st.header("Spectral Overview by Pixel")

# Variable selection for overview
overview_var = st.selectbox(
    "Variable to display:",
    ["L0", "L1_i", "L1_a", "Residual"],
    help="Select which variable to visualize across all measurements"
)

# Add residual calculation if needed
if overview_var == "Residual":
    for r in results:
        r["Residual"] = r["L1_a"] - r["L1_i"]

# Pixel selection
pixel_input = st.text_input(
    "Pixel indices (comma-separated):",
    "500,1000,1500",
    help="Enter pixel indices to analyze (e.g., 500,1000,1500)"
)

try:
    pixel_indices = [int(p.strip()) for p in pixel_input.split(",") if p.strip().isdigit()]
except:
    pixel_indices = [500, 1000, 1500]  # Default

if pixel_indices:
    # Create pixel overview plot (with OneSun theme colors)
    @st.cache_data
    def create_pixel_overview(_results, pixel_indices, var, ncols=3, _experiment_name=None):
        """Create pixel overview visualization with OneSun theme colors"""
        n_pixels = len(pixel_indices)
        n_rows = (n_pixels + ncols - 1) // ncols

        fig = make_subplots(
            rows=n_rows, cols=ncols,
            subplot_titles=[f"Pixel {idx}" for idx in pixel_indices],
            vertical_spacing=0.1
        )

        # Use OneSun theme colors instead of custom colors
        theme_colors = [
            get_color('primary'),    # Teal blue
            get_color('secondary'),  # Golden yellow
            get_color('accent'),     # Orange
            get_color('info')        # Blue
        ]

        for i, pixel_idx in enumerate(pixel_indices):
            row = i // ncols + 1
            col = i % ncols + 1

            # Extract data for this pixel across all measurements
            pixel_data = []
            sample_indices = []

            for sample_idx, r in enumerate(_results):
                data = r[var]
                if torch.is_tensor(data):
                    data = data.detach().cpu().numpy()

                # Extract pixel values for all spectra in this sample
                for spectrum_idx in range(data.shape[0]):
                    if pixel_idx < data.shape[1]:
                        pixel_data.append(data[spectrum_idx, pixel_idx])
                        sample_indices.append(f"S{sample_idx}_M{spectrum_idx}")

            # Add trace for this pixel
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(pixel_data))),
                    y=pixel_data,
                    mode='lines+markers',
                    name=f'Pixel {pixel_idx}',
                    line=dict(color=theme_colors[i % len(theme_colors)]),
                    showlegend=i == 0  # Only show legend for first trace
                ),
                row=row, col=col
            )

        fig.update_layout(
            title=f"Pixel Overview: {var}",
            height=300 * n_rows,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=get_color('text_primary'))
        )

        return fig

    fig_overview = create_pixel_overview(results, pixel_indices, overview_var, _experiment_name=selected_experiment)
    st.plotly_chart(fig_overview, use_container_width=True, key=f"pixel_overview_{selected_experiment}")


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

# Show 3 plots side by side for quick overview
st.subheader("Model Predictions Overview")
st.markdown("Quick overview of 3 consecutive spectra for faster comparison:")

# Select starting spectrum index
if num_spectra >= 3:
    start_spectrum = st.slider("Starting Spectrum Index", 0, max(0, num_spectra-3), int(num_spectra/2))
    spectrum_indices = [start_spectrum, start_spectrum+1, start_spectrum+2]
else:
    # If less than 3 spectra, show what we have
    spectrum_indices = list(range(num_spectra))

# Create single legend at the top for all plots
st.markdown("""
<div style='text-align: center; margin-bottom: 1rem;'>
    <span style='color: {primary}; font-weight: bold;'>━━━</span> L0 (Original) &nbsp;&nbsp;
    <span style='color: {secondary}; font-weight: bold;'>━━━</span> L1_i (Instrument) &nbsp;&nbsp;
    <span style='color: {accent}; font-weight: bold;'>━━━</span> L1_a (Atmospheric) &nbsp;&nbsp;
    <span style='color: {info}; font-weight: bold;'>━━━</span> Residual
</div>
""".format(
    primary=get_color('primary'),
    secondary=get_color('secondary'),
    accent=get_color('accent'),
    info=get_color('info')
), unsafe_allow_html=True)

# Create 3 plots side by side
cols = st.columns(len(spectrum_indices))

for i, spectrum_idx in enumerate(spectrum_indices):
    if spectrum_idx < num_spectra:
        with cols[i]:
            # Extract spectrum data
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

            # Create compact predictions plot for this spectrum
            pixels = np.arange(len(L1_i_example))
            residual = L1_a_example - L1_i_example

            # Create subplots for this spectrum
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=[f'Spectrum {spectrum_idx}', 'Residuals'],
                vertical_spacing=0.15,
                row_heights=[0.7, 0.3]
            )

            # Top plot: Spectra (using OneSun theme colors, NO LEGEND)
            fig.add_trace(
                go.Scatter(x=pixels, y=L0_example, name='L0',
                          line=dict(color=get_color('primary'), width=1.5),
                          showlegend=False),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=pixels, y=L1_i_example, name='L1_i',
                          line=dict(color=get_color('secondary'), width=1.5),
                          showlegend=False),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=pixels, y=L1_a_example, name='L1_a',
                          line=dict(color=get_color('accent'), width=1.5),
                          showlegend=False),
                row=1, col=1
            )

            # Bottom plot: Residuals
            fig.add_trace(
                go.Scatter(x=pixels, y=residual, name='Residual',
                          line=dict(color=get_color('info'), width=1),
                          showlegend=False),
                row=2, col=1
            )

            # Add zero line for residuals
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

            # Calculate metrics for this spectrum
            rmse = np.sqrt(np.mean(residual**2))
            mae = np.mean(np.abs(residual))

            # Update layout for compact display
            fig.update_layout(
                height=500,
                showlegend=False,  # No legend on individual plots
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(size=10),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            fig.update_xaxes(title_text="Pixel", row=2, col=1)
            fig.update_yaxes(title_text="Intensity", row=1, col=1)
            fig.update_yaxes(title_text="Residual", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True, key=f"compact_pred_{sample_idx}_{spectrum_idx}")





# === LATENT SPACE VISUALIZATION SECTION (Updated colors) ===
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
            title="Latent Space Variables Over Measurements",
            xaxis_title="Measurement Index",
            yaxis_title="Latent Variable 1",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=get_color('text_primary'))
        )
    else:
        # Multiple subplots for multi-dimensional latent space
        n_cols = min(3, n_dims)
        n_rows = (n_dims + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"Latent Variable {i+1}" for i in range(n_dims)],
            vertical_spacing=0.1
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
            title="Latent Space Variables Over Measurements",
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
