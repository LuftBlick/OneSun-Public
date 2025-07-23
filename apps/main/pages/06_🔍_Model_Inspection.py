"""
Model Inspection Page
Detailed analysis of trained model parameters and architecture
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

# Add src to path for imports
# Path structure: apps/main_platform/pages/06_... -> project_root
project_root = Path(__file__).resolve().parent.parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from utils.model_eval import ModelEvaluator
from utils.loss_functions import combined_loss
from utils.dataset_manager import DatasetManager

st.title("🔍 Model Inspection")
#st.markdown("Dive into trained model parameters, weights, and architectural components.")

# Get experiment runs directory
experiment_runs_dir = st.session_state.experiment_runs_dir

# Get selected experiment from session state (set in 04_Experiment_Overview)
selected_experiment = st.session_state.get('selected_experiment')

if not selected_experiment:
    st.info("🔍 **No experiment selected.** Please select an experiment first in **📈 Analysis → Experiment Overview**.")
    st.stop()


experiment_dir = experiment_runs_dir / selected_experiment

# Get experiments
if experiment_runs_dir.exists():
    experiments = [
        d.name for d in experiment_runs_dir.iterdir() 
        if d.is_dir()
    ]
else:
    experiments = []
    
st.markdown(f"Current Experiment: {selected_experiment}")

# Load saved models (reuse from previous page but cache separately)
@st.cache_resource
def load_models_for_inspection(experiment_dir):
    """
    Load the model architecture and weights for inspection
    """
    from utils.initialization import load_constants
    from models.atmospheric_model import AtmosphericModel
    from utils.initialization import instrument_model_chooser
    from utils.initialization import atmospheric_model_chooser
    
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
    
    # Debug dataset loading
    available_datasets = dataset_manager.list_datasets()
    print(f"DEBUG: Available datasets: {[d[0] for d in available_datasets]}")
    print(f"DEBUG: Looking for dataset: {dataset_id}")
    
    val_dataset = dataset_manager.load_dataset(dataset_id=dataset_id, dataset_type='val')
    if val_dataset is None:
        available_names = [d[0] for d in available_datasets]
        st.error(f"❌ Validation dataset '{dataset_id}' not found.")
        st.info(f"📊 Available datasets: {available_names}")
        
        # Try to suggest similar names
        similar_datasets = [name for name in available_names if dataset_id.lower() in name.lower() or name.lower() in dataset_id.lower()]
        if similar_datasets:
            st.info(f"💡 Similar datasets found: {similar_datasets}")
        
        return None, None, None
    
    # Load model configs
    if 'model_i_config' not in config or 'model_a_config' not in config:
        st.error("Model configuration not found in config.json.")
        return None, None, None
    
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
            return None, None, None

        gas_ods = config.get('gas_ods', 'gas_ods_conv05.pkl')
        solar_conv = config.get('solar_conv', 'solar_conv05.pkl')
        xsec_gas, le_solar, tau_rayleigh, _, _ = load_constants(
            data_dir=str(data_dir), 
            gas_ods=gas_ods, 
            solar_conv=solar_conv
        )
    except Exception as e:
        st.error(f"Error loading model constants: {e}")
        return None, None, None
    
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
        return None, None, None

    # Initialize atmospheric model
    try:
        model_a = atmospheric_model_chooser(atmospheric_model, model_a_config.get('input_dim', 2000),
                                            model_a_config.get('latent_dim', 4),
                                            model_a_config.get('output_dim', 2000),
                                            xsec_gas.iloc[:, 1:], le_solar.iloc[:, 1:], tau_rayleigh)
    except Exception as e:
        st.error(f"Error initializing atmospheric model: {e}")
        return None, None, None
    
    # Load weights
    models_dir = experiment_dir / "models"
    model_i_path = models_dir / "model_i_final.pth"
    model_a_path = models_dir / "model_a_final.pth"
    
    if not model_i_path.exists() or not model_a_path.exists():
        st.error("Model weight files not found in experiment directory.")
        return None, None, None
    
    try:
        model_i.load_state_dict(torch.load(model_i_path, map_location='cpu'))
        model_a.load_state_dict(torch.load(model_a_path, map_location='cpu'))
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None, None, None
    
    return model_i, model_a, config

# Load models with spinner
with st.spinner("🔄 Loading trained models for inspection..."):
    model_i, model_a, config = load_models_for_inspection(experiment_dir)

if model_i is None or model_a is None:
    st.stop()

#st.success("✅ Models loaded successfully!")

# === MODEL ARCHITECTURE OVERVIEW ===
st.header("Model Architecture")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔧 Instrument Model")
    
    # Count parameters
    total_params_i = sum(p.numel() for p in model_i.parameters())
    trainable_params_i = sum(p.numel() for p in model_i.parameters() if p.requires_grad)
    
    st.markdown(f"**Total Parameters:** {total_params_i:,}")
    st.markdown(f"**Trainable Parameters:** {trainable_params_i:,}")
    
    # Model structure
    with st.expander("🔍 Model Structure", expanded=False):
        st.code(str(model_i), language="text")

with col2:
    st.subheader("🌍 Atmospheric Model")
    
    # Count parameters
    total_params_a = sum(p.numel() for p in model_a.parameters())
    trainable_params_a = sum(p.numel() for p in model_a.parameters() if p.requires_grad)
    
    st.markdown(f"**Total Parameters:** {total_params_a:,}")
    st.markdown(f"**Trainable Parameters:** {trainable_params_a:,}")
    
    # Model structure
    with st.expander("🔍 Model Structure", expanded=False):
        st.code(str(model_a), language="text")

# === PARAMETER VISUALIZATION FUNCTION ===
# Revised visualize_model_params for Model Inspection Page

# Revised visualize_model_params for Model Inspection Page

def visualize_model_params(model, model_name):
    """
    Visualize model parameters with all plots shown sequentially
    """
    st.header(f"📊 {model_name} Parameters")
    param_dict = dict(model.named_parameters())

    # 1D Parameters Section
    st.subheader("📈 1D Parameter Plots")
    params_1d = {name: p for name, p in param_dict.items() if p.dim() == 1}
    if params_1d:
        selected_1d = st.multiselect(
            "Select 1D parameters to visualize:",
            list(params_1d.keys()),
            default=list(params_1d.keys()),
            key=f"{model_name}_1d_params"
        )
        if selected_1d:
            cols1d = st.columns(len(selected_1d))
            for col, name in zip(cols1d, selected_1d):
                with col:
                    arr = params_1d[name].detach().cpu().numpy()
                    st.subheader(f"{name} (shape={arr.shape})")
                    fig = px.line(
                        x=np.arange(arr.shape[0]),
                        y=arr,
                        labels={"x": "Index", "y": "Value"},
                        title=name
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 1D parameters found in this model.")

    st.markdown("---")

    # 2D Parameters Section
    st.subheader("🔥 2D Parameter Heatmaps")
    params_2d = {name: p for name, p in param_dict.items() if p.dim() == 2}
    if params_2d:
        selected_2d = st.multiselect(
            "Select 2D parameters to visualize:",
            list(params_2d.keys()),
            default=list(params_2d.keys()),
            key=f"{model_name}_2d_params"
        )
        if selected_2d:
            cols2d = st.columns(len(selected_2d))
            for col, name in zip(cols2d, selected_2d):
                with col:
                    arr = params_2d[name].detach().cpu().numpy()
                    st.subheader(f"{name} (shape={arr.shape})")
                    fig = px.imshow(
                        arr,
                        labels={"x": "Column", "y": "Row", "color": "Value"},
                        aspect="auto",
                        title=name
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 2D parameters found in this model.")

    st.markdown("---")

    
    st.subheader("📊 Parameter Statistics")
    summary = []
    for name, p in param_dict.items():
        data = p.detach().cpu().numpy()
        summary.append({
            "Parameter": name,
            "Shape": str(data.shape),
            "Elements": data.size,
            "Mean": f"{data.mean():.6f}",
            "Std": f"{data.std():.6f}",
            "Min": f"{data.min():.6f}",
            "Max": f"{data.max():.6f}",
            "Trainable": p.requires_grad
        })
    df = pd.DataFrame(summary)
    st.dataframe(df, use_container_width=True, hide_index=True)


# === INSTRUMENT MODEL INSPECTION ===
visualize_model_params(model_i, "Instrument Model")

# === ATMOSPHERIC MODEL INSPECTION ===
visualize_model_params(model_a, "Atmospheric Model")


# Atmospheric model interpretation
if hasattr(model_a, 'Le') and hasattr(model_a, 'encoder'):
    st.subheader("🌞 Solar Spectrum (Le)")
    
    # Extract learned solar spectrum
    le_learned = model_a.Le.detach().cpu().numpy()
    
    # Plot solar spectrum
    fig = px.line(
        x=np.arange(len(le_learned)),
        y=le_learned,
        title="Learned Extraterrestrial Solar Spectrum",
        labels={"x": "Wavelength Index", "y": "Intensity"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Solar Intensity", f"{np.mean(le_learned):.6f}")
        st.metric("Solar Spectrum Range", f"{np.max(le_learned) - np.min(le_learned):.6f}")
    with col2:
        st.metric("Max Solar Intensity", f"{np.max(le_learned):.6f}")
        st.metric("Min Solar Intensity", f"{np.min(le_learned):.6f}")

# Cross-section analysis
if hasattr(model_a, 'xsec_list'):
    st.subheader("🧪 Gas Cross-Sections")
    
    # Get cross-section information
    try:
        # Load original cross-sections for comparison
        data_dir = project_root / "data" / "constants"
        gas_ods = config.get('gas_ods', 'gas_ods_conv05.pkl')
        solar_conv = config.get('solar_conv', 'solar_conv05.pkl')
        
        if data_dir.exists():
            xsec_gas, _, _, _, _ = load_constants(
                data_dir=str(data_dir), 
                gas_ods=gas_ods, 
                solar_conv=solar_conv
            )
            
            # Display gas species information
            gas_species = list(xsec_gas.columns[1:])  # Skip wavelength column
            st.markdown(f"**Available Gas Species:** {', '.join(gas_species)}")
            
            # Plot cross-sections
            selected_gas = st.selectbox(
                "Select gas species to visualize:",
                gas_species,
                key="gas_species_selector"
            )
            
            if selected_gas:
                gas_xsec = xsec_gas[selected_gas].values
                wavelengths = xsec_gas.iloc[:, 0].values
                
                fig = px.line(
                    x=wavelengths,
                    y=gas_xsec,
                    title=f"{selected_gas} Cross-Section",
                    labels={"x": "Wavelength (nm)", "y": "Cross-Section"}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Could not load original cross-sections for comparison: {e}")

# === MODEL COMPARISON SECTION ===
if len(experiments) > 1:
    st.header("Compare Model Parameters")
    
    # Select comparison experiment
    comparison_experiments = [exp for exp in experiments if exp != selected_experiment]
    selected_comparison = st.selectbox(
        "Select experiment to compare with:",
        ["None"] + comparison_experiments,
        key="comparison_experiment"
    )
    
    if selected_comparison != "None":
        with st.spinner("🔄 Loading comparison model..."):
            comp_model_i, comp_model_a, comp_config = load_models_for_inspection(
                experiment_runs_dir / selected_comparison
            )
        
        if comp_model_i is not None and comp_model_a is not None:
            st.subheader(f"📊 Comparison: {selected_experiment} vs {selected_comparison}")
            
            # Compare parameter counts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Current Experiment**")
                total_i = sum(p.numel() for p in model_i.parameters())
                total_a = sum(p.numel() for p in model_a.parameters())
                st.markdown(f"- Instrument Model: {total_i:,} params")
                st.markdown(f"- Atmospheric Model: {total_a:,} params")
                st.markdown(f"- **Total**: {total_i + total_a:,} params")
            
            with col2:
                st.markdown("**Comparison Experiment**")
                comp_total_i = sum(p.numel() for p in comp_model_i.parameters())
                comp_total_a = sum(p.numel() for p in comp_model_a.parameters())
                st.markdown(f"- Instrument Model: {comp_total_i:,} params")
                st.markdown(f"- Atmospheric Model: {comp_total_a:,} params")
                st.markdown(f"- **Total**: {comp_total_i + comp_total_a:,} params")
            
            # Parameter difference analysis
            if total_i == comp_total_i and total_a == comp_total_a:
                st.success("✅ Models have identical architectures - parameter values can be compared")
                
                # Compare specific parameters
                st.subheader("🔍 Parameter Value Comparison")
                
                # Get common parameter names
                current_params = dict(model_i.named_parameters())
                comp_params = dict(comp_model_i.named_parameters())
                common_params = set(current_params.keys()) & set(comp_params.keys())
                
                if common_params:
                    selected_param_comp = st.selectbox(
                        "Select parameter to compare:",
                        list(common_params),
                        key="param_comparison"
                    )
                    
                    if selected_param_comp:
                        current_param = current_params[selected_param_comp].detach().cpu().numpy()
                        comp_param = comp_params[selected_param_comp].detach().cpu().numpy()
                        
                        # Calculate difference
                        param_diff = current_param - comp_param
                        
                        # Plot comparison
                        if current_param.ndim == 1:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=current_param,
                                name=f"Current ({selected_experiment})",
                                line=dict(color='blue')
                            ))
                            fig.add_trace(go.Scatter(
                                y=comp_param,
                                name=f"Comparison ({selected_comparison})",
                                line=dict(color='red')
                            ))
                            fig.add_trace(go.Scatter(
                                y=param_diff,
                                name="Difference",
                                line=dict(color='green', dash='dash')
                            ))
                            fig.update_layout(
                                title=f"Parameter Comparison: {selected_param_comp}",
                                xaxis_title="Index",
                                yaxis_title="Value"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean Absolute Difference", f"{np.mean(np.abs(param_diff)):.6f}")
                        with col2:
                            st.metric("Max Absolute Difference", f"{np.max(np.abs(param_diff)):.6f}")
                        with col3:
                            st.metric("RMS Difference", f"{np.sqrt(np.mean(param_diff**2)):.6f}")
            else:
                st.warning("⚠️ Models have different architectures - direct parameter comparison not possible")

# === SIDEBAR INFORMATION ===
with st.sidebar:
    st.markdown("### 🔍 Model Inspection")
    
    st.markdown(f"**🔍 Current:** `{selected_experiment}`")
    
    # Model info
    if model_i and model_a:
        total_params = (sum(p.numel() for p in model_i.parameters()) + 
                       sum(p.numel() for p in model_a.parameters()))
        st.markdown(f"**🔢 Total Parameters:** {total_params:,}")
        
        # Model types
        instrument_model = config.get('instrument_model', 'Unknown')
        atmospheric_model = config.get('atmospheric_model', 'Unknown')
        st.markdown(f"**🔧 Instrument:** {instrument_model}")
        st.markdown(f"**🌍 Atmospheric:** {atmospheric_model}")
    
    