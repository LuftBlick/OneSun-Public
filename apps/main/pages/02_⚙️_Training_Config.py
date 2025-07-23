"""
Training Configuration Page
Configure training parameters, model selection, and experiment settings
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import os
from datetime import datetime

# Add src to path for imports
# Path structure: apps/main_platform/pages/02_... -> project_root
project_root = Path(__file__).resolve().parent.parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from utils.initialization import get_default_training_config
from utils.loss_functions import combined_loss, spectral_fidelity_loss, daily_correlation_loss
from utils.colors import apply_theme, get_color, colored_metric, status_badge

# Apply OneSun theme
apply_theme()

st.title("Training Configuration")
st.markdown("Configure training parameters, select models, and set up your experiment.")

# Get references from session state
dataset_manager = st.session_state.dataset_manager
training_config = st.session_state.training_config

# Check for available datasets
datasets = dataset_manager.list_datasets()
dataset_ids = [d[0] for d in datasets] if datasets else []

if not dataset_ids:
    st.error("**No datasets available!** Please create a dataset first in the Dataset Management page.")
    st.info("Go to **Data → Dataset Management** to create your first dataset.")
    st.stop()

# Main configuration form
#st.header("Experiment Configuration")

with st.form("training_config_form", clear_on_submit=False):
    
    # === EXPERIMENT BASICS ===
    st.subheader("Experiment Information")
    col1, col2 = st.columns(2)
    
    with col1:
        experiment_name = st.text_input(
            "Experiment Name", 
            value=training_config.get("experiment_name", "default_experiment"),
            help="Unique name for this training run"
        )
    
    with col2:
        selected_dataset = st.selectbox(
            "Dataset",
            dataset_ids,
            index=dataset_ids.index(training_config["dataset_name"]) if training_config["dataset_name"] in dataset_ids else 0,
            help="Select the dataset to use for training"
        )
    
    # Show dataset info
    if selected_dataset:
        dataset_metadata = dataset_manager.get_dataset_metadata(selected_dataset)
        if dataset_metadata:
            col1, col2, col3 = st.columns(3)
            with col1:
                train_range = dataset_metadata['config']['train_date_range']
                st.markdown(f"""
                <div style="
                    background-color: {get_color('primary')}20;
                    border: 1px solid {get_color('primary')}40;
                    border-radius: 6px;
                    padding: 0.75rem;
                    text-align: center;
                    color: {get_color('primary')};
                    font-weight: 500;
                ">
                    <div style="font-size: 0.9rem;">Training Period</div>
                    <div style="font-size: 1rem; font-weight: 600;">{train_range[0]} → {train_range[1]}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                val_range = dataset_metadata['config']['val_date_range']
                st.markdown(f"""
                <div style="
                    background-color: {get_color('secondary')}20;
                    border: 1px solid {get_color('secondary')}40;
                    border-radius: 6px;
                    padding: 0.75rem;
                    text-align: center;
                    color: {get_color('secondary')};
                    font-weight: 500;
                ">
                    <div style="font-size: 0.9rem;">Validation Period</div>
                    <div style="font-size: 1rem; font-weight: 600;">{val_range[0]} → {val_range[1]}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                source = Path(dataset_metadata['config'].get('source_file', 'Unknown')).name
                st.markdown(f"""
                <div style="
                    background-color: {get_color('info')}20;
                    border: 1px solid {get_color('info')}40;
                    border-radius: 6px;
                    padding: 0.75rem;
                    text-align: center;
                    color: {get_color('info')};
                    font-weight: 500;
                ">
                    <div style="font-size: 0.9rem;">Source File</div>
                    <div style="font-size: 1rem; font-weight: 600;">{source}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # === MODEL SELECTION ===
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get available models from your model files
        models_dir = src_path / "models"
        instrument_model_files = [f for f in os.listdir(models_dir) if f.startswith("instrument") and f.endswith(".py")]
        instrument_model_names = [f.replace(".py", "") for f in instrument_model_files]
        
        if not instrument_model_names:
            instrument_model_names = ["Standard Instrument Model"]  # Fallback
        
        selected_instrument_model = st.selectbox(
            "Instrument Model",
            instrument_model_names,
            index=0,
            help="Select the instrument model architecture"
        )
    
    with col2:
        # Atmospheric models
        models_dir = src_path / "models"
        atmospheric_model_files = [f for f in os.listdir(models_dir) if f.startswith("atmospheric") and f.endswith(".py")]
        atmospheric_model_names = [f.replace(".py", "") for f in atmospheric_model_files]

        selected_atmospheric_model = st.selectbox(
            "Atmospheric Model", 
            atmospheric_model_names,
            index=0,
            help="Select the atmospheric model architecture"
        )
    
    # === PHYSICAL CONSTANTS ===
    st.subheader("Physical Constants")
    
    constants_dir = project_root / "data" / "constants"
    if constants_dir.exists():
        constant_files = os.listdir(constants_dir)
        solar_files = [f for f in constant_files if "solar" in f and f.endswith('.pkl')]
        gas_files = [f for f in constant_files if "gas" in f and f.endswith('.pkl')]
    else:
        solar_files = ["solar_conv05.pkl"]  # Fallback
        gas_files = ["gas_ods_conv05.pkl"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_solar = st.selectbox(
            "Solar Spectrum File",
            solar_files,
            index=0,
            help="Reference solar spectrum for atmospheric modeling"
        )
    
    with col2:
        selected_gas_ods = st.selectbox(
            "Gas Cross-Section File",
            gas_files,
            index=0,
            help="Molecular absorption cross-sections"
        )
    
    st.markdown("---")
    
    # === TRAINING PARAMETERS ===
    st.subheader("Training Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.number_input(
            "Epochs", 
            min_value=1, max_value=1000, 
            value=training_config.get("epochs", 10),
            help="Number of training epochs"
        )
        
        batch_size = st.number_input(
            "Batch Size", 
            min_value=1, max_value=128, 
            value=training_config.get("batch_size", 1),
            help="Training batch size (1 = daily batches)"
        )
    
    with col2:
        learning_rate = st.number_input(
            "Learning Rate", 
            min_value=0.00001, max_value=0.1, 
            value=training_config.get("learning_rate", 1e-4), 
            format="%.5f",
            help="Optimizer learning rate"
        )
        
        weight_decay = st.number_input(
            "Weight Decay", 
            min_value=0.0, max_value=0.1, 
            value=training_config.get("weight_decay", 1e-4), 
            format="%.5f",
            help="L2 regularization strength"
        )
    
    with col3:
        # Loss function selection




        # Loss function parameters
        mse_loss_w = st.number_input(
            "Loss MSE Weight",
            min_value=0.0,
            value=training_config.get("mse_loss_w", 1.0),
            format="%.3f",
            help="Weight for correlation penalty"
        )
        corr_loss_w = st.number_input(
            "Loss Correlation Weight",
            min_value=0.0,
            value=training_config.get("corr_loss_w", 1.0),
            format="%.3f",
            help="Weight for daily correlation loss"
        )
        sam_loss_w = st.number_input(
            "Loss SAM Weight",
            min_value=0.0,
            value=training_config.get("sam_loss_w", 0.0),
            format="%.3f",
            help="Weight for spectral angle mapper loss"
        )
        cos_loss_w = st.number_input(
            "Loss Cosine Similarity Weight",
            min_value=0.0,
            value=training_config.get("cos_loss_w", 0.0),
            format="%.3f",
            help="Weight for cosine similarity loss"
        )




    st.markdown("---")
    
    # === ADVANCED SETTINGS (as normal options) ===
    st.subheader("Advanced Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        save_enabled = st.checkbox(
            "Save Models & Checkpoints", 
            value=training_config.get("save", True),
            help="Save trained models and training checkpoints"
        )
        
        early_stopping = st.checkbox(
            "Enable Early Stopping", 
            value=training_config.get("early_stopping", False),
            help="Stop training if validation loss stops improving"
        )
        
        if early_stopping:
            patience = st.number_input(
                "Patience (epochs)", 
                min_value=1, max_value=50, 
                value=training_config.get("patience", 5),
                help="Number of epochs to wait for improvement"
            )
        else:
            patience = 5
    
    with col2:
        # Advanced optimizer settings
        gradient_clipping = st.number_input(
            "Gradient Clipping", 
            min_value=0.1, max_value=10.0,
            value=training_config.get("gradient_clipping", 1.0),
            format="%.1f",
            help="Maximum gradient norm (prevents exploding gradients)"
        )
    
    # === EXPERIMENT VALIDATION ===
    st.markdown("---")
    st.subheader("Configuration Summary")
    
    # Create summary columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Data & Models**")
        st.markdown(f"- Dataset: `{selected_dataset}`")
        st.markdown(f"- Instrument: {selected_instrument_model}")
        st.markdown(f"- Atmospheric: {selected_atmospheric_model}")
    
    with col2:
        st.markdown("**Training Setup**")
        st.markdown(f"- Epochs: {epochs}")
        st.markdown(f"- Batch Size: {batch_size}")
        st.markdown(f"- Learning Rate: {learning_rate:.5f}")

    with col3:
        st.markdown("**Advanced**")
        st.markdown(f"- Save Models: {'Yes' if save_enabled else 'No'}")
        st.markdown(f"- Early Stopping: {'Yes' if early_stopping else 'No'}")
        if early_stopping:
            st.markdown(f"- Patience: {patience} epochs")
    
    # === SUBMIT BUTTON ===
    st.markdown("---")
    
    submitted = st.form_submit_button(
        "Save Configuration", 
        use_container_width=True,
        type="primary"
    )

# Handle form submission
if submitted:
    # Validate experiment name
    if not experiment_name.strip():
        st.error("Please provide an experiment name.")
    elif not selected_dataset:
        st.error("Please select a dataset.")
    else:
        # Update training configuration
        updated_config = {
            "experiment_name": experiment_name.strip(),
            "dataset_name": selected_dataset,
            "instrument_model": selected_instrument_model,
            "atmospheric_model": selected_atmospheric_model,
            "solar_conv": selected_solar,
            "gas_ods": selected_gas_ods,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "mse_loss_w": mse_loss_w,
            "corr_loss_w": corr_loss_w,
            "sam_loss_w": sam_loss_w,
            "cos_loss_w": cos_loss_w,
            "save": save_enabled,
            "early_stopping": early_stopping,
            "patience": patience if early_stopping else None,
            "gradient_clipping": gradient_clipping,
            "save_dir": str(st.session_state.experiment_runs_dir),
            "created_at": datetime.now().isoformat()
        }
        
        # Save to session state
        st.session_state.training_config = updated_config
        
        # Success message with OneSun styling
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {get_color('success')}20 0%, {get_color('primary')}20 100%);
            border: 1px solid {get_color('success')};
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            text-align: center;
            color: {get_color('success')};
            font-weight: 600;
        ">
            Configuration saved - ready to train!
        </div>
        """, unsafe_allow_html=True)
        
        
        

# === SIDEBAR INFO ===
with st.sidebar:
    st.markdown(f"""
    <div style="color: {get_color('text_primary')}; font-weight: 600; margin-bottom: 1rem;">
    Training Configuration
    </div>
    """, unsafe_allow_html=True)
    
    # Show current config status
    if st.session_state.training_config:
        config = st.session_state.training_config
        st.markdown("**Current Config:**")
        st.markdown(f"- Experiment: `{config.get('experiment_name', 'None')}`")
        st.markdown(f"- Dataset: `{config.get('dataset_name', 'None')}`")
        st.markdown(f"- Epochs: {config.get('epochs', 'None')}")
        
        # Handle loss function name safely
        loss_function = config.get('loss_function')
        if callable(loss_function):
            # It's a function object
            loss_name = loss_function.__name__
        elif isinstance(loss_function, str):
            # It's a string
            loss_name = loss_function
        else:
            loss_name = 'None'
        st.markdown(f"- Loss: {loss_name}")
        
        # Configuration completeness check
        required_fields = ['experiment_name', 'dataset_name', 'epochs', 'learning_rate']
        missing_fields = [field for field in required_fields if not config.get(field)]
        
        if not missing_fields:
            st.markdown(f"""
            <div style="
                background-color: {get_color('success')}20;
                color: {get_color('success')};
                padding: 0.5rem;
                border-radius: 6px;
                font-weight: 600;
                text-align: center;
                margin: 0.5rem 0;
            ">Configuration complete!</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="
                background-color: {get_color('warning')}20;
                color: {get_color('warning')};
                padding: 0.5rem;
                border-radius: 6px;
                font-weight: 600;
                text-align: center;
                margin: 0.5rem 0;
            ">Missing: {', '.join(missing_fields)}</div>
            """, unsafe_allow_html=True)