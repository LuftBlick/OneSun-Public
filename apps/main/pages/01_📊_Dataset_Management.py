"""
Dataset Management Page
Handles dataset creation, viewing, and management
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from utils.daily_dataset import load_and_split_dataset
from utils.initialization import get_default_dataset_config

st.title("📊 Dataset Management")
st.markdown("View or create training datasets.")

# Get dataset manager from session state
dataset_manager = st.session_state.dataset_manager

# Create tabs for different dataset operations
tab1, tab2 = st.tabs(["📋 Existing Datasets", "➕ Create New Dataset"])

# ===== EXISTING DATASETS TAB =====
with tab1:
    st.header("Available Datasets")
    
    # Get and display existing datasets
    datasets = dataset_manager.list_datasets()
    
    if not datasets:
        st.info("🔍 No datasets found. Create your first dataset in the 'Create New Dataset' tab.")
    else:
        # Create a DataFrame to display dataset information
        dataset_info = []
        for dataset_id, info in datasets:
            dataset_info.append({
                "🆔 Dataset ID": dataset_id,
                "📅 Created": info['created_at'][:19].replace('T', ' '),
                "🚆 Train Period": f"{info['config']['train_date_range'][0]} → {info['config']['train_date_range'][1]}",
                "✅ Val Period": f"{info['config']['val_date_range'][0]} → {info['config']['val_date_range'][1]}",
                "📄 Source File": Path(info['config'].get('source_file', 'Unknown')).name
            })
        
        df_display = pd.DataFrame(dataset_info)
        
        # Display dataset table
        #st.dataframe(df_display, use_container_width=True, hide_index=True)
        # Display dataset table with selection
        selected_rows = st.dataframe(
            df_display, 
            use_container_width=True, 
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        print("DEBUG: Selected rows:", selected_rows)

        # Get the selected dataset ID
        selected_dataset = None
        if selected_rows['selection']['rows']:
            selected_row_idx = selected_rows['selection']['rows'][0]
            selected_dataset = [d[0] for d in datasets][selected_row_idx]

        # Dataset selection and details
        st.subheader("Dataset Details & Preview")
        
        #selected_dataset = st.selectbox(
        #    "🔍 Select dataset to inspect:",
        #    [d[0] for d in datasets],
        #    key="existing_dataset_selector"
        #)
        
        if selected_dataset:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Show metadata
                metadata = dataset_manager.get_dataset_metadata(selected_dataset)
                if metadata:
                    st.json(metadata, expanded=True)
                else:
                    st.warning("⚠️ No metadata available for this dataset.")
            
            with col2:
                # Dataset preview option
                show_preview = st.checkbox("📊 Show Dataset Preview", key="preview_checkbox")
                
                if show_preview:
                    with st.spinner("🔄 Loading dataset..."):
                        try:
                            train_dataset = dataset_manager.load_dataset(selected_dataset, dataset_type='train')
                            val_dataset = dataset_manager.load_dataset(selected_dataset, dataset_type='val')
                            
                            if train_dataset and val_dataset:
                                st.success(f"✅ Loaded: {len(train_dataset)} train days, {len(val_dataset)} val days")
                                
                                # Show tensor shapes
                                sample_spectrum, sample_amf, sample_fwpos, sample_tint = train_dataset[0]
                                
                                st.markdown("**🔢 Data Tensor Shapes:**")
                                st.markdown(f"- **Spectral data**: `{tuple(sample_spectrum.shape)}` (measurements × pixels)")
                                st.markdown(f"- **Air mass factors**: `{tuple(sample_amf.shape)}` (measurements,)")
                                st.markdown(f"- **Filter positions**: `{tuple(sample_fwpos.shape)}` (measurements × dimensions)")
                                st.markdown(f"- **Integration times**: `{tuple(sample_tint.shape)}` (measurements,)")
                        except Exception as e:
                            st.error(f"❌ Error loading dataset: {str(e)}")
            
            # Detailed day-by-day analysis
            if show_preview and 'train_dataset' in locals():
                st.markdown("---")
                st.subheader("📈 Daily Data Analysis")
                
                # Day selection
                day_labels = [d.strftime("%Y-%m-%d") for d in train_dataset.days]
                selected_day = st.selectbox("📅 Select day to analyze:", day_labels, key="day_selector")
                
                if selected_day:
                    # Extract data for selected day
                    spec_df = train_dataset.spectral_data.loc[selected_day]
                    amf_s = train_dataset.amf.loc[selected_day]
                    fw_df = train_dataset.fwpos_df.loc[selected_day]
                    tint_s = train_dataset.t_int.loc[selected_day]
                    
                    # Create three columns for time series plots
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        fig = px.line(
                            x=tint_s.index, y=tint_s.values,
                            labels={"x": "Time", "y": "Integration Time (s)"},
                            title="⏱️ Integration Time"
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.line(
                            x=amf_s.index, y=amf_s.values,
                            labels={"x": "Time", "y": "Air Mass Factor"},
                            title="🌍 Air Mass Factor"
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col3:
                        # Extract filter wheel positions
                        cols1 = sorted([c for c in fw_df.columns if c.startswith("fwpos1_")])
                        cols2 = sorted([c for c in fw_df.columns if c.startswith("fwpos2_")])
                        
                        if cols1 and cols2:
                            pos1 = fw_df[cols1].values.dot([int(c.split("_")[-1]) for c in cols1])
                            pos2 = fw_df[cols2].values.dot([int(c.split("_")[-1]) for c in cols2])
                            
                            fw_plot = pd.DataFrame({
                                "FW1": pos1, 
                                "FW2": pos2
                            }, index=fw_df.index)
                            
                            fig = px.line(
                                fw_plot,
                                labels={"index": "Time", "value": "Position", "variable": "Wheel"},
                                title="🎛️ Filter Wheel Positions"
                            )
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Spectrum visualization
                    st.markdown("---")
                    st.subheader("Spectral Data")
                    
                    measurement_idx = st.slider(
                        "📊 Select measurement:", 
                        0, len(spec_df) - 1, 0,
                        key="measurement_slider"
                    )
                    
                    if measurement_idx < len(spec_df):
                        spectrum = spec_df.iloc[measurement_idx]
                        timestamp = spectrum.name.strftime("%H:%M:%S")
                        
                        fig = px.line(
                            x=spectrum.index, y=spectrum.values,
                            labels={"x": "Pixel Index", "y": "Intensity"},
                            title=f"📡 Raw Spectrum at {timestamp}"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show measurement metadata
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("⏰ Time", timestamp)
                        with col2:
                            st.metric("🌍 AMF", f"{amf_s.iloc[measurement_idx]:.3f}")
                        with col3:
                            st.metric("⏱️ Int. Time", f"{tint_s.iloc[measurement_idx]:.1f}s")
                        with col4:
                            fw1_pos = pos1[measurement_idx] if 'pos1' in locals() else "N/A"
                            fw2_pos = pos2[measurement_idx] if 'pos2' in locals() else "N/A"
                            st.metric("🎛️ FW Pos", f"{fw1_pos}, {fw2_pos}")

# ===== CREATE NEW DATASET TAB =====
with tab2:
    st.header("➕ Create New Dataset")
    st.markdown("Configure and create a new training/validation dataset from raw L0 data.")
    
    # Get default configuration
    default_config = get_default_dataset_config()
    
    # Dataset creation form
    with st.form("dataset_creation_form"):
        # Basic information
        st.subheader("📝 Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_name = st.text_input(
                "🏷️ Dataset Name", 
                value=default_config["dataset_name"],
                help="Unique identifier for this dataset"
            )
        
        with col2:
            pickle_path = st.text_input(
                "📁 L0 Data File Path", 
                value="",
                placeholder="/path/to/your/data.pkl",
                help="Absolute path to the pickle file containing L0 data"
            )
        
        # Date ranges
        st.subheader("📅 Date Ranges")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🚆 Training Period**")
            train_start = st.date_input(
                "Start Date", 
                value=datetime.strptime(default_config["train_date_range"][0], "%Y-%m-%d"),
                key="train_start"
            )
            train_end = st.date_input(
                "End Date", 
                value=datetime.strptime(default_config["train_date_range"][1], "%Y-%m-%d"),
                key="train_end"
            )
        
        with col2:
            st.markdown("**✅ Validation Period**")
            val_start = st.date_input(
                "Start Date", 
                value=datetime.strptime(default_config["val_date_range"][0], "%Y-%m-%d"),
                key="val_start"
            )
            val_end = st.date_input(
                "End Date", 
                value=datetime.strptime(default_config["val_date_range"][1], "%Y-%m-%d"),
                key="val_end"
            )
        
        # Station parameters
        st.subheader("🗺️ Station Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            latitude = st.number_input(
                "🌍 Latitude (°)", 
                value=default_config["latitude"], 
                format="%.4f",
                help="Station latitude in decimal degrees"
            )
        
        with col2:
            longitude = st.number_input(
                "🌍 Longitude (°)", 
                value=default_config["longitude"], 
                format="%.4f",
                help="Station longitude in decimal degrees"
            )
        
        with col3:
            altitude = st.number_input(
                "⛰️ Altitude (m)", 
                value=default_config["altitude"],
                help="Station altitude above sea level in meters"
            )
        
        # Submit button
        submit_button = st.form_submit_button("🚀 Create Dataset", use_container_width=True)
    
    # Handle form submission
    if submit_button:
        # Validate inputs
        if not dataset_name:
            st.error("❌ Please provide a dataset name.")
        elif not pickle_path:
            st.error("❌ Please provide a path to the L0 data file.")
        elif not Path(pickle_path).exists():
            st.error(f"❌ File not found: {pickle_path}")
        elif train_start >= train_end:
            st.error("❌ Training start date must be before end date.")
        elif val_start >= val_end:
            st.error("❌ Validation start date must be before end date.")
        else:
            # Create the dataset
            config = {
                "dataset_name": dataset_name,
                "pickle_path": pickle_path,
                "train_date_range": [train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")],
                "val_date_range": [val_start.strftime("%Y-%m-%d"), val_end.strftime("%Y-%m-%d")],
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude
            }
            
            # Show progress and create dataset
            progress_container = st.container()
            
            with progress_container:
                with st.spinner("🔄 Creating dataset... This may take a while."):
                    try:
                        # Load and split dataset
                        train_dataset, val_dataset = load_and_split_dataset(
                            pickle_path=config["pickle_path"],
                            train_date_range=config["train_date_range"],
                            val_date_range=config["val_date_range"],
                            latitude=config["latitude"],
                            longitude=config["longitude"],
                            altitude=config["altitude"]
                        )
                        
                        # Save datasets
                        dataset_id = dataset_manager.save_datasets(
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            config=config,
                            friendly_name=dataset_name,
                            metadata={
                                "created_by": "OneSun Platform", 
                                "creation_date": datetime.now().isoformat(),
                                "station_coords": f"{latitude:.4f}, {longitude:.4f}",
                                "station_altitude": f"{altitude}m"
                            }
                        )
                        
                        st.success(f"✅ Dataset created successfully!")
                        st.info(f"🆔 Dataset ID: `{dataset_id}`")
                        
                        # Display dataset statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🚆 Training Days", len(train_dataset))
                        with col2:
                            st.metric("✅ Validation Days", len(val_dataset))
                        with col3:
                            # Get total measurements
                            total_measurements = sum(len(train_dataset.spectral_data.loc[d.strftime("%Y-%m-%d")]) 
                                                   for d in train_dataset.days)
                            st.metric("📊 Total Measurements", total_measurements)
                        
                        # Suggest next steps
                        st.markdown("---")
                        st.success("🎉 **Next Steps:** Your dataset is ready! You can now:")
                        st.markdown("- 🔧 Configure training parameters in the **Training Configuration** page")
                        st.markdown("- 📊 Inspect your data using the preview above")
                        st.markdown("- 🚀 Start training a model")
                        
                    except Exception as e:
                        st.error(f"❌ Error creating dataset: {str(e)}")
                        st.exception(e)

