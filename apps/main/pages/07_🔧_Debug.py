"""
Debug Page
System diagnostics and troubleshooting utilities
"""
import streamlit as st
import pandas as pd
import sys
import os
import json
import torch
from pathlib import Path
from datetime import datetime

# Add src to path for imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

st.title("🔧 System Debug")
st.markdown("Diagnostic tools and system information for troubleshooting.")

# === SYSTEM INFORMATION ===
st.header("💻 System Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🗂️ Path Information")
    st.markdown(f"**Current File:** `{Path(__file__).resolve()}`")
    st.markdown(f"**Project Root:** `{project_root}`")
    st.markdown(f"**Src Path:** `{src_path}`")
    st.markdown(f"**Python Path:** `{sys.path[:3]}...`")

with col2:
    st.subheader("📁 Directory Status")
    st.markdown(f"**Project Root Exists:** {'✅' if project_root.exists() else '❌'}")
    st.markdown(f"**Src Path Exists:** {'✅' if src_path.exists() else '❌'}")
    
    # Check key directories
    data_dir = project_root / "data"
    constants_dir = data_dir / "constants"
    datasets_dir = data_dir / "processed_datasets"
    experiments_dir = project_root / "experiment_runs"
    
    st.markdown(f"**Data Dir:** {'✅' if data_dir.exists() else '❌'}")
    st.markdown(f"**Constants Dir:** {'✅' if constants_dir.exists() else '❌'}")
    st.markdown(f"**Datasets Dir:** {'✅' if datasets_dir.exists() else '❌'}")
    st.markdown(f"**Experiments Dir:** {'✅' if experiments_dir.exists() else '❌'}")

# === SESSION STATE DEBUG ===
st.header("🔄 Session State")

with st.expander("📋 Session State Contents", expanded=False):
    session_keys = list(st.session_state.keys())
    st.markdown(f"**Total Keys:** {len(session_keys)}")
    
    for key in sorted(session_keys):
        value = st.session_state[key]
        value_type = type(value).__name__
        
        if isinstance(value, (str, int, float, bool)):
            st.markdown(f"- **{key}** ({value_type}): `{value}`")
        elif hasattr(value, '__len__'):
            try:
                length = len(value)
                st.markdown(f"- **{key}** ({value_type}): Length {length}")
            except:
                st.markdown(f"- **{key}** ({value_type}): {str(value)[:50]}...")
        else:
            st.markdown(f"- **{key}** ({value_type}): {str(value)[:50]}...")

# === DATASET MANAGER DEBUG ===
st.header("📊 Dataset Manager Debug")

if "dataset_manager" in st.session_state:
    dm = st.session_state.dataset_manager
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗂️ Manager Info")
        st.markdown(f"**Base Directory:** `{dm.base_dir}`")
        st.markdown(f"**Directory Exists:** {'✅' if dm.base_dir.exists() else '❌'}")
        st.markdown(f"**Index File:** `{dm.index_file}`")
        st.markdown(f"**Index Exists:** {'✅' if dm.index_file.exists() else '❌'}")
        
        # Show index contents
        if hasattr(dm, 'index'):
            datasets_count = len(dm.index.get('datasets', {}))
            st.markdown(f"**Datasets in Index:** {datasets_count}")
    
    with col2:
        st.subheader("📋 Available Datasets")
        try:
            datasets = dm.list_datasets()
            if datasets:
                for dataset_id, info in datasets:
                    st.markdown(f"**{dataset_id}**")
                    st.markdown(f"  - Created: {info.get('created_at', 'Unknown')[:19]}")
                    st.markdown(f"  - Path: `{info.get('path', 'Unknown')}`")
                    
                    # Check if files exist
                    dataset_path = Path(info.get('path', ''))
                    if dataset_path.exists():
                        train_file = dataset_path / "train_dataset.pt"
                        val_file = dataset_path / "val_dataset.pt"
                        metadata_file = dataset_path / "metadata.json"
                        
                        st.markdown(f"  - Train: {'✅' if train_file.exists() else '❌'}")
                        st.markdown(f"  - Val: {'✅' if val_file.exists() else '❌'}")
                        st.markdown(f"  - Metadata: {'✅' if metadata_file.exists() else '❌'}")
                    else:
                        st.markdown(f"  - ❌ Directory not found")
                    st.markdown("---")
            else:
                st.info("No datasets found")
        except Exception as e:
            st.error(f"Error listing datasets: {e}")

# === DATASET TESTING ===
st.header("🧪 Dataset Testing")

if "dataset_manager" in st.session_state:
    dm = st.session_state.dataset_manager
    datasets = dm.list_datasets()
    
    if datasets:
        test_dataset_id = st.selectbox(
            "Select dataset to test:",
            [d[0] for d in datasets]
        )
        
        if st.button("🧪 Test Dataset Loading"):
            with st.spinner("Testing dataset loading..."):
                try:
                    # Test loading both splits
                    train_dataset = dm.load_dataset(test_dataset_id, dataset_type='train')
                    val_dataset = dm.load_dataset(test_dataset_id, dataset_type='val')
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if train_dataset:
                            st.success(f"✅ Train dataset loaded: {len(train_dataset)} samples")
                            
                            # Test sample access
                            try:
                                sample = train_dataset[0]
                                st.markdown("**Sample Structure:**")
                                for i, tensor in enumerate(sample):
                                    st.markdown(f"  - Tensor {i}: shape {tensor.shape}, dtype {tensor.dtype}")
                            except Exception as e:
                                st.error(f"Error accessing sample: {e}")
                        else:
                            st.error("❌ Failed to load train dataset")
                    
                    with col2:
                        if val_dataset:
                            st.success(f"✅ Val dataset loaded: {len(val_dataset)} samples")
                            
                            # Test sample access
                            try:
                                sample = val_dataset[0]
                                st.markdown("**Sample Structure:**")
                                for i, tensor in enumerate(sample):
                                    st.markdown(f"  - Tensor {i}: shape {tensor.shape}, dtype {tensor.dtype}")
                            except Exception as e:
                                st.error(f"Error accessing sample: {e}")
                        else:
                            st.error("❌ Failed to load val dataset")
                    
                except Exception as e:
                    st.error(f"Error during testing: {e}")
                    st.exception(e)

# === EXPERIMENT RUNS DEBUG ===
st.header("🧪 Experiment Runs Debug")

if "experiment_runs_dir" in st.session_state:
    exp_dir = st.session_state.experiment_runs_dir
    
    st.markdown(f"**Experiment Directory:** `{exp_dir}`")
    st.markdown(f"**Directory Exists:** {'✅' if exp_dir.exists() else '❌'}")
    
    if exp_dir.exists():
        experiments = [d for d in exp_dir.iterdir() if d.is_dir()]
        st.markdown(f"**Found Experiments:** {len(experiments)}")
        
        if experiments:
            for exp in experiments[:5]:  # Show first 5
                st.markdown(f"**{exp.name}**")
                
                # Check experiment structure
                config_file = exp / "config.json"
                models_dir = exp / "models"
                metrics_file = exp / "experiment_metrics.json"
                
                st.markdown(f"  - Config: {'✅' if config_file.exists() else '❌'}")
                st.markdown(f"  - Models: {'✅' if models_dir.exists() else '❌'}")
                st.markdown(f"  - Metrics: {'✅' if metrics_file.exists() else '❌'}")
                
                if models_dir.exists():
                    model_files = list(models_dir.glob("*.pth"))
                    st.markdown(f"  - Model Files: {len(model_files)}")

# === CONSTANTS DEBUG ===
st.header("📚 Constants Debug")

constants_dir = project_root / "data" / "constants"
if constants_dir.exists():
    st.success(f"✅ Constants directory found: `{constants_dir}`")
    
    const_files = list(constants_dir.glob("*.pkl"))
    st.markdown(f"**Pickle Files Found:** {len(const_files)}")
    
    for file in const_files:
        st.markdown(f"- `{file.name}` ({file.stat().st_size / 1024:.1f} KB)")

else:
    st.error(f"❌ Constants directory not found: `{constants_dir}`")

# === IMPORT TEST ===
st.header("📦 Import Testing")

imports_to_test = [
    "utils.dataset_manager",
    "utils.initialization", 
    "utils.daily_dataset",
    "utils.loss_functions",
    "utils.model_eval",
    "models.atmospheric_model",
    "models.instrument_model"
]

with st.expander("🧪 Test Imports", expanded=False):
    for import_name in imports_to_test:
        try:
            __import__(import_name)
            st.success(f"✅ {import_name}")
        except ImportError as e:
            st.error(f"❌ {import_name}: {e}")
        except Exception as e:
            st.warning(f"⚠️ {import_name}: {e}")

# === PYTORCH DEBUG ===
st.header("🔥 PyTorch Debug")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**PyTorch Version:** {torch.__version__}")
    st.markdown(f"**CUDA Available:** {'✅' if torch.cuda.is_available() else '❌'}")
    if torch.cuda.is_available():
        st.markdown(f"**CUDA Device Count:** {torch.cuda.device_count()}")

with col2:
    # Test tensor creation
    try:
        test_tensor = torch.randn(10, 10)
        st.success(f"✅ Tensor Creation: {test_tensor.shape}")
    except Exception as e:
        st.error(f"❌ Tensor Creation Failed: {e}")

# === ACTIONS ===
st.header("🔧 Diagnostic Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Refresh Session State"):
        st.rerun()

with col2:
    if st.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")

with col3:
    if st.button("📁 Create Missing Directories"):
        dirs_to_create = [
            project_root / "data" / "processed_datasets",
            project_root / "experiment_runs",
            project_root / "data" / "constants"
        ]
        
        created = []
        for dir_path in dirs_to_create:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                created.append(str(dir_path))
        
        if created:
            st.success(f"Created directories: {created}")
        else:
            st.info("All directories already exist")

# === SIDEBAR ===
with st.sidebar:
    st.markdown("### 🔧 Debug Tools")
    st.markdown("**System Status:**")
    
    # Quick status checks
    checks = [
        ("Dataset Manager", "dataset_manager" in st.session_state),
        ("Training Config", "training_config" in st.session_state),
        ("Experiment Dir", st.session_state.get("experiment_runs_dir", Path()).exists()),
        ("Constants Dir", (project_root / "data" / "constants").exists()),
    ]
    
    for name, status in checks:
        icon = "✅" if status else "❌"
        st.markdown(f"- {icon} {name}")
    
    st.markdown("---")
    st.markdown("**💡 Common Issues:**")
    st.markdown("- Dataset path resolution")
    st.markdown("- Missing constants files")
    st.markdown("- Session state corruption")
    st.markdown("- Import path problems")