"""
Experiment Overview Page
Comprehensive overview of experiments including configuration, loss curves, and metrics
Migrated from report_v2.py
"""
import streamlit as st
import pandas as pd
import json
import sys
import re
from pathlib import Path
from datetime import datetime
from utils.colors import apply_theme, get_color, colored_metric, status_badge

# Apply OneSun theme
apply_theme()

# Adjust path for project imports
project_root = Path(__file__).resolve().parents[3]
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Page header
st.title("📋 Experiment Overview")

# Directory for experiment runs (provided via session state)
experiment_runs_dir = st.session_state.experiment_runs_dir

# Helper to parse datetime from experiment folder name
def _parse_experiment_datetime(name: str) -> datetime:
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

# Gather and sort experiment directories
dirs = [d for d in experiment_runs_dir.iterdir() if d.is_dir()]
dirs.sort(key=lambda d: _parse_experiment_datetime(d.name), reverse=True)
if not dirs:
    st.info("🔍 **No experiments found.** Run a training first under 🚀 Training → Training & Monitoring.")
    st.stop()

# Build overview DataFrame entries
overview = []
def load_experiment_data(exp_dir: Path):
    cfg, m = {}, {}
    config_fp = exp_dir / 'config.json'
    metrics_fp = exp_dir / 'experiment_metrics.json'
    if config_fp.exists():
        cfg = json.loads(config_fp.read_text())
    if metrics_fp.exists():
        m = json.loads(metrics_fp.read_text())
    return cfg, m

for d in dirs:
    try:
        # Load configuration and metrics for each experiment
        cfg, m = load_experiment_data(d)
        overview.append({
            'Experiment': d.name,
            'Dataset': cfg.get('dataset_name', 'NA'),
            'Instrument Model': cfg.get('instrument_model', 'NA'),
            'Atmospheric Model': cfg.get('atmospheric_model', 'NA'),
            'Best Val Loss': m.get('best_val_loss'),
            'Epochs': m.get('epochs_completed')
        })
    except Exception as e:
        st.error(f"Error loading experiment {d.name}: {e}")
        continue

df = pd.DataFrame(overview)

# Display DataFrame with single-row selection
selected_rows = st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row"
)

# Determine selected experiment
if selected_rows['selection']['rows']:
    sel_idx = selected_rows['selection']['rows'][0]
    selected = df.iloc[sel_idx]['Experiment']
else:
    selected = st.session_state.get('selected_experiment', df.iloc[0]['Experiment'])

# Persist selection
st.session_state['selected_experiment'] = selected

# Load data for selected experiment
exp_dir = experiment_runs_dir / selected
config, metrics = load_experiment_data(exp_dir)

# Detail tabs
tabs = st.tabs(["Config", "Results & Metrics"])
with tabs[0]:
    st.subheader("Config")
    st.json(config)
with tabs[1]:
    st.subheader("Results & Metrics")
    st.json(
        # only the top-level values that aren't lists
        {k: v for k, v in metrics.items() if not isinstance(v, list)},
        expanded=True
    )

