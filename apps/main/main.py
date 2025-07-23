"""
OneSun Training & Analysis Platform
Main entry point for the multipage Streamlit application
"""
import streamlit as st
import sys
import base64
from pathlib import Path
import os
from streamlit.runtime.secrets import secrets_singleton

secrets_singleton._secrets = {
    "auth":  {
        "redirect_uri": os.getenv("REDIRECT_URI", False),
        "cookie_secret": os.getenv("COOKIE_SECRET", False),
        "google": {
            "client_id": os.getenv("CLIENT_ID", False),
            "client_secret": os.getenv("CLIENT_SECRET", False),
            "server_metadata_url": os.getenv("SERVER_METADATA_URL", False),
        },
    }
} | st.secrets.to_dict()


# Path structure: apps/main_platform/main.py -> project_root  
project_root = Path(__file__).resolve().parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import color utilities
from utils.colors import apply_theme, get_color

# Configure the main app
st.set_page_config(
    page_title="OneSun Platform",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply OneSun theme
apply_theme()
production = os.getenv("ENV", "development").lower() == "production"

if production and not st.experimental_user.is_logged_in:
    # Hide sidebar toggle and sidebar before login
    st.markdown("""<style>
        [data-testid="collapsedControl"], button[aria-label="Toggle sidebar"] { display: none !important; }
        section[data-testid="stSidebar"] { visibility: hidden !important; }
    </style>""", unsafe_allow_html=True)
    st.title("Authentication")
    if st.button("Authenticate"):
        st.login("google")
    st.stop()

if not production or  st.experimental_user.is_logged_in:
    def display_logo_in_sidebar():
        """Display logo in sidebar"""
        logo_path = Path(__file__).parent / "onesun_logo.png"
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()

        st.markdown(f"""
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_b64}" 
                 style="width: 140px; height: auto; margin-bottom: 15px;">
        </div>
        """, unsafe_allow_html=True)

    # Define pages organized by section
    dataset_pages = [
        st.Page("pages/01_📊_Dataset_Management.py", title="Dataset Management", icon="📊"),
    ]

    training_pages = [
        st.Page("pages/02_⚙️_Training_Config.py", title="Training Configuration", icon="⚙️"),
        st.Page("pages/03_🚀_Training_Monitor.py", title="Training & Monitoring", icon="🚀"),
    ]

    analysis_pages = [
        st.Page("pages/04_📋_Experiment_Overview.py", title="Experiment Overview", icon="📋"),
        st.Page("pages/05_📄_Training_Summary.py", title="Training_Summary", icon="📄"),
        #st.Page("pages/05_🎯_Predictions.py", title="Predictions & Residuals", icon="🎯"),
        st.Page("pages/06_🔍_Model_Inspection.py", title="Model Inspection", icon="🔍"),
    ]

    debug_pages = [
        st.Page("pages/07_🔧_Debug.py", title="System Debug", icon="🔧"),
    ]

    # Create navigation with sections
    pg = st.navigation({
        "📊 Data": dataset_pages,
        "🚀 Training": training_pages,
        "📈 Analysis": analysis_pages,
        "🔧 Debug": debug_pages
    })

    # Initialize shared session state and resources
    def init_session_state():
        """Initialize shared session state variables"""
        if "initialized" not in st.session_state:
            # Import here to avoid circular imports
            from utils.dataset_manager import DatasetManager

            # Initialize dataset manager
            if "dataset_manager" not in st.session_state:
                st.session_state.dataset_manager = DatasetManager()

            # Initialize training state
            if "training_config" not in st.session_state:
                from utils.initialization import get_default_training_config
                st.session_state.training_config = get_default_training_config()

            # Initialize other shared state
            if "experiment_runs_dir" not in st.session_state:
                env_exp_runs = os.getenv("EXPERIMENT_RUNS_DIR")
                if env_exp_runs:
                    exp_runs_path = Path(env_exp_runs)
                else:
                    exp_runs_path = project_root / "experiment_runs"
                st.session_state.experiment_runs_dir = exp_runs_path
                exp_runs_path.mkdir(exist_ok=True)

            # Mark as initialized
            st.session_state.initialized = True

    # Display header in sidebar
    with st.sidebar:
        display_logo_in_sidebar()
        if st.experimental_user.is_logged_in:
            if st.button("Logout"):
                st.logout()
        st.markdown(f"""
        <div style="
            text-align: center;
            color: {get_color('text_secondary')};
            font-size: 1rem;
            margin-bottom: 1.5rem;
        ">OneSun Platform</div>
        """, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Run the selected page
    pg.run()

    # Footer information in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"""
        <div style="color: {get_color('text_primary')}; font-weight: 600; margin-bottom: 0.5rem;">
        Platform Status
        </div>
        """, unsafe_allow_html=True)

        # Show current datasets count
        datasets = st.session_state.dataset_manager.list_datasets()
        st.markdown(f"""
        <div style="
            display: flex; 
            justify-content: space-between; 
            align-items: center;
            padding: 0.25rem 0;
            color: {get_color('text_secondary')};
        ">
            <span>Available Datasets:</span>
            <span style="
                background-color: {get_color('primary')}20;
                color: {get_color('primary')};
                padding: 0.2rem 0.5rem;
                border-radius: 10px;
                font-weight: bold;
                font-size: 0.9rem;
            ">{len(datasets)}</span>
        </div>
        """, unsafe_allow_html=True)

        # Show experiment runs count if available
        if st.session_state.experiment_runs_dir.exists():
            experiments = [d for d in st.session_state.experiment_runs_dir.iterdir() if d.is_dir()]
            st.markdown(f"""
            <div style="
                display: flex; 
                justify-content: space-between; 
                align-items: center;
                padding: 0.25rem 0;
                color: {get_color('text_secondary')};
            ">
                <span>Experiment Runs:</span>
                <span style="
                    background-color: {get_color('secondary')}20;
                    color: {get_color('secondary')};
                    padding: 0.2rem 0.5rem;
                    border-radius: 10px;
                    font-weight: bold;
                    font-size: 0.9rem;
                ">{len(experiments)}</span>
            </div>
            """, unsafe_allow_html=True)

        # Show current training status if available
        if "training_status" in st.session_state:
            status = st.session_state.training_status
            if status == "completed":
                status_color = get_color('success')
                status_text = "Training Complete"
            elif status == "running":
                status_color = get_color('info')
                status_text = "🚀 Training in Progress"
            elif status == "error":
                status_color = get_color('error')
                status_text = "Training Error"
            else:
                status_color = get_color('text_muted')
                status_text = "Ready to Train"

            st.markdown(f"""
            <div style="
                background-color: {status_color}20;
                color: {status_color};
                padding: 0.4rem 0.8rem;
                border-radius: 8px;
                font-weight: bold;
                font-size: 0.85rem;
                text-align: center;
                margin: 0.5rem 0;
                border: 1px solid {status_color}40;
            ">{status_text}</div>
            """, unsafe_allow_html=True)