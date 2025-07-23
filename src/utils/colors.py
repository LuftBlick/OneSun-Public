"""
Color utilities for OneSun platform
Self-contained color definitions and styling functions
"""
import streamlit as st

# OneSun Color Palette (extracted from logo)
COLORS = {
    "primary": "#4a90a4",          # Teal blue from instrument
    "secondary": "#daa520",        # Golden yellow from sun  
    "accent": "#ff6b35",           # Orange accent
    "light_blue": "#87ceeb",       # Sky blue
    "dark_blue": "#2f5f8f",        # Deep blue
    "sun_yellow": "#ffd700",       # Bright sun yellow
    "sun_orange": "#ff8c00",       # Warm sun orange
    "background": "#ffffff",
    "surface": "#f8f9fa",
    "surface_secondary": "#e9ecef",
    "text_primary": "#2c3e50",
    "text_secondary": "#6c757d",
    "text_muted": "#adb5bd",
    "success": "#28a745",
    "warning": "#ffc107",
    "error": "#dc3545",
    "info": "#17a2b8"
}

def get_color(color_name):
    """Get a color by name"""
    return COLORS.get(color_name, "#000000")

def get_custom_css():
    """Generate custom CSS using OneSun colors"""
    return f"""
    <style>
        /* OneSun Custom Theme */
        .main-header {{
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
        }}
        
        .logo-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 1rem;
        }}
        
        /* Primary button styling */
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['accent']} 100%);
            border: none;
            border-radius: 6px;
            color: white;
            font-weight: 600;
        }}
        
        .stButton > button[kind="primary"]:hover {{
            background: linear-gradient(135deg, {COLORS['accent']} 0%, {COLORS['primary']} 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        /* Success/info boxes */
        .success-box {{
            background-color: {COLORS['success']}20;
            border-left: 4px solid {COLORS['success']};
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 6px 6px 0;
        }}
        
        .info-box {{
            background-color: {COLORS['info']}20;
            border-left: 4px solid {COLORS['info']};
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 6px 6px 0;
        }}
        
        .warning-box {{
            background-color: {COLORS['warning']}20;
            border-left: 4px solid {COLORS['warning']};
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 6px 6px 0;
        }}
        
        /* Header styling */
        h1, h2, h3 {{
            color: {COLORS['text_primary']};
        }}
        
        /* Links */
        a {{
            color: {COLORS['primary']};
        }}
        
        a:hover {{
            color: {COLORS['accent']};
        }}
    </style>
    """

def apply_theme():
    """Apply OneSun theme to Streamlit app"""
    st.markdown(get_custom_css(), unsafe_allow_html=True)

def colored_metric(label, value, color_name="primary"):
    """Create a colored metric display"""
    color = get_color(color_name)
    st.markdown(f"""
    <div style="
        background: {get_color('surface')};
        border: 1px solid {get_color('surface_secondary')};
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
    ">
        <div style="color: {get_color('text_secondary')}; font-size: 0.9rem;">{label}</div>
        <div style="color: {color}; font-size: 1.5rem; font-weight: bold;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def status_badge(text, status="info"):
    """Create a status badge"""
    color = get_color(status)
    bg_color = f"{color}20"  # 20% opacity
    
    st.markdown(f"""
    <span style="
        background-color: {bg_color};
        color: {color};
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid {color};
    ">{text}</span>
    """, unsafe_allow_html=True)