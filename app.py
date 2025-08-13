# Save this code as app.py (Version 3 - Custom Branded UI)
import streamlit as st
import subprocess
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Aeoros | Motion Control",
    page_icon="🤖",
    layout="wide"
)

# --- <<< NEW: Custom CSS for Branding and Layout >>> ---
st.markdown("""
<style>
    /* Main background and text color */
    .stApp {
        background-color: #0A192F;
        color: #EAEAEA;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a2a47;
    }
    /* Button styling */
    .stButton>button {
        background-color: #B5179E;
        color: #EAEAEA;
        border-radius: 8px;
        border: 2px solid #00F6FF;
    }
    .stButton>button:hover {
        background-color: #00F6FF;
        color: #0A192F;
        border: 2px solid #B5179E;
    }
    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        color: #00F6FF;
    }
    /* Center aligning main content */
    .block-container {
        padding-left: 10rem;
        padding-right: 10rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    # You can create a logo and upload it to your GitHub repo
    # st.image("logo.png", width=100) 
    st.title("Controls")
    
    st.header("1. Tuning")
    sensitivity = st.slider("Movement Sensitivity (Coming Soon)", 1, 100, 5
