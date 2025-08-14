# Save this code as app.py (FINAL DEBUGGER VERSION)
import streamlit as st
import subprocess
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Aeoros | Motion Control",
    page_icon="🤖",
    layout="wide"
)

# --- Custom CSS for Branding and Layout ---
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
    st.title("Controls")
    st.header("1. Tuning")
    sensitivity = st.slider("Movement Sensitivity (Coming Soon)", min_value=1, max_value=100, value=50, help="This will control the level of detail in the analysis.")
    st.header("2. How to Use")
    st.info(
        "**For best results:**\n"
        "1.  Use a clear, well-lit video.\n"
        "2.  Ensure the subject's full body is visible.\n"
        "3.  Avoid baggy clothing that hides the body's form.\n"
        "4.  Fast, blurry motion may reduce accuracy."
    )

# --- Main Page ---
st.title("Aeoros")
st.markdown("### *See the unseen. Control the uncreated.*")
st.write("---")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload a video to begin analysis", type=["mp4"])

# --- Main Logic ---
if uploaded_file is not None:
    left_col, mid_col, right_col = st.columns([1,2,1])
    with mid_col:
        st.subheader("Original Video")
        st.video(uploaded_file)
    
    if st.button("Generate Skeleton Video", use_container_width=True):
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Analyzing motion... This may take a minute."):
            # <<< MODIFIED: This now captures all standard output and error messages >>>
            result = subprocess.run(
                ["python", "motion_analyzer.py", "temp_video.mp4"],
                capture_output=True,
                text=True
            )

        output_video_path = "skeleton_video_final.mp4"
        
        if os.path.exists(output_video_path):
            # If the file exists, the script was successful
            st.success("Analysis Complete!")
            with open(output_video_path, "rb") as video_file:
                video_bytes = video_file.read()
                with mid_col:
                    st.subheader("Generated Skeleton")
                    st.video(video_bytes)
                    st.download_button(
                        label="Download Skeleton Video (.mp4)",
                        data=video_bytes,
                        file_name="aeoros_skeleton_output.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
        else:
            # <<< NEW: If the file doesn't exist, display the captured error message >>>
            st.error("Processing failed. The skeleton video could not be created.")
            st.subheader("Error Log:")
            # The full output from the script, including prints and errors, will be in result.stdout and result.stderr
            st.code(result.stdout + "\n" + result.stderr, language="bash")
            
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")
