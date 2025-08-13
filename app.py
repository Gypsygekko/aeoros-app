import streamlit as st
import subprocess
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Aeoros | Motion Control",
    layout="wide"
)

st.title("Aeoros")
st.write("Upload a video to generate a motion-control skeleton, ready for any generative AI workflow.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a video file (MP4 format)", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    # --- Analysis Trigger ---
    if st.button("Generate Skeleton Video"):
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # --- Run the Analyzer Script ---
        with st.spinner("Analyzing motion... This may take a minute."):
            # NOTE: motion_analyzer.py must be in the same folder as this app.py script.
            subprocess.run(["python", "motion_analyzer.py", "temp_video.mp4"])

        st.success("Analysis Complete!")

        # --- Display and Download Results ---
        output_video_path = "skeleton_video_final.mp4"
        
        if os.path.exists(output_video_path):
            with open(output_video_path, "rb") as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
                
                st.download_button(
                    label="Download Skeleton Video (.mp4)",
                    data=video_bytes,
                    file_name="aeoros_skeleton_output.mp4",
                    mime="video/mp4"
                )
        else:
            st.error("Processing failed. The skeleton video could not be created.")
            
        # Clean up temporary files
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")