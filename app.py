# --- FINAL APP SCRIPT (with Full Error Logging) ---
import streamlit as st
import subprocess
import os

st.set_page_config(page_title="Aeoros | Motion Control", page_icon="🤖", layout="wide")

# Your CSS styling can remain here if you have it

with st.sidebar:
    st.title("Controls")
    st.info("Upload a video to generate a motion-control skeleton.")
    st.header("How to Use")
    st.write("For best results, use a clear, well-lit video where the subject's full body is visible.")

st.title("Aeoros")
st.markdown("### *See the unseen. Control the uncreated.*")
st.write("---")

uploaded_file = st.file_uploader("Upload a video to begin analysis", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)
    if st.button("Generate Skeleton Video"):
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Analyzing motion..."):
            # <<< MODIFIED: This now captures all standard output and error messages >>>
            result = subprocess.run(
                ["python", "motion_analyzer.py", "temp_video.mp4"],
                capture_output=True,
                text=True
            )

        output_video_path = "skeleton_video_final.mp4"
        
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            st.success("Analysis Complete!")
            with open(output_video_path, "rb") as video_file:
                st.video(video_file.read())
                st.download_button(
                    label="Download Skeleton Video (.mp4)",
                    data=video_file,
                    file_name="aeoros_skeleton_output.mp4",
                    mime="video/mp4"
                )
        else:
            # <<< NEW: If the file is empty, display the full log from the analyzer >>>
            st.error("Processing failed. The skeleton video could not be created.")
            st.subheader("Full Analyzer Log:")
            full_log = result.stdout + "\n" + result.stderr
            st.code(full_log, language="bash")
            
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")
