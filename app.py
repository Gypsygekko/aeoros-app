# app.py with heartbeat prints
import streamlit as st
import subprocess
import os

st.set_page_config(page_title="Aeoros | Motion Control", page_icon="🤖", layout="wide")

# ... (CSS styling code remains the same) ...

with st.sidebar:
    # ... (sidebar code remains the same) ...
    pass

st.title("Aeoros")
st.markdown("### *See the unseen. Control the uncreated.*")
st.write("---")

uploaded_file = st.file_uploader("Upload a video to begin analysis", type=["mp4"])

if uploaded_file is not None:
    left_col, mid_col, right_col = st.columns([1,2,1])
    with mid_col:
        st.subheader("Original Video")
        st.video(uploaded_file)
    
    if st.button("Generate Skeleton Video", use_container_width=True):
        st.write("---")
        st.write(" HEARTBEAT: Button clicked! Preparing to run analysis...") # <<< DEBUG PRINT
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Analyzing motion... This may take a minute."):
            command = ["python", "motion_analyzer.py", "temp_video.mp4"]
            st.write(f" HEARTBEAT: Running command: `{' '.join(command)}`") # <<< DEBUG PRINT
            result = subprocess.run(command, capture_output=True, text=True)
            st.write(" HEARTBEAT: Subprocess finished.") # <<< DEBUG PRINT

        output_video_path = "skeleton_video_final.mp4"
        
        if os.path.exists(output_video_path):
            st.success("Analysis Complete!")
            # ... (rest of the success logic) ...
        else:
            st.error("Processing failed. The skeleton video could not be created.")
            st.subheader("Error Log:")
            st.code(result.stderr, language="bash") # This will now show any errors
            
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")
