import streamlit as st
import cv2
import tempfile
from analysis import calculate_pit_stop_time

st.set_page_config(page_title="Pit Stop Analysis", layout="wide")

st.title("🏎️ Pit Stop Analysis")
st.write("Upload an overhead video of a pit stop to calculate the total time.")

# 1. Video Upload
uploaded_file = st.file_uploader("Select an overhead video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    st.write("Analyzing video... Please wait.")
    
    try:
        # 2. Perform Analysis
        total_time, start_frame, end_frame, fps = calculate_pit_stop_time(video_path)
        
        # 3. Display Results
        st.success(f"**Total Pit Stop Time: {total_time:.2f} seconds**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pit Stop Start (seconds)", f"{start_frame / fps:.2f}")
        with col2:
            st.metric("Pit Stop End (seconds)", f"{end_frame / fps:.2f}")

        # Display the video with markers for start and end
        st.video(video_path, start_time=int(start_frame / fps))

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        st.error("Could not detect a valid pit stop. Please ensure the video shows the car stopping horizontally in the middle of the frame.")

else:
    st.info("Please upload a video to begin analysis.")
