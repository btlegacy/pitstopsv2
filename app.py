import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
import altair as alt

# --- Configuration & Setup ---
st.set_page_config(page_title="Pit Stop Analytics", layout="wide")

# Define paths based on user prompt structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(BASE_DIR, "files", "refs")

# --- Class to handle Reference Images (Foundation for Future AI) ---
class ReferenceLoader:
    """
    Loads reference images for Signboard, Fuelport, Probe, and Crew.
    This prepares the app for future Object Detection or Template Matching.
    """
    def __init__(self, base_path):
        self.paths = {
            "signboard": os.path.join(base_path, "signboard"),
            "emptyfuelport": os.path.join(base_path, "emptyfuelport"),
            "probein": os.path.join(base_path, "probein"),
            "probeout": os.path.join(base_path, "probeout"),
            "crew": os.path.join(base_path, "crew")
        }
        self.references = {}

    def load_images(self):
        """Iterates through folders and loads images into memory"""
        for key, path in self.paths.items():
            self.references[key] = []
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img = cv2.imread(os.path.join(path, file))
                        if img is not None:
                            self.references[key].append(img)
        return self.references

# --- Core Analysis Logic ---
def analyze_pitstop(video_path, motion_threshold, min_stop_frames):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0: fps = 30 # Fallback

    frame_data = []
    prev_gray = None
    
    # State Machine Variables
    state = "IDLE" # IDLE -> ARRIVING -> STOPPED -> DEPARTING
    start_frame = None
    end_frame = None
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Preprocessing
        # Convert to grayscale for motion analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply GaussianBlur to reduce noise (fisheye grain)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        motion_score = 0.0

        if prev_gray is not None:
            # 2. Frame Differencing
            # Calculate absolute difference between current frame and previous frame
            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Dilate the thresholded image to fill in holes
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Count non-zero pixels (amount of motion)
            # We focus on the center of the frame where the car stops
            h, w = thresh.shape
            # ROI: Middle 60% of the screen horizontally
            roi = thresh[:, int(w*0.2):int(w*0.8)] 
            motion_score = np.sum(roi) / 255.0 # Normalize slightly

        prev_gray = gray
        
        # 3. State Machine Logic
        # "motion_threshold" is user configurable to tune sensitivity
        is_moving = motion_score > motion_threshold

        if state == "IDLE":
            if is_moving:
                state = "ARRIVING"
        
        elif state == "ARRIVING":
            if not is_moving:
                # Car has stopped moving
                state = "STOPPED"
                start_frame = frame_idx
        
        elif state == "STOPPED":
            if is_moving:
                # Check if it's just a momentary glitch or actual departure
                # (Simple debounce could go here, but we keep it simple for now)
                # We check if it has been stopped for a minimum duration to ignore jitters
                if (frame_idx - start_frame) > min_stop_frames:
                    state = "DEPARTING"
                    end_frame = frame_idx
        
        elif state == "DEPARTING":
            pass # We have our data, usually we can stop or continue to verify

        # Record data for charting
        frame_data.append({
            "Frame": frame_idx,
            "Time (s)": frame_idx / fps,
            "Motion Score": motion_score,
            "State": state
        })

        frame_idx += 1
        if frame_idx % 50 == 0:
            progress_bar.progress(frame_idx / total_frames)

    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(frame_data), start_frame, end_frame, fps

# --- UI Structure ---
def main():
    st.title("🏎️ Pit Stop Analysis V2")
    st.markdown("### Computer Vision Overhead Analysis")

    # Sidebar for Controls
    st.sidebar.header("Settings")
    
    # 1. Video Input
    uploaded_file = st.file_uploader("Upload Pitstop Video", type=["mp4", "mov", "avi"])
    
    # 2. Analysis Tuning
    st.sidebar.subheader("Algorithm Tuning")
    motion_thresh = st.sidebar.slider("Motion Threshold", 
                                      min_value=100, 
                                      max_value=50000, 
                                      value=5000, 
                                      step=100,
                                      help="How much pixel change is considered 'movement'? Lower = more sensitive.")
    
    min_stop_frames = st.sidebar.number_input("Min Stop Duration (Frames)", value=10)

    # 3. Reference Loading (Visualization only for now)
    loader = ReferenceLoader(REF_DIR)
    
    if st.sidebar.button("Reload Reference Images"):
        refs = loader.load_images()
        st.sidebar.success(f"Loaded {sum(len(v) for v in refs.values())} reference images.")

    if uploaded_file is not None:
        # Save uploaded file to temp for OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.video(uploaded_file)
        
        with col2:
            st.info("Video loaded. Click 'Analyze' to calculate pit stop time.")
            analyze_btn = st.button("Analyze Video", type="primary")

        if analyze_btn:
            with st.spinner("Processing frames..."):
                df, start_f, end_f, fps = analyze_pitstop(tfile.name, motion_thresh, min_stop_frames)
            
            # --- Results ---
            st.divider()
            st.subheader("🏁 Results")

            if start_f and end_f:
                total_time = (end_f - start_f) / fps
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Stop Time", f"{start_f / fps:.2f}s")
                m2.metric("Go Time", f"{end_f / fps:.2f}s")
                m3.metric("TOTAL PIT STOP", f"{total_time:.3f}s", delta_color="inverse")
                
                st.success(f"Car was stationary for {end_f - start_f} frames.")
            else:
                st.warning("Could not detect a clear Stop/Go cycle. Try adjusting the Motion Threshold slider.")

            # --- Debugging Visualization ---
            st.subheader("📊 Motion Telemetry")
            st.caption("Spikes indicate movement. The flat valley is the pit stop.")
            
            # Altair Chart for Motion
            c = alt.Chart(df).mark_line().encode(
                x='Time (s)',
                y='Motion Score',
                color='State',
                tooltip=['Frame', 'Time (s)', 'Motion Score', 'State']
            ).interactive()
            
            # Add threshold line
            rule = alt.Chart(pd.DataFrame({'y': [motion_thresh]})).mark_rule(color='red').encode(y='y')
            
            st.altair_chart(c + rule, use_container_width=True)

            # Clean up temp file
            os.remove(tfile.name)

if __name__ == "__main__":
    main()
