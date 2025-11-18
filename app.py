import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
import altair as alt
from scipy.signal import find_peaks, savgol_filter
from ultralytics import YOLO

# --- Configuration & Setup ---
st.set_page_config(page_title="Pit Stop Analytics V3", layout="wide")

# --- Load AI Model (Cached) ---
@st.cache_resource
def load_yolo_model():
    # We use yolov8n (nano) for speed on CPU. 
    # You can change to 'yolov8s.pt' (small) for better accuracy if cloud resources permit.
    return YOLO('yolov8n.pt')

# --- Core Analysis Logic (Motion) ---
def extract_motion_profile(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    motion_scores = []
    prev_gray = None
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    status_text.text("Scanning video motion profile...")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        score = 0.0
        if prev_gray is not None:
            delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            score = np.sum(thresh)

        motion_scores.append(score)
        prev_gray = gray
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    return np.array(motion_scores), fps

def detect_events(motion_data, fps):
    # 1. Smoothing
    window_length = 15 
    smoothed = savgol_filter(motion_data, window_length, 3) if len(motion_data) > window_length else motion_data

    # 2. Find Stop Window (Plateau detection)
    mid_point = len(smoothed) // 2
    arrival_zone = smoothed[:mid_point]
    departure_zone = smoothed[mid_point:]
    
    threshold_arrival = (np.max(arrival_zone) if len(arrival_zone) else 1) * 0.3
    threshold_departure = (np.max(departure_zone) if len(departure_zone) else 1) * 0.3
    
    stop_start_idx = 0
    for i in range(len(arrival_zone)):
        if arrival_zone[i] < threshold_arrival and i > 10:
            stop_start_idx = i
            break
            
    stop_end_idx = len(smoothed) - 1
    for i in range(len(departure_zone)):
        if departure_zone[i] > threshold_departure:
            stop_end_idx = mid_point + i
            break

    # 3. Find Jacks (Peaks within window)
    pit_stop_window = smoothed[stop_start_idx:stop_end_idx]
    peaks, _ = find_peaks(pit_stop_window, prominence=np.max(arrival_zone)*0.1, distance=fps)
    
    jacks_up_idx = stop_start_idx
    jacks_down_idx = stop_end_idx
    
    if len(peaks) >= 1:
        global_peaks = peaks + stop_start_idx
        jacks_up_idx = global_peaks[0]
        jacks_down_idx = global_peaks[-1] if len(peaks) > 1 else global_peaks[0]

    return {
        "stop_start": stop_start_idx / fps,
        "jacks_up": jacks_up_idx / fps,
        "jacks_down": jacks_down_idx / fps,
        "stop_end": stop_end_idx / fps,
        "fps": fps
    }

# --- AI Visualizer Logic ---
def run_ai_analysis(video_path, start_sec, end_sec):
    model = load_yolo_model()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Jump to start time
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    st_frame = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        if not ret or curr_frame > end_frame:
            break
            
        # Run YOLO Inference
        # classes=[0, 2] filters for Person(0) and Car(2) only
        results = model.predict(frame, conf=0.3, classes=[0, 2], verbose=False)
        
        # Plot results on the frame
        annotated_frame = results[0].plot()
        
        # Convert to RGB for Streamlit
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(annotated_frame, caption=f"Time: {curr_frame/fps:.2f}s", use_column_width=True)

    cap.release()

# --- UI Structure ---
def main():
    st.title("🏎️ Pit Stop Analytics V3: AI Enhanced")
    
    uploaded_file = st.file_uploader("Upload Pitstop Video", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        # --- Tabs for workflow ---
        tab1, tab2 = st.tabs(["⏱️ Timing Analysis", "🤖 AI Object Detection"])
        
        # --- TAB 1: Timing Analysis (Previous Logic) ---
        with tab1:
            if "motion_data" not in st.session_state:
                st.session_state.motion_data = None
                st.session_state.fps = 30
                st.session_state.events = None

            if st.button("Run Timing Analysis"):
                motion_arr, fps = extract_motion_profile(tfile.name)
                events = detect_events(motion_arr, fps)
                st.session_state.motion_data = motion_arr
                st.session_state.fps = fps
                st.session_state.events = events

            if st.session_state.motion_data is not None:
                # ... (Same chart/slider logic as V2) ...
                ev = st.session_state.events
                fps = st.session_state.fps
                
                st.subheader("Verify Timeline")
                c1, c2, c3, c4 = st.columns(4)
                t_start = c1.number_input("Stop Start", value=float(ev["stop_start"]), step=0.1)
                t_jack_up = c2.number_input("Jacks Up", value=float(ev["jacks_up"]), step=0.1)
                t_jack_down = c3.number_input("Jacks Down", value=float(ev["jacks_down"]), step=0.1)
                t_end = c4.number_input("Stop End", value=float(ev["stop_end"]), step=0.1)
                
                # Store confirmed times for AI tab
                st.session_state.confirmed_times = (t_start, t_end)

                # Charting
                df = pd.DataFrame({"Time": np.arange(len(st.session_state.motion_data))/fps, "Motion": st.session_state.motion_data})
                
                base = alt.Chart(df).mark_line(color='#a0c4ff').encode(x='Time', y='Motion')
                rules_data = pd.DataFrame([
                    {"Time": t_start, "Event": "Start", "Color": "green"},
                    {"Time": t_jack_up, "Event": "Up", "Color": "orange"},
                    {"Time": t_jack_down, "Event": "Down", "Color": "orange"},
                    {"Time": t_end, "Event": "End", "Color": "red"},
                ])
                rules = alt.Chart(rules_data).mark_rule(strokeWidth=2).encode(x='Time', color=alt.Color('Color', scale=None))
                st.altair_chart((base + rules).interactive(), use_container_width=True)
                
                st.success(f"Total Pit Time: {t_end - t_start:.2f}s")

        # --- TAB 2: AI Object Detection ---
        with tab2:
            st.markdown("### Identify Crew & Cars")
            st.info("This uses YOLOv8 to detect people (Crew) and cars within the video.")
            
            if "confirmed_times" in st.session_state:
                start_t, end_t = st.session_state.confirmed_times
                # Add buffer
                start_viz = max(0, start_t - 2.0)
                end_viz = end_t + 2.0
            else:
                start_viz = 0.0
                end_viz = 10.0
            
            col_a, col_b = st.columns(2)
            s_time = col_a.number_input("Start Time (s)", value=float(start_viz), min_value=0.0)
            e_time = col_b.number_input("End Time (s)", value=float(end_viz), min_value=0.0)
            
            if st.button("Visualize with AI Overlay"):
                with st.spinner("Loading AI Model and Processing Frames..."):
                    run_ai_analysis(tfile.name, s_time, e_time)

        # Clean up handled by OS temp dir cleaning or manual if needed

if __name__ == "__main__":
    main()
