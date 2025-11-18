import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
import altair as alt
from scipy.signal import find_peaks, savgol_filter

# --- Configuration & Setup ---
st.set_page_config(page_title="Pit Stop Analytics V2", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(BASE_DIR, "files", "refs")

# --- Helper Class (kept for future object detection expansion) ---
class ReferenceLoader:
    def __init__(self, base_path):
        self.paths = {
            "signboard": os.path.join(base_path, "signboard"),
            "emptyfuelport": os.path.join(base_path, "emptyfuelport"),
            "probein": os.path.join(base_path, "probein"),
            "probeout": os.path.join(base_path, "probeout"),
            "crew": os.path.join(base_path, "crew")
        }
    def load_images(self):
        # Placeholder for loading logic
        return {}

# --- Core Analysis Logic ---
def extract_motion_profile(video_path):
    """
    Scans the video and returns a raw motion score for every frame.
    Does NOT try to determine state yet.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    motion_scores = []
    prev_gray = None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Scanning video motion profile...")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        score = 0.0
        if prev_gray is not None:
            # Frame Differencing
            delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
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
    """
    Analyzes the motion curve to find:
    1. Stop Start (End of Arrival spike)
    2. Stop End (Start of Departure spike)
    3. Jacks Up (First major peak in stop window)
    4. Jacks Down (Last major peak in stop window)
    """
    
    # 1. Smooth the data to remove minor jitter
    # Window length must be odd and <= len(x)
    window_length = 15 
    if len(motion_data) > window_length:
        smoothed = savgol_filter(motion_data, window_length, 3)
    else:
        smoothed = motion_data

    # 2. Identify Arrival and Departure (The two massive outer mountains)
    # We look for the "Pit Stop Window" where motion is generally lower than the entry/exit
    # Heuristic: The stop usually happens in the middle 80% of the video
    
    mid_point = len(smoothed) // 2
    
    # Find the end of the arrival spike (Local minimum after the first massive peak)
    # We search from 0 to midpoint
    arrival_zone = smoothed[:mid_point]
    # The "Stop Start" is roughly where the massive arrival motion creates a 'knee' or drops significantly
    # Simple approach: Find when motion drops below 20% of the max arrival motion
    max_arrival = np.max(arrival_zone) if len(arrival_zone) > 0 else 1
    threshold_arrival = max_arrival * 0.3
    
    # Find index where it first drops below threshold
    stop_start_idx = 0
    for i in range(len(arrival_zone)):
        if arrival_zone[i] > threshold_arrival:
            # We are in the spike
            pass
        elif arrival_zone[i] < threshold_arrival and i > 10:
            # We dropped out of the spike
            stop_start_idx = i
            break
            
    # Find the start of the departure spike
    # Search from midpoint to end
    departure_zone = smoothed[mid_point:]
    max_departure = np.max(departure_zone) if len(departure_zone) > 0 else 1
    threshold_departure = max_departure * 0.3
    
    stop_end_idx = len(smoothed) - 1
    for i in range(len(departure_zone)):
        if departure_zone[i] > threshold_departure:
            # We hit the departure spike
            stop_end_idx = mid_point + i
            break

    # 3. Find Jacks (Peaks within the stop window)
    # We look strictly between stop_start_idx and stop_end_idx
    pit_stop_window = smoothed[stop_start_idx:stop_end_idx]
    
    # Find peaks in the window with some prominence
    peaks, properties = find_peaks(pit_stop_window, prominence=max_arrival*0.1, distance=fps)
    
    jacks_up_idx = stop_start_idx # Default fallback
    jacks_down_idx = stop_end_idx # Default fallback
    
    if len(peaks) >= 1:
        # Adjust peak indices to be global indices
        global_peaks = peaks + stop_start_idx
        # Assume First Peak = Jacks Up
        jacks_up_idx = global_peaks[0]
        # Assume Last Peak = Jacks Down (if multiple peaks exist)
        jacks_down_idx = global_peaks[-1] if len(peaks) > 1 else global_peaks[0]

    return {
        "stop_start": stop_start_idx / fps,
        "jacks_up": jacks_up_idx / fps,
        "jacks_down": jacks_down_idx / fps,
        "stop_end": stop_end_idx / fps,
        "fps": fps
    }

# --- UI Structure ---
def main():
    st.title("🏎️ Pit Stop Analytics V2")
    st.markdown("### Automated Event Detection with Manual Override")

    # Sidebar
    st.sidebar.header("Input")
    uploaded_file = st.file_uploader("Upload Pitstop Video", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Save temp file
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        # Layout
        col_vid, col_stats = st.columns([3, 2])
        
        with col_vid:
            st.video(uploaded_file)
        
        # Analyze Button
        if "motion_data" not in st.session_state:
            st.session_state.motion_data = None
            st.session_state.fps = 30
            st.session_state.events = None

        if st.sidebar.button("Run Analysis"):
            motion_arr, fps = extract_motion_profile(tfile.name)
            events = detect_events(motion_arr, fps)
            
            st.session_state.motion_data = motion_arr
            st.session_state.fps = fps
            st.session_state.events = events

        # --- Post-Analysis Interface ---
        if st.session_state.motion_data is not None:
            data = st.session_state.motion_data
            fps = st.session_state.fps
            ev = st.session_state.events
            
            # Create DataFrame for Charting
            time_axis = np.arange(len(data)) / fps
            df = pd.DataFrame({"Time": time_axis, "Motion": data})
            
            # --- Interactive Sliders for "Human in the Loop" ---
            st.divider()
            st.subheader("⏱️ Event Timeline verification")
            st.info("The algorithm has estimated the events. Adjust the sliders below to match the spikes exactly.")

            # Sliders initialized with auto-detected values
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                t_start = st.number_input("Stop Start (s)", 
                                          value=float(ev["stop_start"]), 
                                          step=0.1, format="%.2f",
                                          help="Car stops moving horizontally (approx 4.3s)")
            with c2:
                t_jack_up = st.number_input("Jacks Up (s)", 
                                            value=float(ev["jacks_up"]), 
                                            step=0.1, format="%.2f",
                                            help="First sharp spike (approx 6.4s)")
            with c3:
                t_jack_down = st.number_input("Jacks Down (s)", 
                                              value=float(ev["jacks_down"]), 
                                              step=0.1, format="%.2f",
                                              help="Drop spike (approx 22.5s)")
            with c4:
                t_end = st.number_input("Stop End (s)", 
                                        value=float(ev["stop_end"]), 
                                        step=0.1, format="%.2f",
                                        help="Car starts leaving (approx 47.3s)")

            # --- Visualization ---
            
            # Base Line Chart
            base = alt.Chart(df).mark_line(color='#a0c4ff').encode(
                x='Time',
                y='Motion',
                tooltip=['Time', 'Motion']
            ).properties(height=400)

            # Vertical Rules for Events
            rules_data = pd.DataFrame([
                {"Time": t_start, "Event": "Stop Start", "Color": "green"},
                {"Time": t_jack_up, "Event": "Jacks Up", "Color": "orange"},
                {"Time": t_jack_down, "Event": "Jacks Down", "Color": "orange"},
                {"Time": t_end, "Event": "Stop End", "Color": "red"},
            ])
            
            rules = alt.Chart(rules_data).mark_rule(strokeWidth=3).encode(
                x='Time',
                color=alt.Color('Color', scale=None),
                tooltip=['Event', 'Time']
            )
            
            # Text Labels for Rules
            text = alt.Chart(rules_data).mark_text(align='left', dx=5, dy=-100).encode(
                x='Time',
                text='Event',
                color=alt.value('white')
            )

            st.altair_chart((base + rules + text).interactive(), use_container_width=True)

            # --- Final Calculation Display ---
            
            total_pit_time = t_end - t_start
            tire_change_time = t_jack_down - t_jack_up
            
            st.markdown("### 🏁 Final Timing Stats")
            k1, k2, k3 = st.columns(3)
            k1.metric("Total Pit Time", f"{total_pit_time:.2f}s", delta="Stationary Time")
            k2.metric("Tire Change Time", f"{tire_change_time:.2f}s", delta="Jacks Up to Down")
            
            # Clean up
            os.remove(tfile.name)

if __name__ == "__main__":
    main()
