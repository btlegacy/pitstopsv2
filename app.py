import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
import altair as alt
from scipy.signal import savgol_filter, find_peaks

# --- Configuration ---
st.set_page_config(page_title="Pit Stop Analytics AI", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- PASS 1: Extraction (Kinetic Flow) ---
def extract_kinetic_telemetry(video_path, progress_callback):
    """
    Extracts the raw kinetic energy (Optical Flow) of the scene.
    We focus on the 'Dominant Flow' to distinguish the massive car from small crew members.
    """
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    telemetry_data = []
    
    # ROI: Center Box (The "Kill Zone" where the car stops)
    # We make this tight to avoid the outer pit lane traffic
    roi_x1, roi_x2 = int(width * 0.25), int(width * 0.75)
    roi_y1, roi_y2 = int(height * 0.25), int(height * 0.75)
    
    ret, prev_frame = cap.read()
    if not ret: return pd.DataFrame(), fps, width, height
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_roi = prev_gray[roi_y1:roi_y2, roi_x1:roi_x2]
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Dense Optical Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        
        fx = flow[..., 0]
        fy = flow[..., 1]
        
        # --- KINETIC FILTERING ---
        # 1. Magnitude Filter: Ignore tiny jitters (asphalt grain)
        mag = np.sqrt(fx**2 + fy**2)
        active_mask = mag > 1.0 
        
        # 2. Dominant Direction Calculation
        # If the car is moving, the MEDIAN flow will be high.
        # If only crew is moving, the MEDIAN flow will be low (scattered).
        if np.any(active_mask):
            med_x = np.median(fx[active_mask])
            med_y = np.median(fy[active_mask])
            
            # Kinetic Energy (How "Heavy" is the movement?)
            # Sum of magnitudes gives us the total "Oomph" of the frame
            kinetic_energy = np.sum(mag[active_mask]) / (mag.size)
        else:
            med_x = 0.0
            med_y = 0.0
            kinetic_energy = 0.0

        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Flow_X": med_x,
            "Flow_Y": med_y,
            "Energy": kinetic_energy
        })
        
        prev_roi = curr_roi
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            progress_callback(frame_idx / total_frames)

    cap.release()
    return pd.DataFrame(telemetry_data), fps, width, height

# --- PASS 2: Analysis (State Machine) ---
def analyze_kinetic_states(df, fps):
    """
    State Machine:
    IDLE -> ARRIVAL (High Energy X) -> STOP (Energy Crash) -> DEPARTURE (High Energy X)
    """
    # Smooth data heavily to remove frame-by-frame jitter
    window = 15
    if len(df) > window:
        df['X_Smooth'] = savgol_filter(df['Flow_X'], window, 3)
        df['Y_Smooth'] = savgol_filter(df['Flow_Y'], window, 3)
        df['E_Smooth'] = savgol_filter(df['Energy'], window, 3)
    else:
        df['X_Smooth'] = df['Flow_X']
        df['Y_Smooth'] = df['Flow_Y']
        df['E_Smooth'] = df['Energy']

    # --- 1. DETECT ARRIVAL & DEPARTURE SPIKES ---
    # Use absolute X velocity because car might enter Left->Right or Right->Left
    x_mag = df['X_Smooth'].abs()
    
    # Thresholds
    # Arrival/Departure are massive events. 
    # Stop is a quiet event.
    MOVE_THRESH = x_mag.max() * 0.3  # 30% of max speed to trigger "Moving"
    STOP_THRESH = x_mag.max() * 0.05 # Must drop below 5% to be "Stopped"
    
    # Identify Regions of High Horizontal Movement
    is_moving_x = x_mag > MOVE_THRESH
    
    # Find peaks in movement (The center of the Arrival and Departure events)
    peaks, _ = find_peaks(x_mag, height=MOVE_THRESH, distance=fps*5)
    
    t_start, t_end = None, None
    
    if len(peaks) >= 2:
        # Assume First Peak = Arrival, Last Peak = Departure
        # (If there are middle peaks, they might be crew passing by camera, we ignore them for now)
        arrival_idx = peaks[0]
        depart_idx = peaks[-1]
        
        # --- FIND EXACT STOP TIME ---
        # Walk FORWARD from Arrival Peak until speed crashes to zero
        # We look for the moment X-Flow drops below STOP_THRESH
        start_idx = arrival_idx
        for i in range(arrival_idx, depart_idx):
            if x_mag.iloc[i] < STOP_THRESH:
                start_idx = i
                break
        
        # --- FIND EXACT GO TIME ---
        # Walk BACKWARD from Departure Peak until speed rises from zero
        end_idx = depart_idx
        for i in range(depart_idx, start_idx, -1):
            if x_mag.iloc[i] < STOP_THRESH:
                end_idx = i
                break
        
        t_start = df.iloc[start_idx]['Time']
        t_end = df.iloc[end_idx]['Time']
        
    else:
        # Fallback: Just find the quietest block in the middle
        # (Less accurate but works if thresholds failed)
        pass

    # --- 2. DETECT JACKS (Vertical Jolt) ---
    t_up, t_down = None, None
    
    if t_start and t_end:
        # Search strictly INSIDE the stop window
        stop_window = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        
        if not stop_window.empty:
            y_sig = stop_window['Y_Smooth'].values
            
            # Jacks cause a Vertical Jolt.
            # We look for the largest +Y and -Y deviations in the stop window.
            
            # Calculate deviation from mean (to handle slight camera tilt bias)
            y_dev = y_sig - np.mean(y_sig)
            
            # Find Peaks
            # A jack event is a sharp spike
            peaks_idx, _ = find_peaks(np.abs(y_dev), height=0.2, distance=fps*1)
            
            events = []
            for p in peaks_idx:
                events.append(stop_window.iloc[p]['Time'])
            
            if len(events) >= 1:
                t_up = events[0]      # First jolt
                t_down = events[-1]   # Last jolt
                
                # Edge case: If only 1 jolt, assumes Jacks Up. 
                # If no second jolt found, assume Jacks Down happened at Departure
                if len(events) == 1:
                    t_down = t_end

    return t_start, t_end, (t_up, t_down)

# --- PASS 3: Render ---
def render_overlay(input_path, timings, fps, width, height, progress_callback):
    t_start, t_end, t_up, t_down = timings
    cap = cv2.VideoCapture(input_path)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        current_time = frame_idx / fps
        
        # Timers
        val_pit, col_pit = (0.0, (200,200,200))
        if t_start and current_time >= t_start:
            if t_end and current_time >= t_end:
                val_pit = t_end - t_start
                col_pit = (0,0,255)
            else:
                val_pit = current_time - t_start
                col_pit = (0,255,0)

        val_tire, col_tire = (0.0, (200,200,200))
        if t_up and current_time >= t_up:
            if t_down and current_time >= t_down:
                val_tire = t_down - t_up
                col_tire = (0,0,255)
            else:
                val_tire = current_time - t_up
                col_tire = (0,255,255)
        
        # UI
        cv2.rectangle(frame, (width-450, 0), (width, 180), (0,0,0), -1)
        
        cv2.putText(frame, "PIT STOP", (width-430, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{val_pit:.2f}s", (width-180, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col_pit, 3)
        
        cv2.putText(frame, "TIRES (Jacks)", (width-430, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{val_tire:.2f}s", (width-180, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col_tire, 3)

        cv2.putText(frame, "Logic: Kinetic Momentum (X-Flow)", (width-430, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            progress_callback(frame_idx / total_frames)
            
    cap.release()
    out.release()
    return temp_output.name

# --- Main ---
def main():
    st.title("🏁 Pit Stop Analyzer V15")
    st.markdown("### Kinetic Momentum Logic")
    st.info("Detects the 'Crash' in Horizontal Velocity. Ignores background/lighting. Pure motion physics.")

    if 'analysis_done' not in st.session_state:
        st.session_state.update({'analysis_done': False, 'df': None, 'video_path': None, 'timings': None})

    uploaded_file = st.file_uploader("Upload Overhead Video", type=["mp4", "mov", "avi"])

    if uploaded_file and st.button("Start Analysis", type="primary"):
        st.session_state['analysis_done'] = False
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.flush()
        
        try:
            bar = st.progress(0)
            st.write("Step 1: Extracting Kinetic Energy...")
            df, fps, w, h = extract_kinetic_telemetry(tfile.name, bar.progress)
            
            st.write("Step 2: Finding Velocity Crash...")
            t_start, t_end, (t_up, t_down) = analyze_kinetic_states(df, fps)
            timings = (t_start, t_end, t_up, t_down)
            
            if t_start is None:
                st.error("Could not detect distinct Arrival and Departure spikes.")
                c = alt.Chart(df).mark_line().encode(x='Time', y='Flow_X')
                st.altair_chart(c, use_container_width=True)
            else:
                st.write("Step 3: Rendering Overlay...")
                vid_path = render_overlay(tfile.name, timings, fps, w, h, bar.progress)
                
                st.session_state.update({
                    'df': df, 'video_path': vid_path, 'timings': timings, 'analysis_done': True
                })
            bar.empty()
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists(tfile.name): os.remove(tfile.name)

    if st.session_state['analysis_done']:
        df = st.session_state['df']
        vid_path = st.session_state['video_path']
        t_start, t_end, t_up, t_down = st.session_state['timings']
        
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Pit Stop Time", f"{t_end - t_start:.2f}s")
        if t_up and t_down:
            c2.metric("Tire Change Time", f"{t_down - t_up:.2f}s")
        
        st.subheader("📊 Kinetic Telemetry")
        base = alt.Chart(df).encode(x='Time')
        
        # X Flow (The main signal)
        flow = base.mark_line(color='cyan').encode(
            y=alt.Y('X_Smooth', title='Horizontal Momentum')
        )
        
        # Y Flow (Jacks)
        vflow = base.mark_line(color='orange').encode(
            y=alt.Y('Y_Smooth', title='Vertical Momentum')
        )
        
        rules = pd.DataFrame([{'t': t_start, 'c': 'green'}, {'t': t_end, 'c': 'red'}])
        rule_chart = alt.Chart(rules).mark_rule(strokeWidth=2).encode(x='t', color=alt.Color('c', scale=None))
        
        st.altair_chart((flow + vflow + rule_chart).interactive(), use_container_width=True)
        
        st.subheader("Video Result")
        c1, c2 = st.columns([3,1])
        with c1:
            if os.path.exists(vid_path): st.video(vid_path)
        with c2:
            if os.path.exists(vid_path):
                with open(vid_path, 'rb') as f:
                    st.download_button("Download MP4", f, file_name="pitstop_kinetic.mp4")

if __name__ == "__main__":
    main()
