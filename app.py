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

# --- PASS 1: Extraction (Kinetic + Zoom) ---
def extract_kinetic_telemetry(video_path, progress_callback):
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    telemetry_data = []
    
    # ROI: Center Box
    roi_x1, roi_x2 = int(width * 0.25), int(width * 0.75)
    roi_y1, roi_y2 = int(height * 0.25), int(height * 0.75)
    mid_x = int((roi_x2 - roi_x1) / 2)
    
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
        
        # Dense Optical Flow
        flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        fx = flow[..., 0]
        fy = flow[..., 1]
        
        # 1. HORIZONTAL MOMENTUM
        mag = np.sqrt(fx**2 + fy**2)
        active_mask = mag > 1.0 
        if np.any(active_mask):
            med_x = np.median(fx[active_mask])
        else:
            med_x = 0.0
            
        # 2. ZOOM METRIC
        flow_left = fx[:, :mid_x]
        flow_right = fx[:, mid_x:]
        
        mask_l = np.abs(flow_left) > 0.5
        mask_r = np.abs(flow_right) > 0.5
        
        val_l = np.median(flow_left[mask_l]) if np.any(mask_l) else 0.0
        val_r = np.median(flow_right[mask_r]) if np.any(mask_r) else 0.0
        
        zoom_score = val_r - val_l

        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Flow_X": med_x,
            "Zoom_Score": zoom_score
        })
        
        prev_roi = curr_roi
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            progress_callback(frame_idx / total_frames)

    cap.release()
    return pd.DataFrame(telemetry_data), fps, width, height

# --- PASS 2: Analysis (Max Velocity Logic) ---
def analyze_states_max_vel(df, fps):
    # Smooth data
    window_slow = 15
    window_fast = 5
    
    if len(df) > window_slow:
        df['X_Smooth'] = savgol_filter(df['Flow_X'], window_slow, 3)
        df['Zoom_Smooth'] = savgol_filter(df['Zoom_Score'], window_fast, 3)
    else:
        df['X_Smooth'] = df['Flow_X']
        df['Zoom_Smooth'] = df['Zoom_Score']

    # Zoom Velocity (Derivative)
    df['Zoom_Velocity'] = np.gradient(df['Zoom_Smooth'])

    # --- 1. PIT STOP (Horizontal) ---
    x_mag = df['X_Smooth'].abs()
    MOVE_THRESH = x_mag.max() * 0.3 
    STOP_THRESH = x_mag.max() * 0.05 
    
    peaks, _ = find_peaks(x_mag, height=MOVE_THRESH, distance=fps*5)
    t_start, t_end = None, None
    
    if len(peaks) >= 2:
        arrival_idx = peaks[0]
        depart_idx = peaks[-1]
        
        start_idx = arrival_idx
        for i in range(arrival_idx, depart_idx):
            if x_mag.iloc[i] < STOP_THRESH:
                start_idx = i
                break
        
        end_idx = depart_idx
        for i in range(depart_idx, start_idx, -1):
            if x_mag.iloc[i] < STOP_THRESH:
                end_idx = i
                break
        
        t_start = df.iloc[start_idx]['Time']
        t_end = df.iloc[end_idx]['Time']
    
    # --- 2. JACKS (Global Min Velocity) ---
    t_up, t_down = None, None
    
    if t_start and t_end:
        # Creep Gate: Stop looking 1.0s before the car starts moving horizontally
        # This prevents departure artifacts from being seen as "Drops"
        t_creep_limit = t_end - 1.0
        
        stop_window = df[(df['Time'] >= t_start) & (df['Time'] <= t_creep_limit)]
        
        if not stop_window.empty:
            z_pos = stop_window['Zoom_Smooth'].values
            times = stop_window['Time'].values
            
            # A. FIND LIFT (First Expansion)
            peaks_up, _ = find_peaks(z_pos, height=0.2, distance=fps)
            if len(peaks_up) > 0:
                t_up = times[peaks_up[0]]
            else:
                t_up = t_start
                
            # B. FIND DROP (Max Contraction Velocity)
            # Look strictly AFTER the lift
            
            # Create a sub-window for the Drop Search
            drop_search_mask = (stop_window['Time'] > t_up + 1.0) 
            
            if np.any(drop_search_mask):
                drop_window = stop_window[drop_search_mask]
                z_vel_drop = drop_window['Zoom_Velocity'].values
                times_drop = drop_window['Time'].values
                
                # Find the single most violent contraction event
                # Ideally, this is the minimum value in the velocity array
                min_idx = np.argmin(z_vel_drop)
                min_val = z_vel_drop[min_idx]
                
                # Safety Check: It must be a "Negative" snap (Contraction)
                if min_val < -0.05: # Basic noise filter
                    t_down = times_drop[min_idx]
                else:
                    t_down = t_end # No distinct drop found
            else:
                t_down = t_end

    if t_up is None: t_up = t_start
    if t_down is None: t_down = t_end

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
        
        # TIMERS
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

        cv2.putText(frame, "Logic: Max Snap Velocity", (width-430, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            progress_callback(frame_idx / total_frames)
            
    cap.release()
    out.release()
    return temp_output.name

# --- Main ---
def main():
    st.title("🏁 Pit Stop Analyzer V23")
    st.markdown("### Max Velocity Drop Detection")
    st.info("Identifies the **Most Violent** contraction event between the Lift and Departure. This ignores small shakes and late drifts.")

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
            st.write("Step 1: Extracting Kinetic & Zoom Metrics...")
            df, fps, w, h = extract_kinetic_telemetry(tfile.name, bar.progress)
            
            st.write("Step 2: Finding Sharpest Drop...")
            t_start, t_end, (t_up, t_down) = analyze_states_max_vel(df, fps)
            timings = (t_start, t_end, t_up, t_down)
            
            if t_start is None:
                st.error("Could not detect distinct Arrival/Departure.")
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
        
        st.subheader("📊 Zoom Telemetry")
        base = alt.Chart(df).encode(x='Time')
        
        zvel = base.mark_line(color='magenta').encode(
            y=alt.Y('Zoom_Velocity', title='Zoom Vel (Low = Drop)')
        )
        zpos = base.mark_area(color='gray', opacity=0.3).encode(
            y=alt.Y('Zoom_Smooth', title='Zoom Pos')
        )
        
        rules = pd.DataFrame([
            {'t': t_up, 'c': 'green', 'l': 'Jacks Up'}, 
            {'t': t_down, 'c': 'red', 'l': 'Jacks Down'}
        ])
        rule_chart = alt.Chart(rules).mark_rule(strokeWidth=2).encode(x='t', color=alt.Color('c', scale=None))
        
        st.altair_chart((zpos + zvel + rule_chart).interactive(), use_container_width=True)
        
        st.subheader("Video Result")
        c1, c2 = st.columns([3,1])
        with c1:
            if os.path.exists(vid_path): st.video(vid_path)
        with c2:
            if os.path.exists(vid_path):
                with open(vid_path, 'rb') as f:
                    st.download_button("Download MP4", f, file_name="pitstop_final.mp4")

if __name__ == "__main__":
    main()
