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

# --- PASS 1: Extraction (Kinetic + Zoom + Vertical) ---
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
        
        # 1. GLOBAL MOMENTUM (Horizontal Stop)
        mag = np.sqrt(fx**2 + fy**2)
        active_mask = mag > 1.0 
        if np.any(active_mask):
            med_x = np.median(fx[active_mask])
            med_y = np.median(fy[active_mask])
        else:
            med_x = 0.0
            med_y = 0.0
            
        # 2. ZOOM METRIC (Radial Expansion)
        flow_left = fx[:, :mid_x]
        flow_right = fx[:, mid_x:]
        
        mask_l = np.abs(flow_left) > 0.5
        mask_r = np.abs(flow_right) > 0.5
        
        val_l = np.median(flow_left[mask_l]) if np.any(mask_l) else 0.0
        val_r = np.median(flow_right[mask_r]) if np.any(mask_r) else 0.0
        
        # Expansion = Right moves Right (+) - Left moves Left (-) = Positive
        zoom_score = val_r - val_l

        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Flow_X": med_x,
            "Flow_Y": med_y,
            "Zoom_Score": zoom_score
        })
        
        prev_roi = curr_roi
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            progress_callback(frame_idx / total_frames)

    cap.release()
    return pd.DataFrame(telemetry_data), fps, width, height

# --- PASS 2: Analysis ---
def analyze_states(df, fps):
    # Smooth data
    window_fast = 7
    window_slow = 15
    
    if len(df) > window_slow:
        df['X_Smooth'] = savgol_filter(df['Flow_X'], window_slow, 3)
        df['Y_Smooth'] = savgol_filter(df['Flow_Y'], window_fast, 3) 
        df['Zoom_Smooth'] = savgol_filter(df['Zoom_Score'], window_fast, 3)
    else:
        df['X_Smooth'] = df['Flow_X']
        df['Y_Smooth'] = df['Flow_Y']
        df['Zoom_Smooth'] = df['Zoom_Score']

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

    # --- 2. JACKS (Threshold-Gated Drop Detection) ---
    t_up, t_down = None, None
    
    if t_start and t_end:
        stop_window = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        
        if not stop_window.empty:
            z_sig = stop_window['Zoom_Smooth'].values
            y_sig = stop_window['Y_Smooth'].values
            
            # Calculate Dynamic Prominence based on signal strength
            max_z_amp = np.max(np.abs(z_sig))
            max_y_amp = np.max(np.abs(y_sig))
            
            # A. FIND LIFT (Expansion)
            # Look for Positive Zoom Peaks
            peaks_up, props_up = find_peaks(z_sig, height=max_z_amp*0.3, distance=fps)
            
            lift_intensity = 0.0
            
            if len(peaks_up) > 0:
                # First major expansion is the Lift
                first_peak_idx = peaks_up[0]
                t_up = stop_window.iloc[first_peak_idx]['Time']
                lift_intensity = z_sig[first_peak_idx] # Remember how strong the lift was
            else:
                # Fallback: If no lift detected, assume start
                t_up = t_start
                lift_intensity = max_z_amp # Assume standard intensity
                
            # B. FIND DROP (Contraction / Jolt)
            # The Drop must be significant. It cannot be a tiny vibration.
            # Threshold: Must be at least 40% as strong as the Lift (or Max signal)
            
            drop_threshold_z = lift_intensity * 0.4
            drop_threshold_y = max_y_amp * 0.4
            
            # 1. Contraction Candidates (Negative Zoom)
            peaks_contract, props_c = find_peaks(-z_sig, height=drop_threshold_z, distance=fps)
            
            # 2. Vertical Jolt Candidates (Heavy Y Movement)
            peaks_jolt, props_j = find_peaks(np.abs(y_sig), height=drop_threshold_y, distance=fps)
            
            valid_drops = []
            
            # Collect all valid heavy impacts after the Lift
            for p in peaks_contract:
                t_event = stop_window.iloc[p]['Time']
                if t_event > t_up + 1.0: # Must be at least 1s after lift
                    valid_drops.append(t_event)
                    
            for p in peaks_jolt:
                t_event = stop_window.iloc[p]['Time']
                if t_event > t_up + 1.0:
                    valid_drops.append(t_event)
            
            if valid_drops:
                # Sort and pick the LAST *valid* heavy impact.
                # Any small noise later in the video didn't pass the 'drop_threshold',
                # so we won't accidentally pick it.
                valid_drops.sort()
                t_down = valid_drops[-1]
            else:
                # If no heavy drop found, fallback to end
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

        cv2.putText(frame, "Logic: Lift(Exp) -> Heavy Impact(Drop)", (width-430, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            progress_callback(frame_idx / total_frames)
            
    cap.release()
    out.release()
    return temp_output.name

# --- Main ---
def main():
    st.title("🏁 Pit Stop Analyzer V18")
    st.markdown("### Magnitude-Gated Drop Detection")
    st.info("Filters out small vibrations. The 'Drop' must be a heavy impact (Contraction/Jolt) relative to the Lift intensity.")

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
            
            st.write("Step 2: Analyzing Stop/Jacks Logic...")
            t_start, t_end, (t_up, t_down) = analyze_states(df, fps)
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
        
        st.subheader("📊 Zoom & Jolt Telemetry")
        base = alt.Chart(df).encode(x='Time')
        
        # Zoom Metric
        zoom = base.mark_line(color='magenta').encode(
            y=alt.Y('Zoom_Smooth', title='Zoom (Up)')
        )
        
        # Vertical Jolt Metric
        jolt = base.mark_line(color='orange').encode(
            y=alt.Y('Y_Smooth', title='Vertical Jolt (Down)')
        )
        
        rules = pd.DataFrame([
            {'t': t_up, 'c': 'green', 'l': 'Jacks Up'}, 
            {'t': t_down, 'c': 'red', 'l': 'Jacks Down'}
        ])
        rule_chart = alt.Chart(rules).mark_rule(strokeWidth=2).encode(x='t', color=alt.Color('c', scale=None))
        
        st.altair_chart((zoom + jolt + rule_chart).interactive(), use_container_width=True)
        
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
