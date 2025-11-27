import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
import altair as alt
from scipy.signal import savgol_filter, find_peaks

# --- Configuration ---
st.set_page_config(page_title="Pit Stop Analytics - Simple", layout="wide")

# --- PASS 1: Global Extraction (Optical Flow) ---
def extract_telemetry(video_path, progress_callback):
    """
    Extracts Global Horizontal Flow (for Pit Stop) and Global Zoom (for Tires).
    No YOLO/Object Detection required.
    """
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    telemetry_data = []
    
    # Focus on Center 60% of screen to ignore pit lane traffic
    g_x1, g_x2 = int(width * 0.20), int(width * 0.80)
    g_y1, g_y2 = int(height * 0.20), int(height * 0.80)
    mid_x = int((g_x2 - g_x1) / 2)
    
    ret, prev_frame = cap.read()
    if not ret: return pd.DataFrame(), fps, width, height
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_roi = prev_gray[g_y1:g_y2, g_x1:g_x2]
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_roi = gray[g_y1:g_y2, g_x1:g_x2]
        
        # Dense Optical Flow
        flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx = flow[..., 0]
        fy = flow[..., 1]
        
        # 1. Horizontal Momentum (For Pit Stop Start/End)
        mag = np.sqrt(fx**2 + fy**2)
        active = mag > 1.0 
        flow_x = np.median(fx[active]) if np.any(active) else 0.0
        
        # 2. Radial Zoom (For Tires Up/Down)
        # Left side moves Left (-), Right side moves Right (+) = Expansion (Positive Zoom)
        f_left = fx[:, :mid_x]
        f_right = fx[:, mid_x:]
        
        val_l = np.median(f_left[np.abs(f_left)>0.5]) if np.any(np.abs(f_left)>0.5) else 0.0
        val_r = np.median(f_right[np.abs(f_right)>0.5]) if np.any(np.abs(f_right)>0.5) else 0.0
        
        zoom_score = val_r - val_l
        
        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Flow_X": flow_x,
            "Zoom_Score": zoom_score
        })
        
        prev_roi = curr_roi
        frame_idx += 1
        
        if frame_idx % 50 == 0: 
            progress_callback(frame_idx / total_frames)

    cap.release()
    return pd.DataFrame(telemetry_data), fps, width, height

# --- PASS 2: Logic Analysis ---
def analyze_simple_states(df, fps):
    if df.empty: return (None, None), (None, None)

    # Smoothing
    window = 15
    if len(df) > window:
        df['X_Sm'] = savgol_filter(df['Flow_X'], window, 3)
        df['Zoom_Sm'] = savgol_filter(df['Zoom_Score'], 5, 3) # Fast response for zoom
    else:
        df['X_Sm'] = df['Flow_X']
        df['Zoom_Sm'] = df['Zoom_Score']

    # Zoom Velocity (Derivative for Snap detection)
    df['Zoom_Vel'] = np.gradient(df['Zoom_Sm'])

    # 1. PIT STOP (Horizontal Stop)
    x_mag = df['X_Sm'].abs()
    # Find huge spikes for arrival/departure
    peaks, _ = find_peaks(x_mag, height=x_mag.max()*0.3, distance=fps*5)
    
    t_start, t_end = None, None
    
    if len(peaks) >= 2:
        arrival_idx = peaks[0]
        depart_idx = peaks[-1]
        
        # Stop threshold: 5% of max speed
        STOP_THRESH = x_mag.max() * 0.05
        
        for i in range(arrival_idx, depart_idx):
            if x_mag.iloc[i] < STOP_THRESH:
                t_start = df.iloc[i]['Time']
                break
        
        for i in range(depart_idx, arrival_idx, -1):
            if x_mag.iloc[i] < STOP_THRESH:
                t_end = df.iloc[i]['Time']
                break

    # 2. TIRES (Jacks Snap Logic)
    t_up, t_down = None, None
    
    if t_start and t_end:
        # Search Window: Stop Start -> 1.0s before Stop End
        # (Avoid confusing departure with jacks down)
        t_limit = t_end - 1.0
        stop_win = df[(df['Time'] >= t_start) & (df['Time'] <= t_limit)]
        
        if not stop_win.empty:
            z_pos = stop_win['Zoom_Sm'].values
            z_vel = stop_win['Zoom_Vel'].values
            times = stop_win['Time'].values
            
            # A. FIND LIFT (First Expansion)
            peaks_up, _ = find_peaks(z_pos, height=0.2, distance=fps)
            if len(peaks_up) > 0:
                t_up = times[peaks_up[0]]
            else:
                t_up = t_start # Fallback
                
            # B. FIND DROP (Sharpest Contraction)
            # Look for Minimum Velocity (Negative Spike) AFTER Lift
            mask_drop = times > t_up + 2.0 # Min 2s tire change
            
            if np.any(mask_drop):
                drop_vels = z_vel[mask_drop]
                drop_times = times[mask_drop]
                
                min_idx = np.argmin(drop_vels)
                min_val = drop_vels[min_idx]
                
                # Threshold for "Snap"
                if min_val < -0.02:
                    t_down = drop_times[min_idx]
                else:
                    t_down = t_end
            else:
                t_down = t_end

    if t_up is None: t_up = t_start
    if t_down is None: t_down = t_end

    return (t_start, t_end), (t_up, t_down)

# --- PASS 3: Render ---
def render_simple_overlay(input_path, pit_times, tire_times, fps, width, height, progress_callback):
    cap = cv2.VideoCapture(input_path)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
    
    t_start, t_end = pit_times
    t_up, t_down = tire_times
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        curr = frame_idx / fps
        
        # 1. Pit Timer
        if t_start and curr >= t_start:
            if t_end and curr >= t_end:
                val_pit = t_end - t_start
                col_pit = (0, 0, 255) # Red (Done)
            else:
                val_pit = curr - t_start
                col_pit = (0, 255, 0) # Green (Running)
        else:
            val_pit, col_pit = 0.0, (200, 200, 200)

        # 2. Tire Timer
        if t_up and curr >= t_up:
            if t_down and curr >= t_down:
                val_tire = t_down - t_up
                col_tire = (0, 0, 255)
            else:
                val_tire = curr - t_up
                col_tire = (0, 255, 255) # Yellow/Cyan (Running)
        else:
            val_tire, col_tire = 0.0, (200, 200, 200)
        
        # Draw
        box_w, box_h = 350, 120
        cv2.rectangle(frame, (width - box_w, 0), (width, box_h), (0, 0, 0), -1)
        
        cv2.putText(frame, "PIT STOP", (width - 330, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{val_pit:.2f}s", (width - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col_pit, 3)

        cv2.putText(frame, "TIRES", (width - 330, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{val_tire:.2f}s", (width - 150, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col_tire, 3)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)
            
    cap.release()
    out.release()
    return temp_output.name

# --- Main ---
def main():
    st.title("🏁 Pit Stop Analyzer V76")
    st.markdown("### Simplified Metrics")
    st.info("Tracks Total Pit Stop Time & Total Tire Change Time using Global Optical Flow.")
    
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
            st.write("Processing Video (Optical Flow Extraction)...")
            df, fps, w, h = extract_telemetry(tfile.name, bar.progress)
            
            pit_t, tire_t = analyze_simple_states(df, fps)
            
            if pit_t[0] is None:
                st.error("Could not detect Pit Stop start.")
            else:
                st.write("Rendering Video...")
                vid_path = render_simple_overlay(tfile.name, pit_t, tire_t, fps, w, h, bar.progress)
                
                st.session_state.update({
                    'df': df, 'video_path': vid_path, 
                    'timings': (pit_t, tire_t), 'analysis_done': True
                })
            bar.empty()
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists(tfile.name): os.remove(tfile.name)

    if st.session_state['analysis_done']:
        df = st.session_state['df']
        vid_path = st.session_state['video_path']
        pit_t, tire_t = st.session_state['timings']
        
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Total Pit Stop Time", f"{pit_t[1] - pit_t[0]:.2f}s")
        c2.metric("Total Tire Change Time", f"{tire_t[1] - tire_t[0]:.2f}s")
        
        st.subheader("Video Result")
        c1, c2 = st.columns([3,1])
        with c1:
            if os.path.exists(vid_path): st.video(vid_path)
        with c2:
            if os.path.exists(vid_path):
                with open(vid_path, 'rb') as f:
                    st.download_button("Download MP4", f, file_name="pitstop_debug.mp4")

if __name__ == "__main__":
    main()
