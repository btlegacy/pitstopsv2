import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import glob
import pandas as pd
import altair as alt
from scipy.signal import savgol_filter, find_peaks
from ultralytics import YOLO

# --- Configuration ---
st.set_page_config(page_title="Pit Stop Analytics AI", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load AI Model ---
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

# --- Helper: Reference Loading ---
def load_ref_image(folder_name):
    """Loads the first image in the folder to use as a template."""
    path = os.path.join(BASE_DIR, "refs", folder_name, "*")
    files = glob.glob(path)
    if not files:
        path = os.path.join(BASE_DIR, "refs", f"{folder_name}.*")
        files = glob.glob(path)
    if files:
        return cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    return None

# --- PASS 1: Extraction (Flow for Tires, YOLO for Fuel Area) ---
def extract_telemetry(video_path, progress_callback):
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load Fuel Probe Reference
    ref_probe = load_ref_image("probein")
    
    telemetry_data = []
    
    # Optical Flow Setup (Center ROI)
    roi_x1, roi_x2 = int(width * 0.20), int(width * 0.80)
    roi_y1, roi_y2 = int(height * 0.20), int(height * 0.80)
    mid_x = int((roi_x2 - roi_x1) / 2)
    
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_roi = prev_gray[roi_y1:roi_y2, roi_x1:roi_x2]
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # --- 1. OPTICAL FLOW (For Pit Stop & Tires) ---
        # We use this because it is extremely sensitive to "Lift" (Zoom)
        flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        fx = flow[..., 0]
        
        # Horizontal Momentum (Stop detection)
        mag = np.sqrt(fx**2 + flow[..., 1]**2)
        active_mask = mag > 1.0 
        if np.any(active_mask):
            flow_x = np.median(fx[active_mask])
        else:
            flow_x = 0.0
            
        # Zoom Metric (Tire detection)
        # Right moves Right - Left moves Left = Expansion
        flow_left = fx[:, :mid_x]
        flow_right = fx[:, mid_x:]
        mask_l = np.abs(flow_left) > 0.5
        mask_r = np.abs(flow_right) > 0.5
        val_l = np.median(flow_left[mask_l]) if np.any(mask_l) else 0.0
        val_r = np.median(flow_right[mask_r]) if np.any(mask_r) else 0.0
        zoom_score = val_r - val_l
        
        # --- 2. FUEL DETECTION (YOLO + Template) ---
        # Strategy: Use YOLO to find the car, then search ONLY the car for the probe.
        # This avoids false positives on the pit wall.
        results = model.track(frame, persist=True, classes=[2], verbose=False, conf=0.15)
        probe_score = 0.0
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            # Filter out pit wall (bottom 10%)
            valid_indices = [i for i, b in enumerate(boxes) if b[1] < height * 0.9]
            
            if valid_indices:
                valid_boxes = boxes[valid_indices]
                areas = valid_boxes[:, 2] * valid_boxes[:, 3]
                largest_idx = np.argmax(areas)
                cx, cy, cw, ch = valid_boxes[largest_idx]
                
                # Define Car ROI
                x1 = int(cx - cw/2)
                y1 = int(cy - ch/2)
                x2 = int(cx + cw/2)
                y2 = int(cy + ch/2)
                
                # Ensure within bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                if ref_probe is not None and (x2>x1) and (y2>y1):
                    car_roi = gray[y1:y2, x1:x2]
                    if car_roi.shape[0] >= ref_probe.shape[0] and car_roi.shape[1] >= ref_probe.shape[1]:
                        res = cv2.matchTemplate(car_roi, ref_probe, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(res)
                        probe_score = max_val

        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Flow_X": flow_x,
            "Zoom_Score": zoom_score,
            "Probe_Match": probe_score
        })
        
        prev_roi = curr_roi
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)

    cap.release()
    return pd.DataFrame(telemetry_data), fps, width, height

# --- PASS 2: Analysis ---
def analyze_states_v30(df, fps):
    # Smoothing
    window_slow = 15
    window_fast = 5
    
    if len(df) > window_slow:
        df['Flow_X_Smooth'] = savgol_filter(df['Flow_X'], window_slow, 3)
        df['Zoom_Smooth'] = savgol_filter(df['Zoom_Score'], window_fast, 3)
        df['Probe_Smooth'] = savgol_filter(df['Probe_Match'], window_slow, 3)
    else:
        df['Flow_X_Smooth'] = df['Flow_X']
        df['Zoom_Smooth'] = df['Zoom_Score']
        df['Probe_Smooth'] = df['Probe_Match']

    df['Zoom_Velocity'] = np.gradient(df['Zoom_Smooth'])

    # 1. PIT STOP (Optical Flow X - Robust)
    x_mag = df['Flow_X_Smooth'].abs()
    MOVE_THRESH = x_mag.max() * 0.3 
    STOP_THRESH = x_mag.max() * 0.05 
    
    peaks, _ = find_peaks(x_mag, height=MOVE_THRESH, distance=fps*5)
    t_start, t_end = None, None
    
    if len(peaks) >= 2:
        arrival_idx = peaks[0]
        depart_idx = peaks[-1]
        for i in range(arrival_idx, depart_idx):
            if x_mag.iloc[i] < STOP_THRESH:
                t_start = df.iloc[i]['Time']
                break
        for i in range(depart_idx, arrival_idx, -1):
            if x_mag.iloc[i] < STOP_THRESH:
                t_end = df.iloc[i]['Time']
                break
    
    # 2. TIRES (Optical Flow Zoom - Sensitive)
    # Uses "Snap" detection from V23 which worked best for you
    t_up, t_down = None, None
    
    if t_start and t_end:
        # Search inside stop window (minus 1s buffer for departure)
        t_creep = t_end - 1.0
        stop_window = df[(df['Time'] >= t_start) & (df['Time'] <= t_creep)]
        
        if not stop_window.empty:
            z_pos = stop_window['Zoom_Smooth'].values
            z_vel = stop_window['Zoom_Velocity'].values
            times = stop_window['Time'].values
            
            # LIFT: First Positive Zoom Peak
            peaks_up, _ = find_peaks(z_pos, height=0.2, distance=fps)
            if len(peaks_up) > 0:
                t_up = times[peaks_up[0]]
            else:
                t_up = t_start
                
            # DROP: Global Minimum Velocity (The Snap)
            # Must happen after Lift
            mask_drop = times > t_up + 1.0
            if np.any(mask_drop):
                drop_vel = z_vel[mask_drop]
                drop_times = times[mask_drop]
                
                # Find the sharpest contraction
                min_idx = np.argmin(drop_vel)
                if drop_vel[min_idx] < -0.05: # Noise threshold
                    t_down = drop_times[min_idx]
                else:
                    t_down = t_end
            else:
                t_down = t_end

    # 3. FUELING (Visual Match)
    t_fuel_start, t_fuel_end = None, None
    
    if t_start and t_end:
        fuel_window = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        if not fuel_window.empty:
            matches = fuel_window['Probe_Smooth'].values
            times = fuel_window['Time'].values
            
            # Match Threshold
            # 0.6 is usually a good match for TM_CCOEFF_NORMED
            is_fueling = matches > 0.55
            indices = np.where(is_fueling)[0]
            
            # Must match for at least 0.5 seconds to be valid
            if len(indices) > int(fps * 0.5):
                t_fuel_start = times[indices[0]]
                t_fuel_end = times[indices[-1]]
                
                # Sanity check: Fueling cannot continue after car leaves
                if t_fuel_end > t_end - 0.5:
                    t_fuel_end = t_end - 0.5

    # Fallbacks
    if t_up is None: t_up = t_start
    if t_down is None: t_down = t_end

    return (t_start, t_end), (t_up, t_down), (t_fuel_start, t_fuel_end)

# --- PASS 3: Render ---
def render_overlay(input_path, pit_times, tire_times, fuel_times, fps, width, height, progress_callback):
    t_start, t_end = pit_times
    t_up, t_down = tire_times
    t_f_start, t_f_end = fuel_times
    
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
        if t_start and current_time >= t_start:
            if t_end and current_time >= t_end:
                val_pit, col_pit = t_end - t_start, (0,0,255)
            else:
                val_pit, col_pit = current_time - t_start, (0,255,0)
        else:
            val_pit, col_pit = 0.0, (200,200,200)

        if t_up and current_time >= t_up:
            if t_down and current_time >= t_down:
                val_tire, col_tire = t_down - t_up, (0,0,255)
            else:
                val_tire, col_tire = current_time - t_up, (0,255,255)
        else:
            val_tire, col_tire = 0.0, (200,200,200)
            
        if t_f_start and current_time >= t_f_start:
            if t_f_end and current_time >= t_f_end:
                val_fuel, col_fuel = t_f_end - t_f_start, (0,0,255)
            else:
                val_fuel, col_fuel = current_time - t_f_start, (255,165,0)
        else:
            val_fuel, col_fuel = 0.0, (200,200,200)
        
        # UI
        cv2.rectangle(frame, (width-450, 0), (width, 240), (0,0,0), -1)
        cv2.putText(frame, "PIT STOP", (width-430, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{val_pit:.2f}s", (width-180, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col_pit, 3)
        
        cv2.putText(frame, "FUELING", (width-430, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{val_fuel:.2f}s", (width-180, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col_fuel, 3)

        cv2.putText(frame, "TIRES", (width-430, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{val_tire:.2f}s", (width-180, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col_tire, 3)
        
        cv2.putText(frame, "V30: Optical Zoom + Fuel ROI", (width-430, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)
            
    cap.release()
    out.release()
    return temp_output.name

# --- Main ---
def main():
    st.title("🏁 Pit Stop Analyzer V30")
    st.markdown("### Best of Both Worlds")
    st.info("Restored **Optical Zoom** for precise Tire timing. Enabled **Global Fuel Search** within the Car ROI.")

    missing = []
    for r in ["probein"]:
        p = os.path.join(BASE_DIR, "refs", r)
        if not os.path.exists(p) and not glob.glob(p+".*"): missing.append(r)
    if missing: st.error(f"Missing refs: {missing}")

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
            st.write("Step 1: Extraction (Flow, Zoom, Fuel Scan)...")
            df, fps, w, h = extract_telemetry(tfile.name, bar.progress)
            
            st.write("Step 2: Logic Analysis...")
            pit_t, tire_t, fuel_t = analyze_states_v30(df, fps)
            
            if pit_t[0] is None:
                st.error("Could not detect Stop.")
            else:
                st.write("Step 3: Rendering Video...")
                vid_path = render_overlay(tfile.name, pit_t, tire_t, fuel_t, fps, w, h, bar.progress)
                
                st.session_state.update({
                    'df': df, 'video_path': vid_path, 'timings': (pit_t, tire_t, fuel_t), 'analysis_done': True
                })
            bar.empty()
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists(tfile.name): os.remove(tfile.name)

    if st.session_state['analysis_done']:
        df = st.session_state['df']
        vid_path = st.session_state['video_path']
        pit_t, tire_t, fuel_t = st.session_state['timings']
        
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Pit Stop Time", f"{pit_t[1] - pit_t[0]:.2f}s")
        c2.metric("Fueling Time", f"{fuel_t[1] - fuel_t[0]:.2f}s" if fuel_t[0] else "N/A")
        c3.metric("Tire Change Time", f"{tire_t[1] - tire_t[0]:.2f}s")
        
        st.subheader("📊 Telemetry")
        base = alt.Chart(df).encode(x='Time')
        
        fuel_chart = base.mark_area(color='orange', opacity=0.3).encode(y=alt.Y('Probe_Smooth', title='Fuel Match'))
        zoom_chart = base.mark_line(color='magenta').encode(y=alt.Y('Zoom_Smooth', title='Zoom (Tires)'))
        
        st.altair_chart((fuel_chart + zoom_chart).interactive(), use_container_width=True)
        
        st.subheader("Video Result")
        c1, c2 = st.columns([3,1])
        with c1:
            if os.path.exists(vid_path): st.video(vid_path)
        with c2:
            if os.path.exists(vid_path):
                with open(vid_path, 'rb') as f:
                    st.download_button("Download MP4", f, file_name="pitstop_v30.mp4")

if __name__ == "__main__":
    main()
