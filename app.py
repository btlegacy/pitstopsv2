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

# --- Helper: Load Reference Image ---
def load_ref_image(folder_name):
    path = os.path.join(BASE_DIR, "refs", folder_name, "*")
    files = glob.glob(path)
    if not files:
        path = os.path.join(BASE_DIR, "refs", f"{folder_name}.*")
        files = glob.glob(path)
    if files:
        return cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    return None

# --- PASS 1: Hybrid Extraction (Flow + YOLO + Template) ---
def extract_hybrid_telemetry(video_path, progress_callback):
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load Refs
    ref_port = load_ref_image("emptyfuelport")
    ref_probe = load_ref_image("probein")
    
    telemetry_data = []
    
    # Optical Flow Setup (ROI: Center)
    roi_x1, roi_x2 = int(width * 0.20), int(width * 0.80)
    roi_y1, roi_y2 = int(height * 0.20), int(height * 0.80)
    
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_roi = prev_gray[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # State variables
    fuel_roi = None 
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # --- 1. OPTICAL FLOW (For Stop/Go) ---
        flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        fx = flow[..., 0]
        fy = flow[..., 1]
        
        mag = np.sqrt(fx**2 + fy**2)
        active_mask = mag > 1.0 
        if np.any(active_mask):
            flow_x = np.median(fx[active_mask])
        else:
            flow_x = 0.0
            
        # --- 2. YOLO (For Tire Change / Car Y) ---
        results = model.track(frame, persist=True, classes=[2], verbose=False, conf=0.15)
        car_y = np.nan
        car_box = None
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            valid_indices = [i for i, b in enumerate(boxes) if b[1] < height * 0.9]
            
            if valid_indices:
                valid_boxes = boxes[valid_indices]
                areas = valid_boxes[:, 2] * valid_boxes[:, 3]
                largest_idx = np.argmax(areas)
                cx, cy, cw, ch = valid_boxes[largest_idx]
                car_y = cy
                
                x1 = int(cx - cw/2)
                y1 = int(cy - ch/2)
                x2 = int(cx + cw/2)
                y2 = int(cy + ch/2)
                car_box = (max(0,x1), max(0,y1), min(width,x2), min(height,y2))

        # --- 3. FUEL (Template Match) ---
        # Locate Port Once
        if car_box and ref_port is not None and fuel_roi is None:
            x1, y1, x2, y2 = car_box
            car_roi_img = gray[y1:y2, x1:x2]
            if car_roi_img.shape[0] > ref_port.shape[0] and car_roi_img.shape[1] > ref_port.shape[1]:
                res = cv2.matchTemplate(car_roi_img, ref_port, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > 0.6:
                    fx = x1 + max_loc[0]
                    fy = y1 + max_loc[1]
                    fw, fh = ref_port.shape[1], ref_port.shape[0]
                    fuel_roi = (max(0, fx-20), max(0, fy-20), fw+40, fh+40)

        # Detect Probe
        probe_score = 0.0
        if fuel_roi and ref_probe is not None:
            fx, fy, fw, fh = fuel_roi
            # Boundary checks
            fx, fy = max(0, fx), max(0, fy)
            fw = min(width - fx, fw)
            fh = min(height - fy, fh)
            
            if fw > 0 and fh > 0:
                fuel_zone = gray[fy:fy+fh, fx:fx+fw]
                if fuel_zone.shape[0] >= ref_probe.shape[0] and fuel_zone.shape[1] >= ref_probe.shape[1]:
                    res = cv2.matchTemplate(fuel_zone, ref_probe, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    probe_score = max_val

        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Flow_X": flow_x,
            "Car_Y": car_y,
            "Probe_Match": probe_score
        })
        
        prev_roi = curr_roi
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)

    cap.release()
    return pd.DataFrame(telemetry_data), fps, width, height, fuel_roi

# --- PASS 2: Hybrid Analysis ---
def analyze_hybrid_states(df, fps):
    # Smoothing
    window = 15
    if len(df) > window:
        df['Flow_X_Smooth'] = savgol_filter(df['Flow_X'], window, 3)
        df['Car_Y'] = df['Car_Y'].interpolate(method='linear', limit_direction='both')
        df['Car_Y_Smooth'] = savgol_filter(df['Car_Y'], window, 3)
        df['Probe_Smooth'] = savgol_filter(df['Probe_Match'], window, 3)
    else:
        df['Flow_X_Smooth'] = df['Flow_X']
        df['Car_Y_Smooth'] = df['Car_Y']
        df['Probe_Smooth'] = df['Probe_Match']

    # 1. PIT STOP (Using Optical Flow - Proven V27 Logic)
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
    
    # 2. TIRES (Using YOLO Car_Y - Object Isolated)
    t_up, t_down = None, None
    
    if t_start and t_end:
        stop_window = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        if not stop_window.empty:
            # Calculate Velocity of the CAR OBJECT
            y_pos = stop_window['Car_Y_Smooth'].values
            y_vel = np.gradient(y_pos)
            times = stop_window['Time'].values
            
            # LIFT: Car moves UP (Y decreases) -> Negative Velocity
            # Find min velocity (max upward speed)
            min_idx = np.argmin(y_vel)
            if y_vel[min_idx] < -0.2: # Threshold for "Moving Up"
                t_up = times[min_idx]
            else:
                t_up = t_start
            
            # DROP: Car moves DOWN (Y increases) -> Positive Velocity
            # Look for Max Velocity AFTER the lift
            drop_mask = times > t_up + 1.0
            if np.any(drop_mask):
                drop_vel = y_vel[drop_mask]
                drop_times = times[drop_mask]
                
                max_idx = np.argmax(drop_vel)
                if drop_vel[max_idx] > 0.2: # Threshold for "Moving Down"
                    t_down = drop_times[max_idx]
                else:
                    t_down = t_end
            else:
                t_down = t_end

    # 3. FUELING (Using ROI Template Match)
    t_fuel_start, t_fuel_end = None, None
    
    if t_start and t_end:
        fuel_window = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        if not fuel_window.empty:
            matches = fuel_window['Probe_Smooth'].values
            times = fuel_window['Time'].values
            
            # High confidence match
            is_fueling = matches > 0.6
            indices = np.where(is_fueling)[0]
            
            if len(indices) > 5:
                t_fuel_start = times[indices[0]]
                
                # Find where it drops off significantly
                # Look for the last index where match > 0.6
                # Then walk forward a bit? No, strict cut off is better for "runs too long"
                t_fuel_end = times[indices[-1]]

    if t_up is None: t_up = t_start
    if t_down is None: t_down = t_end

    return (t_start, t_end), (t_up, t_down), (t_fuel_start, t_fuel_end)

# --- PASS 3: Render ---
def render_overlay(input_path, pit_times, tire_times, fuel_times, fps, width, height, fuel_roi, progress_callback):
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
        
        # Timer Logic
        if t_start and current_time >= t_start:
            val_pit = (t_end - t_start) if (t_end and current_time >= t_end) else (current_time - t_start)
            col_pit = (0,0,255) if (t_end and current_time >= t_end) else (0,255,0)
        else:
            val_pit, col_pit = 0.0, (200,200,200)

        if t_up and current_time >= t_up:
            val_tire = (t_down - t_up) if (t_down and current_time >= t_down) else (current_time - t_up)
            col_tire = (0,0,255) if (t_down and current_time >= t_down) else (0,255,255)
        else:
            val_tire, col_tire = 0.0, (200,200,200)
            
        if t_f_start and current_time >= t_f_start:
            val_fuel = (t_f_end - t_f_start) if (t_f_end and current_time >= t_f_end) else (current_time - t_f_start)
            col_fuel = (0,0,255) if (t_f_end and current_time >= t_f_end) else (255,165,0)
        else:
            val_fuel, col_fuel = 0.0, (200,200,200)
        
        # Draw
        cv2.rectangle(frame, (width-450, 0), (width, 240), (0,0,0), -1)
        
        cv2.putText(frame, "PIT STOP", (width-430, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{val_pit:.2f}s", (width-180, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col_pit, 3)
        
        cv2.putText(frame, "FUELING", (width-430, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{val_fuel:.2f}s", (width-180, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col_fuel, 3)

        cv2.putText(frame, "TIRES", (width-430, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{val_tire:.2f}s", (width-180, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col_tire, 3)

        cv2.putText(frame, "V29: Hybrid Flow+YOLO", (width-430, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
        
        # Visualize Fuel ROI
        if fuel_roi and t_start and current_time > t_start:
            fx, fy, fw, fh = fuel_roi
            color = (0,255,0) if (t_f_start and t_f_start <= current_time <= t_f_end) else (0,165,255)
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), color, 2)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)
            
    cap.release()
    out.release()
    return temp_output.name

# --- Main ---
def main():
    st.title("🏁 Pit Stop Analyzer V29")
    st.markdown("### Hybrid Architecture")
    st.info("Stop/Go: Optical Flow (Robust). Tires: YOLO Body Track (Precise). Fuel: ROI Template (Targeted).")

    missing = []
    for r in ["emptyfuelport", "probein"]:
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
            st.write("Step 1: Hybrid Extraction (Flow + YOLO + Matches)...")
            df, fps, w, h, froi = extract_hybrid_telemetry(tfile.name, bar.progress)
            
            st.write("Step 2: Multi-Signal Analysis...")
            pit_t, tire_t, fuel_t = analyze_hybrid_states(df, fps)
            
            if pit_t[0] is None:
                st.error("Could not detect Stop.")
                st.altair_chart(alt.Chart(df).mark_line().encode(x='Time', y='Flow_X'), use_container_width=True)
            else:
                st.write("Step 3: Rendering Video...")
                vid_path = render_overlay(tfile.name, pit_t, tire_t, fuel_t, fps, w, h, froi, bar.progress)
                
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
        
        st.subheader("Video Result")
        c1, c2 = st.columns([3,1])
        with c1:
            if os.path.exists(vid_path): st.video(vid_path)
        with c2:
            if os.path.exists(vid_path):
                with open(vid_path, 'rb') as f:
                    st.download_button("Download MP4", f, file_name="pitstop_hybrid_v29.mp4")

if __name__ == "__main__":
    main()
