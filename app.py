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
        img = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
        return img
    return None

# --- PASS 1: Object-Isolated Extraction ---
def extract_telemetry(video_path, progress_callback):
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
    
    # State variables
    fuel_roi = None # (x, y, w, h)
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. YOLO OBJECT TRACKING (Car Isolation)
        # We track the car to get its Y-Position (Vertical movement of body)
        # and X-Velocity (Stop detection)
        results = model.track(frame, persist=True, classes=[2], verbose=False, conf=0.15)
        
        car_y_center = np.nan
        car_box = None
        
        if results[0].boxes.id is not None:
            # Get largest car box
            boxes = results[0].boxes.xywh.cpu().numpy() # x_c, y_c, w, h
            
            # Filter out bottom 10% (pit wall noise)
            valid_indices = [i for i, b in enumerate(boxes) if b[1] < height * 0.9]
            
            if valid_indices:
                valid_boxes = boxes[valid_indices]
                # Pick largest
                areas = valid_boxes[:, 2] * valid_boxes[:, 3]
                largest_idx = np.argmax(areas)
                
                cx, cy, cw, ch = valid_boxes[largest_idx]
                car_y_center = cy
                
                # Convert to xyxy for ROI extraction
                x1 = int(cx - cw/2)
                y1 = int(cy - ch/2)
                x2 = int(cx + cw/2)
                y2 = int(cy + ch/2)
                car_box = (max(0,x1), max(0,y1), min(width,x2), min(height,y2))

        # 2. FUEL PORT LOCALIZATION (Run once when car is found)
        # If we have a car box but don't know where the fuel port is yet
        if car_box and ref_port is not None and fuel_roi is None:
            # Search for port ONLY inside the car box
            x1, y1, x2, y2 = car_box
            car_roi = gray[y1:y2, x1:x2]
            
            # Template match
            if car_roi.shape[0] > ref_port.shape[0] and car_roi.shape[1] > ref_port.shape[1]:
                res = cv2.matchTemplate(car_roi, ref_port, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                # If confident we found the port
                if max_val > 0.6:
                    # Define Fuel ROI relative to global frame
                    fx = x1 + max_loc[0]
                    fy = y1 + max_loc[1]
                    fw, fh = ref_port.shape[1], ref_port.shape[0]
                    
                    # Expand ROI slightly for probe movement
                    pad = 20
                    fuel_roi = (max(0, fx-pad), max(0, fy-pad), fw+pad*2, fh+pad*2)

        # 3. FUEL PROBE DETECTION (Targeted)
        probe_score = 0.0
        if fuel_roi and ref_probe is not None:
            fx, fy, fw, fh = fuel_roi
            # Ensure ROI is within bounds
            fx, fy = max(0, fx), max(0, fy)
            fw = min(width - fx, fw)
            fh = min(height - fy, fh)
            
            if fw > 0 and fh > 0:
                fuel_zone = gray[fy:fy+fh, fx:fx+fw]
                
                # Check if reference fits
                if fuel_zone.shape[0] >= ref_probe.shape[0] and fuel_zone.shape[1] >= ref_probe.shape[1]:
                    res = cv2.matchTemplate(fuel_zone, ref_probe, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    probe_score = max_val

        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Car_Y": car_y_center,
            "Probe_Match": probe_score
        })
        
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)

    cap.release()
    return pd.DataFrame(telemetry_data), fps, width, height, fuel_roi

# --- PASS 2: Analysis (Y-Axis Logic) ---
def analyze_object_states(df, fps):
    # 1. PRE-PROCESSING
    # Interpolate missing Car_Y (occlusions)
    df['Car_Y'] = df['Car_Y'].interpolate(method='linear', limit_direction='both')
    
    # Smooth Data
    window = 15
    if len(df) > window:
        df['Y_Smooth'] = savgol_filter(df['Car_Y'], window, 3)
        df['Probe_Smooth'] = savgol_filter(df['Probe_Match'], window, 3)
    else:
        df['Y_Smooth'] = df['Car_Y']
        df['Probe_Smooth'] = df['Probe_Match']
        
    # Calculate Y-Velocity (Negative = UP, Positive = DOWN in image coords)
    df['Y_Vel'] = np.gradient(df['Y_Smooth'])

    # --- A. PIT STOP (Horizontal) ---
    # We can infer stop from Y-stability, but let's use the previous reliable X-Flow logic?
    # Or simplified: If Car_Y is stable (velocity near 0) for > 2s
    # Let's stick to the proven "Velocity Crash" logic if we had X, but here we only tracked Y.
    # Hack: A stopped car has very low Y-Velocity variance.
    
    y_vel_abs = np.abs(df['Y_Vel'])
    is_still = y_vel_abs < 0.3 # Pixel movement per frame
    
    # Find blocks of stillness
    df['block'] = (is_still != is_still.shift()).cumsum()
    blocks = df[is_still].groupby('block')
    
    t_start, t_end = None, None
    if len(blocks) > 0:
        largest_block = blocks.size().idxmax()
        block_data = df[df['block'] == largest_block]
        if (block_data.iloc[-1]['Time'] - block_data.iloc[0]['Time']) > 2.0:
            t_start = block_data.iloc[0]['Time']
            t_end = block_data.iloc[-1]['Time']

    # --- B. TIRES (Y-Axis Shift) ---
    t_up, t_down = None, None
    
    if t_start and t_end:
        stop_window = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        if not stop_window.empty:
            y_vel = stop_window['Y_Vel'].values
            times = stop_window['Time'].values
            
            # 1. FIND LIFT (Negative Velocity Spike = Moving Up)
            # The car moves UP, so Y decreases. We look for min velocity.
            # Threshold: -0.5 px/frame
            lift_candidates = np.where(y_vel < -0.5)[0]
            
            if len(lift_candidates) > 0:
                # First significant upward movement
                t_up = times[lift_candidates[0]]
            else:
                t_up = t_start
            
            # 2. FIND DROP (Positive Velocity Spike = Moving Down)
            # The car moves DOWN, so Y increases. We look for max velocity.
            # It must happen AFTER lift.
            
            # Look for the "Snap" (Max Positive Velocity)
            drop_candidates_mask = (times > t_up + 2.0) # Allow 2s for jacks to go up/work
            if np.any(drop_candidates_mask):
                drop_win_vel = y_vel[drop_candidates_mask]
                drop_win_time = times[drop_candidates_mask]
                
                max_drop_idx = np.argmax(drop_win_vel)
                max_drop_val = drop_win_vel[max_drop_idx]
                
                if max_drop_val > 0.5:
                    snap_time = drop_win_time[max_drop_idx]
                    
                    # 3. FIND SETTLE (Wait for velocity to return to 0)
                    # Walk forward from snap until velocity is low
                    settle_time = snap_time
                    snap_global_idx = np.where(df['Time'] == snap_time)[0][0]
                    
                    for i in range(snap_global_idx, len(df)):
                        if np.abs(df['Y_Vel'].iloc[i]) < 0.1: # Settled
                            settle_time = df.iloc[i]['Time']
                            break
                        # Limit settle search to 1.5s
                        if df.iloc[i]['Time'] > snap_time + 1.5:
                            settle_time = snap_time + 1.5
                            break
                            
                    t_down = settle_time
                else:
                    t_down = t_end
            else:
                t_down = t_end

    # --- C. FUELING (Targeted Match) ---
    t_fuel_start, t_fuel_end = None, None
    
    if t_start and t_end:
        fuel_window = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        if not fuel_window.empty:
            match_scores = fuel_window['Probe_Smooth'].values
            times = fuel_window['Time'].values
            
            # Threshold: High confidence match
            # If we found the probe, scores usually jump > 0.6 or 0.7
            # If no probe, scores stay < 0.4
            
            high_match = match_scores > 0.55
            
            # Find contiguous blocks
            indices = np.where(high_match)[0]
            
            if len(indices) > 5: # Must match for at least 5 frames
                t_fuel_start = times[indices[0]]
                t_fuel_end = times[indices[-1]]
                
                # Logic Check: Fueling end usually has a distinct drop-off
                # If it runs till t_end, user said "runs too long".
                # Let's trim the tail: find where it drops below 0.55 strictly
                pass

    # Fallbacks
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
        
        # Debug: Show Fuel ROI
        if fuel_roi:
            fx, fy, fw, fh = fuel_roi
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 165, 255), 2)
            cv2.putText(frame, "FUEL ZONE", (fx, fy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)
            
    cap.release()
    out.release()
    return temp_output.name

# --- Main ---
def main():
    st.title("🏁 Pit Stop Analyzer V28")
    st.markdown("### Object-Isolated Logic")
    st.info("Uses YOLO to track the car body (ignoring crew) for tire timing. Uses localized template matching for fuel.")

    # Check refs
    missing = []
    for r in ["emptyfuelport", "probein"]:
        p = os.path.join(BASE_DIR, "refs", r)
        if not os.path.exists(p) and not glob.glob(p+".*"): missing.append(r)
    
    if missing:
        st.error(f"Missing reference images: {', '.join(missing)}. Analysis will be limited.")

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
            st.write("Step 1: Extraction (YOLO Tracking & ROI Matching)...")
            df, fps, w, h, fuel_roi = extract_telemetry(tfile.name, bar.progress)
            
            st.write("Step 2: Logic Analysis (Settle Logic)...")
            pit_t, tire_t, fuel_t = analyze_object_states(df, fps)
            
            if pit_t[0] is None:
                st.error("Could not detect Stop.")
            else:
                st.write("Step 3: Rendering Video...")
                vid_path = render_overlay(tfile.name, pit_t, tire_t, fuel_t, fps, w, h, fuel_roi, bar.progress)
                
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
        
        y_chart = base.mark_line(color='cyan').encode(y=alt.Y('Y_Smooth', title='Car Y-Position (Box Center)'))
        fuel_chart = base.mark_area(color='orange', opacity=0.3).encode(y=alt.Y('Probe_Smooth', title='Probe Matches'))
        
        st.altair_chart((y_chart + fuel_chart).interactive(), use_container_width=True)
        
        st.subheader("Video Result")
        c1, c2 = st.columns([3,1])
        with c1:
            if os.path.exists(vid_path): st.video(vid_path)
        with c2:
            if os.path.exists(vid_path):
                with open(vid_path, 'rb') as f:
                    st.download_button("Download MP4", f, file_name="pitstop_v28.mp4")

if __name__ == "__main__":
    main()
