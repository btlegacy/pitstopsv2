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

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

def load_ref_image(folder_name):
    path = os.path.join(BASE_DIR, "refs", folder_name, "*")
    files = glob.glob(path)
    if not files:
        path = os.path.join(BASE_DIR, "refs", f"{folder_name}.*")
        files = glob.glob(path)
    if files:
        return cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    return None

# --- PASS 1: Extraction (Spatial Fuel Search) ---
def extract_telemetry(video_path, progress_callback):
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ref_probe = load_ref_image("probein")

    telemetry_data = []
    
    g_x1, g_x2 = int(width * 0.15), int(width * 0.85)
    g_y1, g_y2 = int(height * 0.15), int(height * 0.85)
    mid_x = int((g_x2 - g_x1) / 2)
    
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_roi = prev_gray[g_y1:g_y2, g_x1:g_x2]
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_roi = gray[g_y1:g_y2, g_x1:g_x2]
        
        # 1. Optical Flow
        flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx = flow[..., 0]
        fy = flow[..., 1]
        
        mag = np.sqrt(fx**2 + fy**2)
        active = mag > 1.0 
        flow_x = np.median(fx[active]) if np.any(active) else 0.0
        
        # Zoom
        f_left = fx[:, :mid_x]
        f_right = fx[:, mid_x:]
        val_l = np.median(f_left[np.abs(f_left)>0.5]) if np.any(np.abs(f_left)>0.5) else 0.0
        val_r = np.median(f_right[np.abs(f_right)>0.5]) if np.any(np.abs(f_right)>0.5) else 0.0
        zoom_score = val_r - val_l
        
        # 2. 4-Corner Activity (and Vertical Flow for Hand Raise)
        h_roi, w_roi = curr_roi.shape
        mid_h, mid_w = h_roi // 2, w_roi // 2
        
        q_mag = mag 
        act_tl = np.mean(q_mag[:mid_h, :mid_w])
        act_tr = np.mean(q_mag[:mid_h, mid_w:])
        act_bl = np.mean(q_mag[mid_h:, :mid_w])
        act_br = np.mean(q_mag[mid_h:, mid_w:])
        
        q_fy = fy
        fy_tl = np.mean(q_fy[:mid_h, :mid_w])
        fy_tr = np.mean(q_fy[:mid_h, mid_w:])
        fy_bl = np.mean(q_fy[mid_h:, :mid_w])
        fy_br = np.mean(q_fy[mid_h:, mid_w:])
        
        # 3. FUEL (Spatial Filter: Inside/Bottom of Car)
        probe_score = 0.0
        
        # Run YOLO to find car bounds
        results = model.track(frame, persist=True, classes=[2], verbose=False, conf=0.15)
        
        if results[0].boxes.id is not None and ref_probe is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            # Filter bottom 10% (wall)
            valid_indices = [i for i, b in enumerate(boxes) if b[1] < height * 0.9]
            
            if valid_indices:
                valid_boxes = boxes[valid_indices]
                areas = valid_boxes[:, 2] * valid_boxes[:, 3]
                largest_idx = np.argmax(areas)
                cx, cy, cw, ch = valid_boxes[largest_idx]
                
                # Define Fuel Search Zone: Bottom Half of Car (Inside)
                # Car Box: (x1, y1) top-left, (x2, y2) bottom-right
                x1 = int(cx - cw/2)
                x2 = int(cx + cw/2)
                # "Inside" is the bottom half (closest to wall)
                y1 = int(cy) # Start from middle
                y2 = int(cy + ch/2) # End at bottom edge
                
                # Boundary checks
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                if (x2 > x1) and (y2 > y1):
                    fuel_zone = gray[y1:y2, x1:x2]
                    
                    if fuel_zone.shape[0] >= ref_probe.shape[0] and fuel_zone.shape[1] >= ref_probe.shape[1]:
                        res = cv2.matchTemplate(fuel_zone, ref_probe, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(res)
                        probe_score = max_val

        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Flow_X": flow_x,
            "Zoom_Score": zoom_score,
            "Probe_Match": probe_score,
            "Act_TL": act_tl, "Fy_TL": fy_tl,
            "Act_TR": act_tr, "Fy_TR": fy_tr,
            "Act_BL": act_bl, "Fy_BL": fy_bl,
            "Act_BR": act_br, "Fy_BR": fy_br
        })
        
        prev_roi = curr_roi
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)

    cap.release()
    return pd.DataFrame(telemetry_data), fps, width, height

# --- Helper: Active Window ---
def find_active_window(times, signal, start_gate, end_gate, fps):
    mask = (times >= start_gate) & (times <= end_gate)
    if not np.any(mask): return start_gate, start_gate
    
    t_win = times[mask]
    s_win = signal[mask]
    
    baseline = np.percentile(s_win, 10)
    peak_val = np.max(s_win)
    dynamic_range = peak_val - baseline
    
    if dynamic_range < 0.5: return start_gate, start_gate

    thresh_start = baseline + (dynamic_range * 0.30)
    thresh_end = baseline + (dynamic_range * 0.25)
    
    active_high = np.where(s_win > thresh_start)[0]
    if len(active_high) == 0: return start_gate, start_gate
    idx_start = active_high[0]
    
    peak_idx = np.argmax(s_win)
    search_start = max(idx_start, peak_idx)
    idx_end = len(s_win) - 1
    
    buffer_frames = int(fps * 0.5)
    
    for i in range(search_start, len(s_win) - buffer_frames):
        if s_win[i] < thresh_end:
            future_segment = s_win[i : i + buffer_frames]
            if np.mean(future_segment) < thresh_end:
                idx_end = i
                break
    
    return t_win[idx_start], t_win[idx_end]

# --- PASS 2: Analysis V40 ---
def analyze_states_v40(df, fps):
    window = 15
    cols_to_smooth = ['Flow_X', 'Zoom_Score', 'Probe_Match', 
                      'Act_TL', 'Act_TR', 'Act_BL', 'Act_BR',
                      'Fy_TL', 'Fy_TR', 'Fy_BL', 'Fy_BR']
    
    for col in cols_to_smooth:
        if len(df) > window:
            df[f'{col}_Sm'] = savgol_filter(df[col], window, 3)
        else:
            df[f'{col}_Sm'] = df[col]

    df['Zoom_Vel'] = np.gradient(df['Zoom_Score_Sm'])

    # 1. PIT STOP
    x_mag = df['Flow_X_Sm'].abs()
    MOVE_THRESH = x_mag.max() * 0.3 
    STOP_THRESH = x_mag.max() * 0.05 
    
    peaks, _ = find_peaks(x_mag, height=MOVE_THRESH, distance=fps*5)
    t_start, t_end = None, None
    arrival_dir = 0 
    
    if len(peaks) >= 2:
        arrival_idx = peaks[0]
        depart_idx = peaks[-1]
        arr_flow = df['Flow_X_Sm'].iloc[arrival_idx]
        arrival_dir = 1 if arr_flow > 0 else -1 
        
        for i in range(arrival_idx, depart_idx):
            if x_mag.iloc[i] < STOP_THRESH:
                t_start = df.iloc[i]['Time']
                break
        for i in range(depart_idx, arrival_idx, -1):
            if x_mag.iloc[i] < STOP_THRESH:
                t_end = df.iloc[i]['Time']
                break

    # 2. JACKS
    t_up, t_down = None, None
    if t_start and t_end:
        t_creep = t_end - 1.0
        stop_window = df[(df['Time'] >= t_start) & (df['Time'] <= t_creep)]
        
        if not stop_window.empty:
            z_pos = stop_window['Zoom_Score_Sm'].values
            z_vel = stop_window['Zoom_Vel'].values
            times = stop_window['Time'].values
            
            peaks_up, _ = find_peaks(z_pos, height=0.2, distance=fps)
            if len(peaks_up) > 0:
                t_up = times[peaks_up[0]]
            else:
                t_up = t_start
                
            mask_drop = times > t_up + 2.0 
            if np.any(mask_drop):
                drop_vel = z_vel[mask_drop]
                drop_times = times[mask_drop]
                min_idx = np.argmin(drop_vel)
                if drop_vel[min_idx] < -0.02: 
                    t_down = drop_times[min_idx]
                else:
                    t_down = t_end
            else:
                t_down = t_end

    # 3. CORNER TIMING (Hand Raise Logic)
    map_act = {} 
    map_fy = {} 
    
    if arrival_dir > 0: 
        map_act['Inside Rear'] = 'Act_BL_Sm'; map_fy['Inside Rear'] = 'Fy_BL_Sm'
        map_act['Inside Front'] = 'Act_BR_Sm'; map_fy['Inside Front'] = 'Fy_BR_Sm'
        map_act['Outside Rear'] = 'Act_TL_Sm'; map_fy['Outside Rear'] = 'Fy_TL_Sm'
        map_act['Outside Front'] = 'Act_TR_Sm'; map_fy['Outside Front'] = 'Fy_TR_Sm'
    else: 
        map_act['Inside Rear'] = 'Act_BR_Sm'; map_fy['Inside Rear'] = 'Fy_BR_Sm'
        map_act['Inside Front'] = 'Act_BL_Sm'; map_fy['Inside Front'] = 'Fy_BL_Sm'
        map_act['Outside Rear'] = 'Act_TR_Sm'; map_fy['Outside Rear'] = 'Fy_TR_Sm'
        map_act['Outside Front'] = 'Act_TL_Sm'; map_fy['Outside Front'] = 'Fy_TL_Sm'

    corner_stats = {}
    
    if t_up and t_down:
        t_analysis_end = t_end 
        df_jacks = df[(df['Time'] >= t_up) & (df['Time'] <= t_analysis_end)]
        times_j = df_jacks['Time'].values
        
        # A. FRONT (OF -> IF)
        sig_of = df_jacks[map_act['Outside Front']].values
        sig_if = df_jacks[map_act['Inside Front']].values
        
        t_of_start, t_of_end = find_active_window(times_j, sig_of, t_up, t_analysis_end, fps)
        
        gate_if = t_of_start + 1.5
        t_if_start, t_if_end = find_active_window(times_j, sig_if, gate_if, t_analysis_end, fps)
        
        corner_stats['Outside Front'] = (t_of_start, t_of_end)
        corner_stats['Inside Front'] = (t_if_start, t_if_end)
        
        # B. REAR (IR -> OR) with Hand Raise
        sig_ir = df_jacks[map_act['Inside Rear']].values
        sig_or = df_jacks[map_act['Outside Rear']].values
        
        t_ir_start, t_ir_raw_end = find_active_window(times_j, sig_ir, t_up, t_analysis_end, fps)
        
        gate_or = t_ir_start + 2.0
        t_or_start, t_or_raw_end = find_active_window(times_j, sig_or, gate_or, t_analysis_end, fps)
        
        # Strict Handover IR->OR
        if t_or_start > t_ir_start: t_ir_end = t_or_start
        else: t_ir_end = t_ir_raw_end
            
        # OR Hand Raise Check
        t_search_start = t_or_raw_end - 1.5
        t_search_end = t_or_raw_end + 2.0
        or_fy_col = map_fy['Outside Rear']
        mask_hr = (df['Time'] >= t_search_start) & (df['Time'] <= t_search_end)
        
        if np.any(mask_hr):
            fy_segment = df.loc[mask_hr, or_fy_col].values
            t_segment = df.loc[mask_hr, 'Time'].values
            peaks, _ = find_peaks(-fy_segment, height=0.5, distance=fps) 
            if len(peaks) > 0:
                t_or_end = t_segment[peaks[0]]
            else:
                t_or_end = t_or_raw_end
        else:
            t_or_end = t_or_raw_end
        
        corner_stats['Inside Rear'] = (t_ir_start, t_ir_end)
        corner_stats['Outside Rear'] = (t_or_start, t_or_end)

    # 4. FUEL (Precision Logic)
    t_fuel_start, t_fuel_end = None, None
    
    if t_start and t_end:
        # Limit search to Stop Window
        fuel_w = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        
        if not fuel_w.empty:
            matches = fuel_w['Probe_Match_Sm'].values
            times = fuel_w['Time'].values
            
            # Use 65% Confidence Threshold
            # This filters out "Probe Hovering" vs "Probe Inserted"
            is_connected = matches > 0.65
            
            indices = np.where(is_connected)[0]
            
            # Must be connected for at least 1 second (30 frames) to count
            if len(indices) > fps:
                t_fuel_start = times[indices[0]]
                
                # Find end: Look for first drop below threshold
                # Handle potential signal flickers (debouncing)
                
                # Find contiguous blocks
                diffs = np.diff(indices)
                # If gaps > 10 frames, treat as disconnect
                split_points = np.where(diffs > 10)[0]
                
                if len(split_points) > 0:
                    # Take the first major block as the fueling event
                    end_idx = indices[split_points[0]]
                else:
                    end_idx = indices[-1]
                    
                t_fuel_end = times[end_idx]
                
                # Clamp to pit end
                if t_fuel_end > t_end - 0.5: t_fuel_end = t_end - 0.5

    if t_up is None: t_up = t_start
    if t_down is None: t_down = t_end

    return (t_start, t_end), (t_up, t_down), (t_fuel_start, t_fuel_end), corner_stats

# --- PASS 3: Render ---
def render_overlay(input_path, pit, tires, fuel, corner_data, fps, width, height, progress_callback):
    cap = cv2.VideoCapture(input_path)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
    
    t_start, t_end = pit
    t_up, t_down = tires
    t_f_start, t_f_end = fuel
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    
    labels = ["Inside Rear", "Outside Rear", "Outside Front", "Inside Front"]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        curr = frame_idx / fps
        
        # --- TIMERS ---
        if t_start and curr >= t_start:
            vp = (t_end - t_start) if (t_end and curr >= t_end) else (curr - t_start)
            cp = (0,0,255) if (t_end and curr >= t_end) else (0,255,0)
        else: vp, cp = 0.0, (200,200,200)

        if t_up and curr >= t_up:
            vt = (t_down - t_up) if (t_down and curr >= t_down) else (curr - t_up)
            ct = (0,0,255) if (t_down and curr >= t_down) else (0,255,255)
        else: vt, ct = 0.0, (200,200,200)

        if t_f_start and curr >= t_f_start:
            vf = (t_f_end - t_f_start) if (t_f_end and curr >= t_f_end) else (curr - t_f_start)
            cf = (0,0,255) if (t_f_end and curr >= t_f_end) else (255,165,0)
        else: vf, cf = 0.0, (200,200,200)
        
        # --- UI ---
        cv2.rectangle(frame, (width-450, 0), (width, 320), (0,0,0), -1)
        cv2.putText(frame, "PIT STOP", (width-430, 40), 0, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{vp:.2f}s", (width-180, 40), 0, 1.2, cp, 3)
        
        cv2.putText(frame, "FUELING", (width-430, 90), 0, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{vf:.2f}s", (width-180, 90), 0, 1.2, cf, 3)
        
        cv2.putText(frame, "TIRES (Total)", (width-430, 140), 0, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{vt:.2f}s", (width-180, 140), 0, 1.2, ct, 3)

        # Corners
        start_y = 180
        gap_y = 30
        for i, label in enumerate(labels):
            c_start, c_end = corner_data.get(label, (0.0, 0.0))
            c_val = 0.0
            if c_start > 0 and curr >= c_start:
                if curr >= c_end: c_val = c_end - c_start
                else: c_val = curr - c_start
            txt_col = (255,255,255) if c_val > 0 else (150,150,150)
            y_pos = start_y + (i*gap_y)
            cv2.putText(frame, label, (width-430, y_pos), 0, 0.6, txt_col, 1)
            cv2.putText(frame, f"{c_val:.1f}s", (width-100, y_pos), 0, 0.6, txt_col, 2)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)
            
    cap.release()
    out.release()
    return temp_output.name

# --- Main ---
def main():
    st.title("🏁 Pit Stop Analyzer V40")
    st.markdown("### Precision Fueling")
    st.info("Restricts fuel search to **Inside/Bottom Car Zone**. High threshold (65%) prevents early triggering.")

    missing = []
    for r in ["probein", "emptyfuelport"]:
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
            st.write("Step 1: Extraction (Spatial Filters)...")
            df, fps, w, h = extract_telemetry(tfile.name, bar.progress)
            
            st.write("Step 2: Analysis...")
            pit_t, tire_t, fuel_t, corners = analyze_states_v40(df, fps)
            
            if pit_t[0] is None:
                st.error("Could not detect Stop.")
            else:
                st.write("Step 3: Rendering Video...")
                vid_path = render_overlay(tfile.name, pit_t, tire_t, fuel_t, corners, fps, w, h, bar.progress)
                
                st.session_state.update({
                    'df': df, 'video_path': vid_path, 
                    'timings': (pit_t, tire_t, fuel_t, corners), 'analysis_done': True
                })
            bar.empty()
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists(tfile.name): os.remove(tfile.name)

    if st.session_state['analysis_done']:
        df = st.session_state['df']
        vid_path = st.session_state['video_path']
        pit_t, tire_t, fuel_t, corners = st.session_state['timings']
        
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Pit Stop Time", f"{pit_t[1] - pit_t[0]:.2f}s")
        
        f_dur = (fuel_t[1] - fuel_t[0]) if (fuel_t[0] and fuel_t[1]) else 0
        c2.metric("Fueling Time", f"{f_dur:.2f}s" if f_dur > 0 else "N/A")
        
        t_dur = (tire_t[1] - tire_t[0]) if (tire_t[0] and tire_t[1]) else 0
        c3.metric("Tire Change Time", f"{t_dur:.2f}s")
        
        st.subheader("Video Result")
        c1, c2 = st.columns([3,1])
        with c1:
            if os.path.exists(vid_path): st.video(vid_path)
        with c2:
            if os.path.exists(vid_path):
                with open(vid_path, 'rb') as f:
                    st.download_button("Download MP4", f, file_name="pitstop_v40.mp4")

if __name__ == "__main__":
    main()
