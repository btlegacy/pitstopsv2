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

def load_templates(folder_name):
    templates = []
    path = os.path.join(BASE_DIR, "refs", folder_name, "*")
    files = glob.glob(path)
    if not files:
        path = os.path.join(BASE_DIR, "refs", f"{folder_name}.*")
        files = glob.glob(path)
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None: templates.append(img)
    return templates

# --- PASS 1: Extraction ---
def extract_telemetry(video_path, progress_callback):
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    temps_in = load_templates("probein")
    
    telemetry_data = []
    
    # ROI
    g_x1, g_x2 = int(width * 0.15), int(width * 0.85)
    g_y1, g_y2 = int(height * 0.15), int(height * 0.85)
    mid_x = int((g_x2 - g_x1) / 2)
    
    frame_idx = 0
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_roi = prev_gray[g_y1:g_y2, g_x1:g_x2]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_roi = gray[g_y1:g_y2, g_x1:g_x2]
        
        # 1. Flow
        flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx = flow[..., 0]
        mag = np.sqrt(fx**2 + flow[..., 1]**2)
        active = mag > 1.0 
        flow_x = np.median(fx[active]) if np.any(active) else 0.0
        
        # 2. Zoom
        f_left = fx[:, :mid_x]
        f_right = fx[:, mid_x:]
        val_l = np.median(f_left[np.abs(f_left)>0.5]) if np.any(np.abs(f_left)>0.5) else 0.0
        val_r = np.median(f_right[np.abs(f_right)>0.5]) if np.any(np.abs(f_right)>0.5) else 0.0
        zoom_score = val_r - val_l
        
        # 3. 4-Corner
        h_roi, w_roi = curr_roi.shape
        mid_h, mid_w = h_roi // 2, w_roi // 2
        q_mag = mag 
        act_tl = np.mean(q_mag[:mid_h, :mid_w])
        act_tr = np.mean(q_mag[:mid_h, mid_w:])
        act_bl = np.mean(q_mag[mid_h:, :mid_w])
        act_br = np.mean(q_mag[mid_h:, mid_w:])
        
        # 4. Fuel (Wide Search)
        score_in = 0.0
        fuel_box = None 
        
        results = model.track(frame, persist=True, classes=[2], verbose=False, conf=0.15)
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            valid_indices = [i for i, b in enumerate(boxes) if b[1] < height * 0.9]
            if valid_indices:
                valid_boxes = boxes[valid_indices]
                largest_idx = np.argmax(valid_boxes[:, 2] * valid_boxes[:, 3])
                cx, cy, cw, ch = valid_boxes[largest_idx]
                
                margin = 20
                x1 = int(cx - cw/2) - margin
                y1 = int(cy - ch/2) - margin
                x2 = int(cx + cw/2) + margin
                y2 = int(cy + ch/2) + margin
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                if (x2 > x1) and (y2 > y1):
                    fuel_zone = gray[y1:y2, x1:x2]
                    fuel_box = (x1, y1, x2-x1, y2-y1)
                    for t in temps_in:
                        if fuel_zone.shape[0] >= t.shape[0] and fuel_zone.shape[1] >= t.shape[1]:
                            res = cv2.matchTemplate(fuel_zone, t, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, _ = cv2.minMaxLoc(res)
                            if max_val > score_in: score_in = max_val

        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Flow_X": flow_x,
            "Zoom_Score": zoom_score,
            "S_In": score_in,
            "Fuel_Box": fuel_box, 
            "Act_TL": act_tl, "Act_TR": act_tr,
            "Act_BL": act_bl, "Act_BR": act_br
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
    
    # Find End
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

# --- PASS 2: Analysis V43 ---
def analyze_states_v43(df, fps):
    window = 15
    cols = ['Flow_X', 'Zoom_Score', 'S_In', 'Act_TL', 'Act_TR', 'Act_BL', 'Act_BR']
    for col in cols:
        if len(df) > window:
            df[f'{col}_Sm'] = savgol_filter(df[col], window, 3)
        else:
            df[f'{col}_Sm'] = df[col]

    df['Zoom_Vel'] = np.gradient(df['Zoom_Score_Sm'])

    # 1. PIT STOP
    x_mag = df['Flow_X_Sm'].abs()
    peaks, _ = find_peaks(x_mag, height=x_mag.max()*0.3, distance=fps*5)
    t_start, t_end = None, None
    arrival_dir = 0 
    
    if len(peaks) >= 2:
        arrival_idx = peaks[0]
        depart_idx = peaks[-1]
        arr_flow = df['Flow_X_Sm'].iloc[arrival_idx]
        arrival_dir = 1 if arr_flow > 0 else -1 
        
        STOP_THRESH = x_mag.max() * 0.05
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

    # 3. CORNER TIMING (Cross-Check Logic)
    map_corners = {}
    if arrival_dir > 0: 
        map_corners['Inside Rear'] = 'Act_BL_Sm'; map_corners['Outside Rear'] = 'Act_TL_Sm'
        map_corners['Inside Front'] = 'Act_BR_Sm'; map_corners['Outside Front'] = 'Act_TR_Sm'
    else: 
        map_corners['Inside Rear'] = 'Act_BR_Sm'; map_corners['Outside Rear'] = 'Act_TR_Sm'
        map_corners['Inside Front'] = 'Act_BL_Sm'; map_corners['Outside Front'] = 'Act_TL_Sm'

    corner_stats = {}
    if t_up and t_down:
        t_ae = t_end
        # Jacks Window
        df_j = df[(df['Time'] >= t_up) & (df['Time'] <= t_ae)]
        times_j = df_j['Time'].values
        
        # A. FRONT
        of = df_j[map_corners['Outside Front']].values
        if_ = df_j[map_corners['Inside Front']].values
        ts_of, te_of = find_active_window(times_j, of, t_up, t_ae, fps)
        ts_if, te_if = find_active_window(times_j, if_, ts_of+1.5, t_ae, fps)
        corner_stats['Outside Front'] = (ts_of, te_of)
        corner_stats['Inside Front'] = (ts_if, te_if)
        
        # B. REAR (Cross-Check Fix)
        ir = df_j[map_corners['Inside Rear']].values
        or_ = df_j[map_corners['Outside Rear']].values
        
        # 1. Calculate Outside Rear First (It's cleaner - no fueler)
        # It can only start after a reasonable gap from Jacks Up
        ts_or, te_or = find_active_window(times_j, or_, t_up + 3.0, t_ae, fps)
        
        # 2. Calculate Inside Rear Start normally
        ts_ir, te_ir_raw = find_active_window(times_j, ir, t_up, t_ae, fps)
        
        # 3. FORCE IR END based on OR Start
        # The tire changer cannot be in two places. 
        # If OR starts at X, IR *must* have ended at X minus Transit Time (~1.5s)
        # We ignore the 'raw' end because that's likely the fueler staying behind.
        
        if ts_or > ts_ir:
            # Transit time assumption: 1.5s
            transit_time = 1.5
            te_ir_forced = ts_or - transit_time
            
            # Safety: Don't make it negative length
            if te_ir_forced < ts_ir + 2.0:
                te_ir = ts_ir + 2.0 # Min duration
            else:
                te_ir = te_ir_forced
        else:
            # Fallback if OR detection failed
            te_ir = te_ir_raw
            
        corner_stats['Inside Rear'] = (ts_ir, te_ir)
        corner_stats['Outside Rear'] = (ts_or, te_or)

    # 4. FUEL
    t_fuel_start, t_fuel_end = None, None
    if t_start and t_end:
        fuel_w = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        if not fuel_w.empty:
            s_in = fuel_w['S_In_Sm'].values
            times = fuel_w['Time'].values
            is_fueling = s_in > 0.50
            indices = np.where(is_fueling)[0]
            if len(indices) > int(fps * 2.0): 
                t_fuel_start = times[indices[0]]
                diffs = np.diff(indices)
                splits = np.where(diffs > 20)[0]
                if len(splits) > 0: end_idx = indices[splits[0]]
                else: end_idx = indices[-1]
                t_fuel_end = times[end_idx]
                if t_fuel_end > t_end - 0.5: t_fuel_end = t_end - 0.5

    if t_up is None: t_up = t_start
    if t_down is None: t_down = t_end

    return (t_start, t_end), (t_up, t_down), (t_fuel_start, t_fuel_end), corner_stats

# --- PASS 3: Render ---
def render_overlay(input_path, pit, tires, fuel, corner_data, df, fps, width, height, show_debug, progress_callback):
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
        
        # Timers
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
        
        # UI
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

        # Debug
        if show_debug:
            safe_idx = min(frame_idx, len(df)-1)
            row = df.iloc[safe_idx]
            fb = row['Fuel_Box']
            if fb is not None:
                x, y, w, h = fb
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            sc = row['S_In_Sm']
            cv2.putText(frame, f"Probe: {sc:.2f}", (width-430, 290), 0, 0.6, (0, 255, 255), 2)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)
            
    cap.release()
    out.release()
    return temp_output.name

# --- Main ---
def main():
    st.title("🏁 Pit Stop Analyzer V43")
    st.markdown("### Cross-Check Logic")
    st.info("Inside Rear Timer now cuts off automatically when Outside Rear activity starts (minus transit time). Ignores fueler.")

    show_debug = st.sidebar.checkbox("Show Fuel Debug Overlay", value=False)

    missing = []
    for r in ["probein", "probeout", "emptyfuelport"]:
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
            st.write("Step 1: Extraction (Multi-Template)...")
            df, fps, w, h = extract_telemetry(tfile.name, bar.progress)
            
            st.write("Step 2: Analysis...")
            pit_t, tire_t, fuel_t, corners = analyze_states_v43(df, fps)
            
            if pit_t[0] is None:
                st.error("Could not detect Stop.")
            else:
                st.write("Step 3: Rendering Video...")
                vid_path = render_overlay(tfile.name, pit_t, tire_t, fuel_t, corners, df, fps, w, h, show_debug, bar.progress)
                
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
        
        st.write("### 🛞 Tire Stats")
        t1, t2, t3, t4 = st.columns(4)
        
        def get_dur(k):
            s, e = corners.get(k, (0,0))
            return e - s

        t1.metric("Inside Rear", f"{get_dur('Inside Rear'):.2f}s")
        t2.metric("Outside Front", f"{get_dur('Outside Front'):.2f}s")
        t3.metric("Outside Rear", f"{get_dur('Outside Rear'):.2f}s")
        t4.metric("Inside Front", f"{get_dur('Inside Front'):.2f}s")
        
        st.subheader("Video Result")
        c1, c2 = st.columns([3,1])
        with c1:
            if os.path.exists(vid_path): st.video(vid_path)
        with c2:
            if os.path.exists(vid_path):
                with open(vid_path, 'rb') as f:
                    st.download_button("Download MP4", f, file_name="pitstop_v43.mp4")

if __name__ == "__main__":
    main()
