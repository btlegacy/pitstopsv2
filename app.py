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
from collections import Counter

# --- Configuration ---
st.set_page_config(page_title="Pit Stop Analytics AI", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    # Reverted to Standard YOLO to ensure Car Detection works (Pose model ignores cars)
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

# --- PASS 1: Extraction (Flow + Latching) ---
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
    
    g_x1, g_x2 = int(width * 0.15), int(width * 0.85)
    g_y1, g_y2 = int(height * 0.15), int(height * 0.85)
    mid_x = int((g_x2 - g_x1) / 2)
    
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_roi = prev_gray[g_y1:g_y2, g_x1:g_x2]
    
    frame_idx = 0
    
    # Car Latching
    latched_car_box = None
    car_stable_count = 0
    
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
        
        # 4. Detection (Standard YOLO)
        score_in = 0.0
        current_car_box = None
        person_tracks = []
        
        results = model.track(frame, persist=True, classes=[0, 2], verbose=False, conf=0.15)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            cls = results[0].boxes.cls.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            
            # A. Find Car
            car_indices = [i for i, c in enumerate(cls) if int(c) == 2]
            if car_indices:
                valid_cars = [i for i in car_indices if boxes[i][1] < height * 0.9]
                if valid_cars:
                    best_car = max(valid_cars, key=lambda i: boxes[i][2]*boxes[i][3])
                    current_car_box = boxes[best_car] # cx, cy, w, h
                    
                    # Latch Logic: If stationary
                    if abs(flow_x) < 0.5:
                        car_stable_count += 1
                        if car_stable_count > 5: latched_car_box = current_car_box
                    else:
                        car_stable_count = 0

            # Use Latched if lost
            if current_car_box is None and latched_car_box is not None:
                current_car_box = latched_car_box
                
            # Fuel Search
            if current_car_box is not None:
                cx, cy, cw, ch = current_car_box
                margin = 20
                x1 = int(cx - cw/2) - margin
                y1 = int(cy - ch/2) - margin
                x2 = int(cx + cw/2) + margin
                y2 = int(cy + ch/2) + margin
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                if (x2 > x1) and (y2 > y1):
                    fuel_zone = gray[y1:y2, x1:x2]
                    for t in temps_in:
                        if fuel_zone.shape[0]>=t.shape[0] and fuel_zone.shape[1]>=t.shape[1]:
                            res = cv2.matchTemplate(fuel_zone, t, cv2.TM_CCOEFF_NORMED)
                            sc = cv2.minMaxLoc(res)[1]
                            if sc > score_in: score_in = sc

            # B. People Tracks
            for i, c in enumerate(cls):
                if int(c) == 0:
                    if boxes[i][1] < height * 0.95:
                        # Store: ID, cx, cy, w, h
                        person_tracks.append((int(ids[i]), boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]))

        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Flow_X": flow_x,
            "Zoom_Score": zoom_score,
            "Probe_Match": score_in,
            "Act_TL": act_tl, "Act_TR": act_tr, "Act_BL": act_bl, "Act_BR": act_br,
            "Person_Tracks": person_tracks,
            "Car_Box": current_car_box 
        })
        
        prev_roi = curr_roi
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)

    cap.release()
    return pd.DataFrame(telemetry_data), fps, width, height

# --- PASS 2: Analysis V67 ---
def analyze_states_v67(df, fps, width, height):
    window = 15
    cols = ['Flow_X', 'Zoom_Score', 'Probe_Match', 'Act_TL', 'Act_TR', 'Act_BL', 'Act_BR']
    for col in cols:
        if col in df.columns:
            if len(df) > window:
                df[f'{col}_Sm'] = savgol_filter(df[col], window, 3)
            else:
                df[f'{col}_Sm'] = df[col]
        else:
            df[f'{col}_Sm'] = 0.0

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
        
        for i in range(arrival_idx, depart_idx):
            if x_mag.iloc[i] < x_mag.max()*0.05:
                t_start = df.iloc[i]['Time']
                break
        for i in range(depart_idx, arrival_idx, -1):
            if x_mag.iloc[i] < x_mag.max()*0.05:
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
            if len(peaks_up) > 0: t_up = times[peaks_up[0]]
            else: t_up = t_start
                
            mask_drop = times > t_up + 2.0 
            if np.any(mask_drop):
                drop_vel = z_vel[mask_drop]
                drop_times = times[mask_drop]
                min_idx = np.argmin(drop_vel)
                if drop_vel[min_idx] < -0.02: t_down = drop_times[min_idx]
                else: t_down = t_end
            else: t_down = t_end

    # 3. CORNER TIMING
    map_corners = {}
    if arrival_dir > 0: 
        map_corners['Inside Rear'] = 'Act_BL_Sm'; map_corners['Outside Rear'] = 'Act_TL_Sm'
        map_corners['Inside Front'] = 'Act_BR_Sm'; map_corners['Outside Front'] = 'Act_TR_Sm'
    else: 
        map_corners['Inside Rear'] = 'Act_BR_Sm'; map_corners['Outside Rear'] = 'Act_TR_Sm'
        map_corners['Inside Front'] = 'Act_BL_Sm'; map_corners['Outside Front'] = 'Act_TL_Sm'

    corner_stats = {}
    crew_ids = {'Rear_Changer': None, 'Front_Changer': None}
    
    if t_up and t_down:
        t_ae = t_end 
        df_j = df[(df['Time'] >= t_up) & (df['Time'] <= t_ae)]
        times_j = df_j['Time'].values
        
        def get_window(sig, start_g, end_g, sens=0.3):
            mask = (times_j >= start_g) & (times_j <= end_g)
            if not np.any(mask): return start_g, start_g
            s_win = sig[mask]
            t_win = times_j[mask]
            base = np.percentile(s_win, 10)
            peak = np.max(s_win)
            if peak < 0.5: return start_g, start_g
            t_s = base + (peak-base) * sens
            t_e = base + (peak-base) * 0.20
            active = np.where(s_win > t_s)[0]
            if len(active) == 0: return start_g, start_g
            i_start = active[0]
            peak_idx = np.argmax(s_win)
            search_s = max(i_start, peak_idx)
            i_end = len(s_win)-1
            buf = int(fps*0.5)
            for i in range(search_s, len(s_win)-buf):
                if s_win[i] < t_e and np.mean(s_win[i:i+buf]) < t_e:
                    i_end = i
                    break
            return t_win[i_start], t_win[i_end]

        def find_gun_start(sig, start_g, end_g):
            mask = (times_j >= start_g) & (times_j <= end_g)
            if not np.any(mask): return start_g
            s_win = sig[mask]
            t_win = times_j[mask]
            grad = np.gradient(s_win)
            thresh = np.max(grad) * 0.3
            spikes = np.where(grad > thresh)[0]
            if len(spikes)>0: return t_win[spikes[0]]
            return get_window(sig, start_g, end_g, 0.5)[0]

        # Front
        of = df_j[map_corners['Outside Front']].values
        if_ = df_j[map_corners['Inside Front']].values
        t_of_start, t_of_end = get_window(of, t_up, t_ae, 0.3)
        t_if_start, t_if_end = get_window(if_, t_of_start+1.5, t_ae, 0.3)
        if t_if_start < t_of_end:
             mid = (t_of_end + t_if_start) / 2
             t_trans_f_start = mid
             t_trans_f_end = mid
        else:
             t_trans_f_start = t_of_end
             t_trans_f_end = t_if_start
        corner_stats['Outside Front'] = (t_of_start, t_of_end)
        corner_stats['Front Transition'] = (t_trans_f_start, t_trans_f_end)
        corner_stats['Inside Front'] = (t_if_start, t_if_end)
        
        # Rear
        ir = df_j[map_corners['Inside Rear']].values
        or_ = df_j[map_corners['Outside Rear']].values
        t_or_start = find_gun_start(or_, t_up + 2.5, t_ae)
        _, t_or_end = get_window(or_, t_or_start, t_ae, 0.2)
        t_ir_start, t_ir_raw_end = get_window(ir, t_up, t_ae, 0.15)
        
        if t_or_start > t_up + 3.0:
            if t_ir_raw_end < t_or_start:
                t_ir_end = t_ir_raw_end
            else:
                t_ir_end = t_or_start - 1.0
                if t_ir_end < t_ir_start + 1.5: t_ir_end = t_ir_start + 1.5
            t_trans_start = t_ir_end
            t_trans_end = t_or_start
        else:
            t_ir_end = t_ir_raw_end
            t_trans_start, t_trans_end = t_ir_end, t_ir_end
            
        corner_stats['Inside Rear'] = (t_ir_start, t_ir_end)
        corner_stats['Rear Transition'] = (t_trans_start, t_trans_end)
        corner_stats['Outside Rear'] = (t_or_start, t_or_end)
        
        # --- ID IDENTIFICATION (Fixed V67) ---
        valid_car_frames = df[df['Car_Box'].notnull()]
        if not valid_car_frames.empty:
            # Use middle frame of stop to define zones
            ref_idx = len(valid_car_frames) // 2
            ref_row = valid_car_frames.iloc[ref_idx]
            cx, cy, cw, ch = ref_row['Car_Box']
            
            margin = 60
            # Define zones based on car corners (extended outward)
            if arrival_dir > 0: # L->R
                # Rear Changer Zone: Bottom-Left Corner of Car
                zr_x1, zr_y1 = int(cx - cw/2 - margin), int(cy + ch/2 - margin)
                zr_x2, zr_y2 = int(cx - cw/2 + margin*2), int(cy + ch/2 + margin*2)
                
                # Front Changer Zone: Top-Right Corner of Car
                zf_x1, zf_y1 = int(cx + cw/2 - margin), int(cy - ch/2 - margin*2)
                zf_x2, zf_y2 = int(cx + cw/2 + margin), int(cy - ch/2 + margin)
            else: # R->L
                # Rear Changer Zone: Bottom-Right
                zr_x1, zr_y1 = int(cx + cw/2 - margin), int(cy + ch/2 - margin)
                zr_x2, zr_y2 = int(cx + cw/2 + margin), int(cy + ch/2 + margin*2)
                
                # Front Changer Zone: Top-Left
                zf_x1, zf_y1 = int(cx - cw/2 - margin), int(cy - ch/2 - margin*2)
                zf_x2, zf_y2 = int(cx - cw/2 + margin), int(cy - ch/2 + margin)
            
            # Helper to find ID in zone
            def find_id(t_s, t_e, box):
                bx1, by1, bx2, by2 = box
                counts = Counter()
                mask = (df['Time'] >= t_s) & (df['Time'] <= t_e)
                rows = df[mask]
                for _, r in rows.iterrows():
                    tracks = r['Person_Tracks']
                    if isinstance(tracks, list):
                        for (pid, px, py, pw, ph) in tracks:
                            # Check if person center is in zone
                            if bx1 < px < bx2 and by1 < py < by2:
                                counts[pid] += 1
                if counts: return counts.most_common(1)[0][0]
                return None

            # Assign IDs
            crew_ids['Rear_Changer'] = find_id(t_ir_start, t_ir_end, (zr_x1, zr_y1, zr_x2, zr_y2))
            crew_ids['Front_Changer'] = find_id(t_of_start, t_of_end, (zf_x1, zf_y1, zf_x2, zf_y2))

    # 4. FUEL
    t_fuel_start, t_fuel_end = None, None
    if t_start and t_end:
        fuel_w = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        matches = fuel_w['Probe_Match_Sm'].values
        if len(matches) > 0 and np.max(matches) > 0.55:
            active = matches > 0.55
            idx = np.where(active)[0]
            if len(idx) > int(fps*2):
                t_fuel_start = fuel_w.iloc[idx[0]]['Time']
                diffs = np.diff(idx)
                splits = np.where(diffs > 20)[0]
                end_idx = idx[splits[0]] if len(splits) > 0 else idx[-1]
                t_fuel_end = fuel_w.iloc[end_idx]['Time']
                if t_fuel_end > t_end - 0.5: t_fuel_end = t_end - 0.5

    if t_up is None: t_up = t_start
    if t_down is None: t_down = t_end

    return (t_start, t_end), (t_up, t_down), (t_fuel_start, t_fuel_end), corner_stats, crew_ids

# --- PASS 3: Render ---
def render_overlay(input_path, pit, tires, fuel, corner_data, crew_ids, df, fps, width, height, show_debug, progress_callback):
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
        
        # Global Timers
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
        cv2.rectangle(frame, (width-450, 0), (width, 400), (0,0,0), -1)
        cv2.putText(frame, "PIT STOP", (width-430, 40), 0, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{vp:.2f}s", (width-180, 40), 0, 1.2, cp, 3)
        cv2.putText(frame, "FUELING", (width-430, 90), 0, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{vf:.2f}s", (width-180, 90), 0, 1.2, cf, 3)
        cv2.putText(frame, "TIRES (Total)", (width-430, 140), 0, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{vt:.2f}s", (width-180, 140), 0, 1.2, ct, 3)

        # Corners
        start_y = 180
        gap_y = 30
        trans_start, trans_end = corner_data.get("Rear Transition", (0,0))
        
        labels_ui = [
            ("Inside Rear", "Inside Rear"),
            ("TRANSITION", "  > Transition"),
            ("Outside Rear", "Outside Rear"),
            ("Outside Front", "Outside Front"),
            ("Front Transition", "  > F-Transition"),
            ("Inside Front", "Inside Front")
        ]
        
        for i, (key, display) in enumerate(labels_ui):
            y_pos = start_y + (i*gap_y)
            
            if "Transition" in key:
                c_start, c_end = corner_data.get(key, (0.0, 0.0))
                label_col = (255, 255, 0)
            else:
                c_start, c_end = corner_data.get(key, (0.0, 0.0))
                label_col = (255, 255, 255)
            
            c_val = 0.0
            if c_start > 0 and curr >= c_start:
                val = (c_end - c_start) if (curr >= c_end) else (curr - c_start)
                c_val = max(0.0, val)
            
            txt_col = label_col if c_val > 0 else (100,100,100)
            cv2.putText(frame, display, (width-430, y_pos), 0, 0.6, txt_col, 1)
            cv2.putText(frame, f"{c_val:.2f}s", (width-100, y_pos), 0, 0.6, txt_col, 2)

        # CREW DRAWING (Fixed Logic)
        if frame_idx < len(df):
            row = df.iloc[frame_idx]
            tracks = row.get('Person_Tracks')
            
            if isinstance(tracks, list):
                for (pid, x, y, w, h) in tracks:
                    lbl = None
                    color = (200,200,200)
                    
                    if pid == crew_ids.get('Rear_Changer'):
                        lbl = "Rear Changer"
                        color = (0, 255, 0)
                    elif pid == crew_ids.get('Front_Changer'):
                        lbl = "Front Changer"
                        color = (0, 165, 255)
                    
                    if lbl:
                        cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color, 2)
                        cv2.putText(frame, lbl, (int(x-w/2), int(y-h/2)-5), 0, 0.5, color, 1)
                    elif show_debug:
                         cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (100,100,100), 1)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)
            
    cap.release()
    out.release()
    return temp_output.name

# --- Main ---
def main():
    st.title("🏁 Pit Stop Analyzer V67")
    st.markdown("### Fixed Tracking & UI")
    st.info("Restored Standard YOLO for Car/Latching. Fixed UI overlay to show all timers.")

    show_debug = st.sidebar.checkbox("Show All IDs (Debug)", value=False)

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
            st.write("Step 1: Extraction (Flow + Standard YOLO)...")
            df, fps, w, h = extract_telemetry(tfile.name, bar.progress)
            
            st.write("Step 2: Analysis (ID Assignment)...")
            pit_t, tire_t, fuel_t, corners, crew_ids = analyze_states_v67(df, fps, w, h)
            
            if pit_t[0] is None:
                st.error("Could not detect Stop.")
            else:
                st.write("Step 3: Rendering Video...")
                vid_path = render_overlay(tfile.name, pit_t, tire_t, fuel_t, corners, crew_ids, df, fps, w, h, show_debug, bar.progress)
                
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
        timings = st.session_state['timings']
        
        if len(timings) == 4:
            pit_t, tire_t, fuel_t, corners = timings
            
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Pit Stop Time", f"{pit_t[1] - pit_t[0]:.2f}s")
            f_dur = (fuel_t[1] - fuel_t[0]) if (fuel_t[0] and fuel_t[1]) else 0
            c2.metric("Fueling Time", f"{f_dur:.2f}s" if f_dur > 0 else "N/A")
            t_dur = (tire_t[1] - tire_t[0]) if (tire_t[0] and tire_t[1]) else 0
            c3.metric("Tire Change Time", f"{t_dur:.2f}s")
            
            st.write("### 🛞 Axle Breakdown")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Rear Axle**")
                st.write(f"Inside Rear: {corners.get('Inside Rear', (0,0))[1] - corners.get('Inside Rear', (0,0))[0]:.2f}s")
                st.write(f"Transition: {corners.get('Rear Transition', (0,0))[1] - corners.get('Rear Transition', (0,0))[0]:.2f}s")
                st.write(f"Outside Rear: {corners.get('Outside Rear', (0,0))[1] - corners.get('Outside Rear', (0,0))[0]:.2f}s")
                
            with c2:
                st.markdown("**Front Axle**")
                st.write(f"Outside Front: {corners.get('Outside Front', (0,0))[1] - corners.get('Outside Front', (0,0))[0]:.2f}s")
                st.write(f"Transition: {corners.get('Front Transition', (0,0))[1] - corners.get('Front Transition', (0,0))[0]:.2f}s")
                st.write(f"Inside Front: {corners.get('Inside Front', (0,0))[1] - corners.get('Inside Front', (0,0))[0]:.2f}s")
            
            st.subheader("Video Result")
            c1, c2 = st.columns([3,1])
            with c1:
                if os.path.exists(vid_path): st.video(vid_path)
            with c2:
                if os.path.exists(vid_path):
                    with open(vid_path, 'rb') as f:
                        st.download_button("Download MP4", f, file_name="pitstop_v67.mp4")
        else:
            st.warning("Please re-run analysis.")

if __name__ == "__main__":
    main()
