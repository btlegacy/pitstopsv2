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
        
        # 1. Flow
        flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx = flow[..., 0]
        fy = flow[..., 1]
        mag = np.sqrt(fx**2 + fy**2)
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
        
        # Vertical Flow per corner (Direction)
        q_fy = fy
        fy_tl = np.mean(q_fy[:mid_h, :mid_w])
        fy_tr = np.mean(q_fy[:mid_h, mid_w:])
        fy_bl = np.mean(q_fy[mid_h:, :mid_w])
        fy_br = np.mean(q_fy[mid_h:, mid_w:])
        
        # 4. Fuel
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
            "Act_BL": act_bl, "Act_BR": act_br,
            "Fy_TL": fy_tl, "Fy_TR": fy_tr, 
            "Fy_BL": fy_bl, "Fy_BR": fy_br
        })
        
        prev_roi = curr_roi
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)

    cap.release()
    return pd.DataFrame(telemetry_data), fps, width, height

# --- PASS 2: Analysis V53 (Gun-Spike Logic) ---
def analyze_states_v53(df, fps):
    window = 15
    cols = ['Flow_X', 'Zoom_Score', 'S_In', 'Act_TL', 'Act_TR', 'Act_BL', 'Act_BR',
            'Fy_TL', 'Fy_TR', 'Fy_BL', 'Fy_BR']
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
    def get_window(sig, start_g, end_g, sens):
        mask = (df['Time'] >= start_g) & (df['Time'] <= end_g)
        if not np.any(mask): return start_g, start_g
        s_win = sig[mask]
        t_win = df['Time'][mask].values
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

    # Specific "Gun Spike" Detection for Outside Rear
    def find_gun_start(sig, start_g, end_g):
        # Look for sudden spike in Derivative (Acceleration of activity)
        mask = (df['Time'] >= start_g) & (df['Time'] <= end_g)
        if not np.any(mask): return start_g
        
        s_win = sig[mask]
        t_win = df['Time'][mask].values
        
        # Rate of change
        grad = np.gradient(s_win)
        
        # Find the first major positive spike in activity (Gun On)
        # Threshold: The spike must be sharp (> 30% of max gradient)
        max_grad = np.max(grad)
        thresh_spike = max_grad * 0.3
        
        spikes = np.where(grad > thresh_spike)[0]
        
        if len(spikes) > 0:
            return t_win[spikes[0]]
        else:
            # Fallback to standard threshold
            t_s, _ = get_window(sig, start_g, end_g, 0.5)
            return t_s

    map_act = {}
    map_fy = {}
    if arrival_dir > 0: 
        map_act['Inside Rear'] = 'Act_BL_Sm'; map_fy['Inside Rear'] = 'Fy_BL_Sm'
        map_act['Outside Rear'] = 'Act_TL_Sm'; map_fy['Outside Rear'] = 'Fy_TL_Sm'
        map_act['Inside Front'] = 'Act_BR_Sm'; map_fy['Inside Front'] = 'Fy_BR_Sm'
        map_act['Outside Front'] = 'Act_TR_Sm'; map_fy['Outside Front'] = 'Fy_TR_Sm'
    else: 
        map_act['Inside Rear'] = 'Act_BR_Sm'; map_fy['Inside Rear'] = 'Fy_BR_Sm'
        map_act['Outside Rear'] = 'Act_TR_Sm'; map_fy['Outside Rear'] = 'Fy_TR_Sm'
        map_act['Inside Front'] = 'Act_BL_Sm'; map_fy['Inside Front'] = 'Fy_BL_Sm'
        map_act['Outside Front'] = 'Act_TL_Sm'; map_fy['Outside Front'] = 'Fy_TL_Sm'

    corner_stats = {}
    if t_up and t_down:
        t_ae = t_end
        
        # Front
        of = df[map_act['Outside Front']].values
        if_ = df[map_act['Inside Front']].values
        ts_of, te_of = get_window(of, t_up, t_ae, 0.3)
        ts_if, te_if = get_window(if_, ts_of+1.5, t_ae, 0.3)
        corner_stats['Outside Front'] = (ts_of, te_of)
        corner_stats['Inside Front'] = (ts_if, te_if)
        
        # Rear
        ir = df[map_act['Inside Rear']].values
        or_ = df[map_act['Outside Rear']].values
        fy_ir = df[map_fy['Inside Rear']].values
        
        # 1. IR Start (Sensitive)
        ts_ir, _ = get_window(ir, t_up, t_ae, 0.15)
        
        # 2. OR Start (GUN SPIKE LOGIC)
        # Look for the "Gun On" moment specifically
        ts_or = find_gun_start(or_, t_up + 3.0, t_ae)
        
        # Find OR End
        _, te_or = get_window(or_, ts_or, t_ae, 0.3)
        
        # 3. IR End (Movement Vector / Transition)
        # Look for upward movement in the gap
        search_end = ts_or
        search_start = max(ts_ir + 1.5, search_end - 2.0) # Look in 2s window before OR start
        
        mask_gap = (df['Time'] >= search_start) & (df['Time'] <= search_end)
        if np.any(mask_gap):
            fy_gap = fy_ir[mask_gap]
            t_gap = df['Time'][mask_gap].values
            
            # Find peak upward movement
            min_idx = np.argmin(fy_gap)
            if fy_gap[min_idx] < -0.5:
                te_ir = t_gap[min_idx]
            else:
                # If no clear movement vector, assume transition took ~1.2s
                te_ir = max(ts_ir+2.0, ts_or - 1.2)
        else:
            te_ir = ts_or - 1.2
            
        # Safety Clamps
        if te_ir > ts_or: te_ir = ts_or - 0.5
        if te_ir < ts_ir + 2.0: te_ir = ts_ir + 2.0
        
        corner_stats['Inside Rear'] = (ts_ir, te_ir)
        corner_stats['Rear Transition'] = (te_ir, ts_or)
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
                end_idx = indices[splits[0]] if len(splits) > 0 else indices[-1]
                t_fuel_end = times[end_idx]
                if t_fuel_end > t_end - 0.5: t_fuel_end = t_end - 0.5

    if t_up is None: t_up = t_start
    if t_down is None: t_down = t_end

    return (t_start, t_end), (t_up, t_down), (t_fuel_start, t_fuel_end), corner_stats, df

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
        cv2.rectangle(frame, (width-450, 0), (width, 360), (0,0,0), -1)
        cv2.putText(frame, "PIT STOP", (width-430, 40), 0, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{vp:.2f}s", (width-180, 40), 0, 1.2, cp, 3)
        cv2.putText(frame, "FUELING", (width-430, 90), 0, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{vf:.2f}s", (width-180, 90), 0, 1.2, cf, 3)
        cv2.putText(frame, "TIRES (Total)", (width-430, 140), 0, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{vt:.2f}s", (width-180, 140), 0, 1.2, ct, 3)

        start_y = 180
        gap_y = 30
        trans_start, trans_end = corner_data.get("Rear Transition", (0,0))
        
        labels_ui = [
            ("Inside Rear", "Inside Rear"),
            ("TRANSITION", "  > Transition"),
            ("Outside Rear", "Outside Rear"),
            ("Outside Front", "Outside Front"),
            ("Inside Front", "Inside Front")
        ]
        
        for i, (key, display) in enumerate(labels_ui):
            y_pos = start_y + (i*gap_y)
            if key == "TRANSITION":
                c_start, c_end = trans_start, trans_end
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

        if show_debug:
            safe_idx = min(frame_idx, len(df)-1)
            row = df.iloc[safe_idx]
            sc = row.get('S_In_Sm', 0.0)
            ppl = row.get('Fueler_Present_Sm', 0.0)
            cv2.putText(frame, f"P:{sc:.2f} H:{ppl:.1f}", (width-430, 330), 0, 0.5, (0, 255, 255), 1)
            
            fb = row.get('Fuel_Box')
            if fb is not None and not pd.isna(fb):
                x,y,w,h = fb
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)
            
    cap.release()
    out.release()
    return temp_output.name

# --- Main ---
def main():
    st.title("🏁 Pit Stop Analyzer V53")
    st.markdown("### Gun-Spike Logic")
    st.info("Detects 'Gun On' for Outside Rear (Sharp Activity Spike) to extend transition time correctly.")

    show_debug = st.sidebar.checkbox("Show Debug Info", value=False)

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
            pit_t, tire_t, fuel_t, corners, df_final = analyze_states_v53(df, fps)
            
            if pit_t[0] is None:
                st.error("Could not detect Stop.")
            else:
                st.write("Step 3: Rendering Video...")
                vid_path = render_overlay(tfile.name, pit_t, tire_t, fuel_t, corners, df_final, fps, w, h, show_debug, bar.progress)
                
                st.session_state.update({
                    'df': df_final, 'video_path': vid_path, 
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
        else:
            st.warning("Please re-run analysis.")
            st.stop()
        
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Pit Stop Time", f"{pit_t[1] - pit_t[0]:.2f}s")
        f_dur = (fuel_t[1] - fuel_t[0]) if (fuel_t[0] and fuel_t[1]) else 0
        c2.metric("Fueling Time", f"{f_dur:.2f}s" if f_dur > 0 else "N/A")
        t_dur = (tire_t[1] - tire_t[0]) if (tire_t[0] and tire_t[1]) else 0
        c3.metric("Tire Change Time", f"{t_dur:.2f}s")
        
        st.write("### 🛞 Rear Breakdown")
        c_ir = corners.get('Inside Rear', (0,0))
        c_or = corners.get('Outside Rear', (0,0))
        c_tr = corners.get('Rear Transition', (0,0))
        
        t1, t2, t3 = st.columns(3)
        t1.metric("Inside Rear", f"{c_ir[1]-c_ir[0]:.2f}s")
        t2.metric("Transition", f"{c_tr[1]-c_tr[0]:.2f}s")
        t3.metric("Outside Rear", f"{c_or[1]-c_or[0]:.2f}s")
        
        st.subheader("Video Result")
        c1, c2 = st.columns([3,1])
        with c1:
            if os.path.exists(vid_path): st.video(vid_path)
        with c2:
            if os.path.exists(vid_path):
                with open(vid_path, 'rb') as f:
                    st.download_button("Download MP4", f, file_name="pitstop_v53.mp4")

if __name__ == "__main__":
    main()
