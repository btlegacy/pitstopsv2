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

def best_match_template(search_area_gray, templates):
    if not templates: return 0.0
    best_score = 0.0
    for t in templates:
        if search_area_gray.shape[0] >= t.shape[0] and search_area_gray.shape[1] >= t.shape[1]:
            res = cv2.matchTemplate(search_area_gray, t, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_score: best_score = max_val
    return best_score

# --- HELPER: FISHEYE CORRECTION ---
def get_dewarp_map(w, h):
    """
    Generates a mapping to undistort a generic wide-angle/fisheye lens.
    """
    # Estimated Coefficients for a generic wide overhead camera (e.g. GoPro)
    # K = Intrinsic Matrix, D = Distortion Coefficients
    # We assume center principal point (w/2, h/2)
    
    K = np.array([[w, 0, w/2],
                  [0, w, h/2],
                  [0, 0, 1]])
    
    # D = [k1, k2, p1, p2] - Negative k1/k2 removes barrel distortion
    D = np.array([-0.35, 0.1, 0, 0])
    
    # Generate optimal matrix to keep all pixels (alpha=1)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 1, (w,h))
    
    # Create lookup maps (fast remapping)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w,h), 5)
    return map1, map2

# --- PASS 1: Extraction (With De-Warping) ---
def extract_telemetry(video_path, progress_callback):
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 1. Generate De-Warp Map
    map1, map2 = get_dewarp_map(width, height)
    
    temps_in = load_templates("probein")
    temps_out = load_templates("probeout")
    
    telemetry_data = []
    
    # ROI (applied to DEWARPED frame)
    # Since de-warping stretches corners, we can be slightly more generous with margins
    g_x1, g_x2 = int(width * 0.15), int(width * 0.85)
    g_y1, g_y2 = int(height * 0.15), int(height * 0.85)
    mid_x = int((g_x2 - g_x1) / 2)
    
    # Init Optical Flow
    ret, prev_frame_raw = cap.read()
    # De-warp first frame
    prev_frame = cv2.remap(prev_frame_raw, map1, map2, cv2.INTER_LINEAR)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_roi = prev_gray[g_y1:g_y2, g_x1:g_x2]
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame_raw = cap.read()
        if not ret: break
        
        # --- DE-WARP ---
        frame = cv2.remap(frame_raw, map1, map2, cv2.INTER_LINEAR)
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
        
        # 4. Fuel (Spatial Overlap on DEWARPED Frame)
        score_in = 0.0
        fuel_box = None
        fueler_overlap = 0 
        
        results = model.track(frame, persist=True, classes=[0, 2], verbose=False, conf=0.15)
        
        car_box = None
        person_boxes = []
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            cls = results[0].boxes.cls.cpu().numpy()
            
            for i, c in enumerate(cls):
                if int(c) == 2: # Car
                    if boxes[i][1] < height * 0.95:
                        area = boxes[i][2] * boxes[i][3]
                        if car_box is None or area > (car_box[2]*car_box[3]):
                            car_box = boxes[i]
                elif int(c) == 0: # Person
                    person_boxes.append(boxes[i])
        
        if car_box is not None:
            cx, cy, cw, ch = car_box
            
            # Defined Fuel Pit: Center-Bottom of Car
            # With fisheye gone, this rectangle is much more accurate to the physical space
            w_fuel = cw * 0.3
            h_fuel = ch * 0.5
            
            fx1 = int(cx - w_fuel/2)
            fy1 = int(cy)
            fx2 = int(cx + w_fuel/2)
            fy2 = int(cy + h_fuel/2)
            
            fuel_box = (fx1, fy1, int(w_fuel), int(h_fuel))
            
            # A. Person Overlap
            for p in person_boxes:
                px, py, pw, ph = p
                overlap_x = max(0, min(fx2, px+pw/2) - max(fx1, px-pw/2))
                overlap_y = max(0, min(fy2, py+ph/2) - max(fy1, py-ph/2))
                if (overlap_x * overlap_y) > 500: 
                    fueler_overlap = 1
                    break
            
            # B. Template Match
            sx1, sy1 = max(0, fx1-20), max(0, fy1-20)
            sx2, sy2 = min(width, fx2+20), min(height, fy2+20)
            
            if sx2 > sx1 and sy2 > sy1:
                fuel_zone = gray[sy1:sy2, sx1:sx2]
                score_in = best_match_template(fuel_zone, temps_in)

        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Flow_X": flow_x,
            "Zoom_Score": zoom_score,
            "Probe_Match": score_in,
            "Fueler_Overlap": fueler_overlap,
            "Fuel_Box": fuel_box,
            "Act_TL": act_tl, "Act_TR": act_tr,
            "Act_BL": act_bl, "Act_BR": act_br
        })
        
        prev_roi = curr_roi
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)

    cap.release()
    return pd.DataFrame(telemetry_data), fps, width, height

# --- PASS 2: Analysis V71 (Hysteresis Logic) ---
def analyze_states_v71(df, fps):
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

        # A. FRONT
        of = df_j[map_corners['Outside Front']].values
        if_ = df_j[map_corners['Inside Front']].values
        t_of_start, t_of_end = get_window(of, t_up, t_ae, 0.3)
        t_if_start, t_if_end = get_window(if_, t_of_start+1.5, t_ae, 0.3)
        corner_stats['Outside Front'] = (t_of_start, t_of_end)
        corner_stats['Inside Front'] = (t_if_start, t_if_end)
        
        # B. REAR
        ir = df_j[map_corners['Inside Rear']].values
        or_ = df_j[map_corners['Outside Rear']].values
        t_or_start, t_or_end = get_window(or_, t_up+2.5, t_ae, 0.55)
        t_ir_start, t_ir_raw_end = get_window(ir, t_up, t_ae, 0.15)
        
        if t_or_start > t_up + 3.0:
            if t_ir_raw_end < t_or_start: t_ir_end = t_ir_raw_end
            else: t_ir_end = t_or_start - 1.0
            t_trans_start = t_ir_end
            t_trans_end = t_or_start
        else:
            t_ir_end = t_ir_raw_end
            t_trans_start, t_trans_end = t_ir_end, t_ir_end
            
        corner_stats['Inside Rear'] = (t_ir_start, t_ir_end)
        corner_stats['Rear Transition'] = (t_trans_start, t_trans_end)
        corner_stats['Outside Rear'] = (t_or_start, t_or_end)

    # 4. FUEL (V71 Overlap + Probe Logic)
    t_fuel_start, t_fuel_end = None, None
    if t_start and t_end:
        fuel_w = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        if not fuel_w.empty:
            probe_val = fuel_w['Probe_Match_Sm'].values
            overlap = fuel_w['Fueler_Overlap'].values
            times = fuel_w['Time'].values
            
            # Fueling if: (Person Overlap AND Probe>0.4) OR (Probe>0.55)
            is_fueling = ((overlap > 0) & (probe_val > 0.4)) | (probe_val > 0.55)
            
            indices = np.where(is_fueling)[0]
            
            if len(indices) > 0:
                gap_limit = int(fps * 1.5)
                t_start_cand = times[indices[0]]
                t_end_cand = times[indices[0]]
                last_idx = indices[0]
                
                for i in range(1, len(indices)):
                    curr_idx = indices[i]
                    if (curr_idx - last_idx) < gap_limit:
                        last_idx = curr_idx
                    else:
                        current_dur = (times[last_idx] - t_start_cand)
                        if current_dur > 2.0:
                            t_end_cand = times[last_idx]
                            break 
                        else:
                            t_start_cand = times[curr_idx]
                            last_idx = curr_idx
                            
                t_end_cand = times[last_idx]
                
                if (t_end_cand - t_start_cand) > 2.0:
                    t_fuel_start = t_start_cand
                    t_fuel_end = t_end_cand
                    if t_fuel_end > t_end - 0.5: t_fuel_end = t_end - 0.5

    if t_up is None: t_up = t_start
    if t_down is None: t_down = t_end

    return (t_start, t_end), (t_up, t_down), (t_fuel_start, t_fuel_end), corner_stats

# --- PASS 3: Render ---
def render_overlay(input_path, pit, tires, fuel, corner_data, df, fps, width, height, show_debug, progress_callback):
    cap = cv2.VideoCapture(input_path)
    # IMPORTANT: Generate De-Warp Map once here for renderer
    map1, map2 = get_dewarp_map(width, height)
    
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
        ret, frame_raw = cap.read()
        if not ret: break
        
        # DE-WARP FOR RENDER
        frame = cv2.remap(frame_raw, map1, map2, cv2.INTER_LINEAR)
        
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
        for i, label in enumerate(labels):
            c_start, c_end = corner_data.get(label, (0.0, 0.0))
            c_val = 0.0
            if c_start > 0 and curr >= c_start:
                val = (c_end - c_start) if (curr >= c_end) else (curr - c_start)
                c_val = max(0.0, val)
            txt_col = (255,255,255) if c_val > 0 else (150,150,150)
            y_pos = start_y + (i*gap_y)
            cv2.putText(frame, label, (width-430, y_pos), 0, 0.6, txt_col, 1)
            cv2.putText(frame, f"{c_val:.2f}s", (width-100, y_pos), 0, 0.6, txt_col, 2)

        if show_debug:
            safe_idx = min(frame_idx, len(df)-1)
            row = df.iloc[safe_idx]
            sc = row.get('Probe_Match_Sm', 0.0)
            ov = row.get('Fueler_Overlap', 0)
            cv2.putText(frame, f"P:{sc:.2f} O:{ov}", (width-430, 330), 0, 0.5, (0, 255, 255), 1)
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
    st.title("🏁 Pit Stop Analyzer V72")
    st.markdown("### Fisheye Correction Layer")
    st.info("De-warps video first. Fuel logic now correctly targets the 'flat' vehicle side.")

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
            st.write("Step 1: Extraction (De-Warped)...")
            df, fps, w, h = extract_telemetry(tfile.name, bar.progress)
            
            st.write("Step 2: Analysis...")
            pit_t, tire_t, fuel_t, corners = analyze_states_v71(df, fps)
            
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
                    st.download_button("Download MP4", f, file_name="pitstop_v72.mp4")

if __name__ == "__main__":
    main()
