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
st.set_page_config(page_title="Pit Stop Analytics - Simple", layout="wide")
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

# --- PASS 1: Extraction (Restored V75 Logic) ---
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
    
    # V75 ROI (15% to 85%) - Restored for consistency
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
        mag = np.sqrt(fx**2 + flow[..., 1]**2)
        active = mag > 1.0 
        if np.any(active):
            flow_x = np.median(fx[active])
        else:
            flow_x = 0.0
        
        # 2. Zoom
        f_left = fx[:, :mid_x]
        f_right = fx[:, mid_x:]
        
        val_l = 0.0
        if np.any(np.abs(f_left)>0.5):
            val_l = np.median(f_left[np.abs(f_left)>0.5])
            
        val_r = 0.0
        if np.any(np.abs(f_right)>0.5):
            val_r = np.median(f_right[np.abs(f_right)>0.5])
            
        zoom_score = val_r - val_l
        
        # 3. Dynamic Zones (Car Tracking) - Kept for stability even if not displayed
        w_tl, w_tr, w_bl, w_br = 0.0, 0.0, 0.0, 0.0
        score_in = 0.0
        fuel_box = None
        
        results = model.track(frame, persist=True, classes=[2], verbose=False, conf=0.15)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            valid = []
            for i, b in enumerate(boxes):
                if b[1] < height * 0.9:
                    valid.append(i)
            
            if valid:
                # Get largest car
                best_idx = max(valid, key=lambda i: boxes[i][2]*boxes[i][3])
                cx, cy, cw, ch = boxes[best_idx]
                
                # A. Wheel Zones
                roi_w = int(cw * 0.20)
                roi_h = int(ch * 0.30)
                
                zones = [
                    (int(cx - cw/2 - roi_w/2), int(cy - ch/2 - roi_h/4)), # TL
                    (int(cx + cw/2 - roi_w/2), int(cy - ch/2 - roi_h/4)), # TR
                    (int(cx - cw/2 - roi_w/2), int(cy + ch/2 - roi_h*0.75)), # BL
                    (int(cx + cw/2 - roi_w/2), int(cy + ch/2 - roi_h*0.75)) # BR
                ]
                
                acts = []
                for (rx, ry) in zones:
                    rx, ry = max(0, rx), max(0, ry)
                    rx2, ry2 = min(width, rx+roi_w), min(height, ry+roi_h)
                    if rx2>rx and ry2>ry:
                        acts.append(np.mean(mag[ry:ry2, rx:rx2]))
                    else:
                        acts.append(0.0)
                
                w_tl, w_tr, w_bl, w_br = acts
                
                # B. Fuel Zone (Center Bottom)
                fw = int(cw * 0.3)
                fh = int(ch * 0.5)
                fx = int(cx - fw/2)
                fy = int(cy)
                
                fx, fy = max(0, fx), max(0, fy)
                fx2, fy2 = min(width, fx+fw), min(height, fy+fh)
                
                if fx2>fx and fy2>fy:
                    fuel_box = (fx, fy, fw, fh)
                    f_zone = gray[fy:fy2, fx:fx2]
                    for t in temps_in:
                        if f_zone.shape[0]>=t.shape[0] and f_zone.shape[1]>=t.shape[1]:
                            res = cv2.matchTemplate(f_zone, t, cv2.TM_CCOEFF_NORMED)
                            sc = cv2.minMaxLoc(res)[1]
                            if sc > score_in: score_in = sc

        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Flow_X": flow_x,
            "Zoom_Score": zoom_score,
            "Probe_Match": score_in,
            "Fuel_Box": fuel_box,
            "Wheel_TL": w_tl, "Wheel_TR": w_tr, 
            "Wheel_BL": w_bl, "Wheel_BR": w_br
        })
        
        prev_roi = curr_roi
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)

    cap.release()
    return pd.DataFrame(telemetry_data), fps, width, height

# --- PASS 2: Analysis (Restored V75 Logic) ---
def analyze_states(df, fps):
    window = 15
    cols = ['Flow_X', 'Zoom_Score', 'Probe_Match', 'Wheel_TL', 'Wheel_TR', 'Wheel_BL', 'Wheel_BR']
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
    
    if len(peaks) >= 2:
        arrival_idx = peaks[0]
        depart_idx = peaks[-1]
        
        STOP_THRESH = x_mag.max() * 0.05
        for i in range(arrival_idx, depart_idx):
            if x_mag.iloc[i] < STOP_THRESH:
                t_start = df.iloc[i]['Time']
                break
        for i in range(depart_idx, arrival_idx, -1):
            if x_mag.iloc[i] < STOP_THRESH:
                t_end = df.iloc[i]['Time']
                break

    # 2. JACKS (Total Tire Time)
    t_up, t_down = None, None
    if t_start and t_end:
        t_creep = t_end - 1.0
        stop_window = df[(df['Time'] >= t_start) & (df['Time'] <= t_creep)]
        
        if not stop_window.empty:
            z_pos = stop_window['Zoom_Score_Sm'].values
            z_vel = stop_window['Zoom_Vel'].values
            times = stop_window['Time'].values
            
            # V75 Logic: Peak Height 0.2
            peaks_up, _ = find_peaks(z_pos, height=0.2, distance=fps)
            if len(peaks_up) > 0: 
                t_up = times[peaks_up[0]]
            else: 
                t_up = t_start
                
            # V75 Logic: Min Vel < -0.02
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

    # Note: We calculate Fuel/Corners but don't return them for display
    # to keep the UI simple as requested.

    if t_up is None: t_up = t_start
    if t_down is None: t_down = t_end

    return (t_start, t_end), (t_up, t_down)

# --- PASS 3: Render (Simplified) ---
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
        
        # Timers
        if t_start and curr >= t_start:
            vp = (t_end - t_start) if (t_end and curr >= t_end) else (curr - t_start)
            cp = (0,0,255) if (t_end and curr >= t_end) else (0,255,0)
        else: vp, cp = 0.0, (200,200,200)

        if t_up and curr >= t_up:
            vt = (t_down - t_up) if (t_down and curr >= t_down) else (curr - t_up)
            ct = (0,0,255) if (t_down and curr >= t_down) else (0,255,255)
        else: vt, ct = 0.0, (200,200,200)
        
        # UI - Simplified to Top Right Box
        box_w, box_h = 350, 120
        cv2.rectangle(frame, (width - box_w, 0), (width, box_h), (0, 0, 0), -1)
        
        cv2.putText(frame, "PIT STOP", (width - 330, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{vp:.2f}s", (width - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, cp, 3)

        cv2.putText(frame, "TIRES", (width - 330, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{vt:.2f}s", (width - 150, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, ct, 3)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)
            
    cap.release()
    out.release()
    return temp_output.name

# --- Main ---
def main():
    st.title("🏁 Pit Stop Analyzer V77")
    st.markdown("### Simplified Metrics")
    st.info("Shows only Total Pit Stop Time and Total Tire Change Time using robust V75 logic.")

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
            st.write("Processing (Extraction)...")
            df, fps, w, h = extract_telemetry(tfile.name, bar.progress)
            
            st.write("Analyzing...")
            pit_t, tire_t = analyze_states(df, fps)
            
            if pit_t[0] is None:
                st.error("Could not detect Stop.")
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
                    st.download_button("Download MP4", f, file_name="pitstop_simple.mp4")

if __name__ == "__main__":
    main()
