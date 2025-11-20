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

# --- PASS 1: 4-Corner Extraction ---
def extract_telemetry(video_path, progress_callback):
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Load Fuel Ref
    ref_probe = load_ref_image("probein")

    telemetry_data = []
    
    # Global ROI (Center)
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
        
        # 1. Global Optical Flow (For Stop/Go/Zoom)
        flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx = flow[..., 0]
        
        mag = np.sqrt(fx**2 + flow[..., 1]**2)
        active = mag > 1.0 
        flow_x = np.median(fx[active]) if np.any(active) else 0.0
        
        # Zoom (Right - Left)
        f_left = fx[:, :mid_x]
        f_right = fx[:, mid_x:]
        val_l = np.median(f_left[np.abs(f_left)>0.5]) if np.any(np.abs(f_left)>0.5) else 0.0
        val_r = np.median(f_right[np.abs(f_right)>0.5]) if np.any(np.abs(f_right)>0.5) else 0.0
        zoom_score = val_r - val_l
        
        # 2. 4-CORNER ACTIVITY DETECTION
        # We divide the screen relative to the center to monitor the 4 quadrants
        # This detects crew movement around the tires.
        # Top-Left (TL), Top-Right (TR), Bottom-Left (BL), Bottom-Right (BR)
        
        # We use magnitude of flow in these quadrants
        h_roi, w_roi = curr_roi.shape
        mid_h, mid_w = h_roi // 2, w_roi // 2
        
        # Quadrant Flow Magnitudes (Mean of active pixels)
        # We measure "Energy" in each corner
        q_mag = mag 
        
        # Top Left
        act_tl = np.mean(q_mag[:mid_h, :mid_w])
        # Top Right
        act_tr = np.mean(q_mag[:mid_h, mid_w:])
        # Bottom Left
        act_bl = np.mean(q_mag[mid_h:, :mid_w])
        # Bottom Right
        act_br = np.mean(q_mag[mid_h:, mid_w:])
        
        # 3. FUEL (Template)
        # Simple scan of center area
        probe_score = 0.0
        if ref_probe is not None:
             # Optimization: Only scan if we are stopped (checked later, but extraction happens now)
             # Scan center ROI
             if curr_roi.shape[0] >= ref_probe.shape[0] and curr_roi.shape[1] >= ref_probe.shape[1]:
                res = cv2.matchTemplate(curr_roi, ref_probe, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                probe_score = max_val

        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Flow_X": flow_x,
            "Zoom_Score": zoom_score,
            "Probe_Match": probe_score,
            "Act_TL": act_tl,
            "Act_TR": act_tr,
            "Act_BL": act_bl,
            "Act_BR": act_br
        })
        
        prev_roi = curr_roi
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)

    cap.release()
    return pd.DataFrame(telemetry_data), fps, width, height

# --- PASS 2: Multi-Tire Analysis ---
def analyze_states_v31(df, fps):
    # Smoothing
    window = 15
    for col in ['Flow_X', 'Zoom_Score', 'Probe_Match', 'Act_TL', 'Act_TR', 'Act_BL', 'Act_BR']:
        if len(df) > window:
            df[f'{col}_Sm'] = savgol_filter(df[col], window, 3)
        else:
            df[f'{col}_Sm'] = df[col]

    df['Zoom_Vel'] = np.gradient(df['Zoom_Score_Sm'])

    # 1. PIT STOP (Direction & Duration)
    x_mag = df['Flow_X_Sm'].abs()
    MOVE_THRESH = x_mag.max() * 0.3 
    STOP_THRESH = x_mag.max() * 0.05 
    
    peaks, _ = find_peaks(x_mag, height=MOVE_THRESH, distance=fps*5)
    t_start, t_end = None, None
    arrival_dir = 0 # +1 Right, -1 Left
    
    if len(peaks) >= 2:
        arrival_idx = peaks[0]
        depart_idx = peaks[-1]
        
        # Detect Direction from Arrival Flow
        # Look at raw flow during arrival peak
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

    # 2. CORNER MAPPING
    # Mapping based on Arrival Direction (assuming Inside=Top, Outside=Bottom)
    # If L->R: Front=Right, Rear=Left
    # If R->L: Front=Left, Rear=Right
    
    # Default: L->R
    # TL=Inside Rear, TR=Inside Front, BL=Outside Rear, BR=Outside Front
    
    map_corners = {}
    if arrival_dir > 0: # Moving Right
        map_corners['Inside Rear'] = 'Act_TL_Sm'
        map_corners['Inside Front'] = 'Act_TR_Sm'
        map_corners['Outside Rear'] = 'Act_BL_Sm'
        map_corners['Outside Front'] = 'Act_BR_Sm'
    else: # Moving Left
        map_corners['Inside Rear'] = 'Act_TR_Sm'
        map_corners['Inside Front'] = 'Act_TL_Sm'
        map_corners['Outside Rear'] = 'Act_BR_Sm'
        map_corners['Outside Front'] = 'Act_BL_Sm'
        
    corner_times = {}
    
    if t_start and t_end:
        stop_df = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        
        # Calculate individual tire durations
        # Logic: Time the activity is above a "Busy" threshold
        for name, col in map_corners.items():
            sig = stop_df[col]
            thresh = sig.mean() + (sig.std() * 0.5) # Dynamic threshold
            active = sig > thresh
            duration = active.sum() / fps
            corner_times[name] = duration

    # 3. TOTAL TIRE CHANGE (Jacks)
    t_up, t_down = None, None
    
    if t_start and t_end:
        t_creep = t_end - 1.0
        stop_window = df[(df['Time'] >= t_start) & (df['Time'] <= t_creep)]
        
        if not stop_window.empty:
            z_pos = stop_window['Zoom_Score_Sm'].values
            z_vel = stop_window['Zoom_Vel'].values
            times = stop_window['Time'].values
            
            # LIFT: First Expansion
            peaks_up, _ = find_peaks(z_pos, height=0.2, distance=fps)
            if len(peaks_up) > 0:
                t_up = times[peaks_up[0]]
            else:
                t_up = t_start
                
            # DROP: Last "Snap" (Negative Velocity)
            # We look for the *Deepest* negative velocity (Fastest drop)
            # But we prioritize the *Last* one if there are multiple heavy drops
            
            mask_drop = times > t_up + 2.0 # Min 2s on jacks
            if np.any(mask_drop):
                drop_vel = z_vel[mask_drop]
                drop_times = times[mask_drop]
                
                # Find all significant snaps
                # A snap is where velocity dips below -0.05
                snap_indices = np.where(drop_vel < -0.05)[0]
                
                if len(snap_indices) > 0:
                    # Pick the LAST significant snap
                    # This handles the "Outside Front Done" wobble vs actual drop
                    last_snap_idx = snap_indices[-1]
                    t_down = drop_times[last_snap_idx]
                else:
                    t_down = t_end
            else:
                t_down = t_end

    # 4. FUEL
    t_f_start, t_f_end = None, None
    if t_start and t_end:
        fuel_w = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        matches = fuel_w['Probe_Match_Sm'].values
        if len(matches) > 0 and np.max(matches) > 0.55:
            active = matches > 0.55
            idx = np.where(active)[0]
            if len(idx) > fps:
                t_f_start = fuel_w.iloc[idx[0]]['Time']
                t_f_end = fuel_w.iloc[idx[-1]]['Time']
                if t_f_end > t_end - 0.5: t_f_end = t_end - 0.5

    # Fallbacks
    if t_up is None: t_up = t_start
    if t_down is None: t_down = t_end

    return (t_start, t_end), (t_up, t_down), (t_fuel_start, t_fuel_end), corner_times

# --- PASS 3: Render ---
def render_overlay(input_path, pit, tires, fuel, corners, fps, width, height, progress_callback):
    cap = cv2.VideoCapture(input_path)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
    
    t_start, t_end = pit
    t_up, t_down = tires
    t_f_start, t_f_end = fuel
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        curr = frame_idx / fps
        
        # --- TIMERS ---
        # Pit
        if t_start and curr >= t_start:
            vp = (t_end - t_start) if (t_end and curr >= t_end) else (curr - t_start)
            cp = (0,0,255) if (t_end and curr >= t_end) else (0,255,0)
        else: vp, cp = 0.0, (200,200,200)

        # Total Tires
        if t_up and curr >= t_up:
            vt = (t_down - t_up) if (t_down and curr >= t_down) else (curr - t_up)
            ct = (0,0,255) if (t_down and curr >= t_down) else (0,255,255)
        else: vt, ct = 0.0, (200,200,200)

        # Fuel
        if t_f_start and curr >= t_f_start:
            vf = (t_f_end - t_f_start) if (t_f_end and curr >= t_f_end) else (curr - t_f_start)
            cf = (0,0,255) if (t_f_end and curr >= t_f_end) else (255,165,0)
        else: vf, cf = 0.0, (200,200,200)
        
        # --- UI DRAWING ---
        # Main Box
        cv2.rectangle(frame, (width-450, 0), (width, 320), (0,0,0), -1)
        
        # Main Stats
        cv2.putText(frame, "PIT STOP", (width-430, 40), 0, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{vp:.2f}s", (width-180, 40), 0, 1.2, cp, 3)
        
        cv2.putText(frame, "FUELING", (width-430, 90), 0, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{vf:.2f}s", (width-180, 90), 0, 1.2, cf, 3)
        
        cv2.putText(frame, "TIRES (Total)", (width-430, 140), 0, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{vt:.2f}s", (width-180, 140), 0, 1.2, ct, 3)

        # Individual Tire Stats (Corner Matrix)
        # Draw 2x2 grid visual
        start_y = 180
        gap_y = 30
        # Order: IR, OF, OR, IF
        labels = ["Inside Rear", "Outside Front", "Outside Rear", "Inside Front"]
        for i, label in enumerate(labels):
            val = corners.get(label, 0.0)
            # If total tire time is done, show final val, else show running or final
            # Simplified: Just show the calculated duration constant
            txt_col = (200,200,200)
            if t_up and curr > t_up: txt_col = (255,255,255)
            
            y_pos = start_y + (i*gap_y)
            cv2.putText(frame, label, (width-430, y_pos), 0, 0.6, txt_col, 1)
            cv2.putText(frame, f"{val:.1f}s", (width-100, y_pos), 0, 0.6, txt_col, 2)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0: progress_callback(frame_idx / total_frames)
            
    cap.release()
    out.release()
    return temp_output.name

# --- Main ---
def main():
    st.title("🏁 Pit Stop Analyzer V31")
    st.markdown("### 4-Corner Tire Analysis")
    st.info("Detects individual tire activity zones and 'Total Jacks Time' based on the LAST drop event.")

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
            st.write("Step 1: Extraction (4-Corner Flow)...")
            df, fps, w, h = extract_telemetry(tfile.name, bar.progress)
            
            st.write("Step 2: Analysis (Corner Mapping & Last Drop)...")
            pit_t, tire_t, fuel_t, corners = analyze_states_v31(df, fps)
            
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
        c2.metric("Fueling Time", f"{fuel_t[1] - fuel_t[0]:.2f}s" if fuel_t[0] else "N/A")
        c3.metric("Tire Change Time", f"{tire_t[1] - tire_t[0]:.2f}s")
        
        st.write("### 🛞 Tire Stats")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Inside Rear", f"{corners.get('Inside Rear', 0):.1f}s")
        t2.metric("Outside Front", f"{corners.get('Outside Front', 0):.1f}s")
        t3.metric("Outside Rear", f"{corners.get('Outside Rear', 0):.1f}s")
        t4.metric("Inside Front", f"{corners.get('Inside Front', 0):.1f}s")
        
        st.subheader("Video Result")
        c1, c2 = st.columns([3,1])
        with c1:
            if os.path.exists(vid_path): st.video(vid_path)
        with c2:
            if os.path.exists(vid_path):
                with open(vid_path, 'rb') as f:
                    st.download_button("Download MP4", f, file_name="pitstop_v31.mp4")

if __name__ == "__main__":
    main()
