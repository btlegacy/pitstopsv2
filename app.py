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
    
    ref_probe = load_ref_image("probein")

    telemetry_data = []
    
    # Global ROI
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
        
        mag = np.sqrt(fx**2 + flow[..., 1]**2)
        active = mag > 1.0 
        flow_x = np.median(fx[active]) if np.any(active) else 0.0
        
        # Zoom
        f_left = fx[:, :mid_x]
        f_right = fx[:, mid_x:]
        val_l = np.median(f_left[np.abs(f_left)>0.5]) if np.any(np.abs(f_left)>0.5) else 0.0
        val_r = np.median(f_right[np.abs(f_right)>0.5]) if np.any(np.abs(f_right)>0.5) else 0.0
        zoom_score = val_r - val_l
        
        # 2. 4-CORNER ACTIVITY
        h_roi, w_roi = curr_roi.shape
        mid_h, mid_w = h_roi // 2, w_roi // 2
        q_mag = mag 
        
        act_tl = np.mean(q_mag[:mid_h, :mid_w]) # Top Left
        act_tr = np.mean(q_mag[:mid_h, mid_w:]) # Top Right
        act_bl = np.mean(q_mag[mid_h:, :mid_w]) # Bottom Left
        act_br = np.mean(q_mag[mid_h:, mid_w:]) # Bottom Right
        
        # 3. FUEL
        probe_score = 0.0
        if ref_probe is not None:
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

# --- Helper: Sequential Crossover ---
def find_sequence_times(time_arr, sig1, sig2, start_gate):
    """
    Finds the handover point between Tire 1 and Tire 2.
    Start Gate: Earliest allowed time (e.g. Jacks Up).
    """
    # Only look at data after the start gate
    mask = time_arr >= start_gate
    if not np.any(mask): return start_gate, start_gate, start_gate

    t_valid = time_arr[mask]
    s1_valid = sig1[mask]
    s2_valid = sig2[mask]
    
    # 1. Find Start of T1
    # When does signal 1 first exceed mean+std?
    thresh1 = np.mean(s1_valid) + np.std(s1_valid)
    active1 = np.where(s1_valid > thresh1)[0]
    
    if len(active1) > 0:
        t1_start = t_valid[active1[0]]
    else:
        t1_start = start_gate

    # 2. Find Handover (T1 End / T2 Start)
    # We look for the valley between the two major peaks of activity
    # Or simply the intersection where S2 becomes dominant over S1
    
    # Peak detection
    p1 = np.argmax(s1_valid)
    p2 = np.argmax(s2_valid)
    
    if p2 > p1:
        # Normal sequence: Find min sum between peaks
        slice_between = s1_valid[p1:p2] + s2_valid[p1:p2]
        if len(slice_between) > 0:
            handover_idx = p1 + np.argmin(slice_between)
            t_handover = t_valid[handover_idx]
        else:
            t_handover = t_valid[p2]
    else:
        # Overlap or messy data, split halfway through the valid window
        t_handover = t_valid[len(t_valid)//2]

    # 3. Find End of T2
    # When does S2 drop back to baseline after handover?
    # Filter S2 data after handover
    mask_t2 = t_valid > t_handover
    if np.any(mask_t2):
        s2_post = s2_valid[t_valid > t_handover]
        t2_post = t_valid[t_valid > t_handover]
        thresh2 = np.mean(s2_post) + (np.std(s2_post) * 0.5)
        
        # Find last index above threshold
        active2 = np.where(s2_post > thresh2)[0]
        if len(active2) > 0:
            t2_end = t2_post[active2[-1]]
        else:
            t2_end = t_handover + 2.0 # Default guess
    else:
        t2_end = t_handover

    return t1_start, t_handover, t2_end

# --- PASS 2: Multi-Tire Analysis ---
def analyze_states_v34(df, fps):
    window = 15
    for col in ['Flow_X', 'Zoom_Score', 'Probe_Match', 'Act_TL', 'Act_TR', 'Act_BL', 'Act_BR']:
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

    # 2. JACKS (Snap Detection V23/V27)
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

    # 3. CORNER TIMING (Sequential)
    corner_stats = {}
    
    if t_start and t_end and t_up:
        # Determine Columns based on direction
        # Top=Outside, Bottom=Inside
        if arrival_dir > 0: # L->R
            col_or = 'Act_TL_Sm'
            col_of = 'Act_TR_Sm'
            col_ir = 'Act_BL_Sm'
            col_if = 'Act_BR_Sm'
        else: # R->L
            col_or = 'Act_TR_Sm'
            col_of = 'Act_TL_Sm'
            col_ir = 'Act_BR_Sm'
            col_if = 'Act_BL_Sm'

        # Extract window for analysis
        mask_active = (df['Time'] >= t_up) & (df['Time'] <= t_down)
        df_active = df[mask_active]
        times_active = df_active['Time'].values
        
        # A. FRONT PAIR: Outside Front -> Inside Front
        sig_of = df_active[col_of].values
        sig_if = df_active[col_if].values
        
        t_of_start, t_handover_front, t_if_end = find_sequence_times(times_active, sig_of, sig_if, t_up)
        
        corner_stats['Outside Front'] = (t_of_start, t_handover_front)
        corner_stats['Inside Front'] = (t_handover_front, t_if_end)
        
        # B. REAR PAIR: Inside Rear -> Outside Rear
        sig_ir = df_active[col_ir].values
        sig_or = df_active[col_or].values
        
        t_ir_start, t_handover_rear, t_or_end = find_sequence_times(times_active, sig_ir, sig_or, t_up)
        
        corner_stats['Inside Rear'] = (t_ir_start, t_handover_rear)
        corner_stats['Outside Rear'] = (t_handover_rear, t_or_end)

    # 4. FUEL
    t_fuel_start, t_fuel_end = None, None
    if t_start and t_end:
        fuel_w = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        matches = fuel_w['Probe_Match_Sm'].values
        if len(matches) > 0 and np.max(matches) > 0.55:
            active = matches > 0.55
            idx = np.where(active)[0]
            if len(idx) > fps:
                t_fuel_start = fuel_w.iloc[idx[0]]['Time']
                t_fuel_end = fuel_w.iloc[idx[-1]]['Time']
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

        # --- CONTINUOUS CORNER TIMERS ---
        start_y = 180
        gap_y = 30
        
        for i, label in enumerate(labels):
            # Get pre-calc start/end for this corner
            c_start, c_end = corner_data.get(label, (0.0, 0.0))
            
            c_val = 0.0
            if c_start > 0 and curr >= c_start:
                if curr >= c_end:
                    c_val = c_end - c_start
                else:
                    c_val = curr - c_start
            
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
    st.title("🏁 Pit Stop Analyzer V34")
    st.markdown("### Continuous Sequential Timing")
    st.info("Individual tire timers count continuously. Order: Outside Front -> Inside Front, Inside Rear -> Outside Rear.")

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
            
            st.write("Step 2: Analysis (Continuous Sequence)...")
            pit_t, tire_t, fuel_t, corners = analyze_states_v34(df, fps)
            
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
                    st.download_button("Download MP4", f, file_name="pitstop_v34.mp4")

if __name__ == "__main__":
    main()
