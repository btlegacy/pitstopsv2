import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import glob
import pandas as pd
import altair as alt
from scipy.signal import savgol_filter, find_peaks

# --- Configuration ---
st.set_page_config(page_title="Pit Stop Analytics AI", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Helper: Reference Loading ---
def load_reference_features(folder_name):
    """Loads the first valid image from a folder and computes ORB features."""
    path = os.path.join(BASE_DIR, "refs", folder_name, "*")
    files = glob.glob(path)
    
    if not files:
        # Check flattened structure
        path = os.path.join(BASE_DIR, "refs", f"{folder_name}.*")
        files = glob.glob(path)

    if not files:
        return None, None

    img_path = files[0]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None, None
    
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

# --- PASS 1: Extraction ---
def extract_telemetry(video_path, progress_callback):
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load Fuel References
    fuel_kp, fuel_des = load_reference_features("probein")
    
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    telemetry_data = []
    
    roi_x1, roi_x2 = int(width * 0.20), int(width * 0.80)
    roi_y1, roi_y2 = int(height * 0.20), int(height * 0.80)
    mid_x = int((roi_x2 - roi_x1) / 2)
    
    ret, prev_frame = cap.read()
    if not ret: return pd.DataFrame(), fps, width, height
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_roi = prev_gray[roi_y1:roi_y2, roi_x1:roi_x2]
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Optical Flow
        flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        fx = flow[..., 0]
        fy = flow[..., 1]
        
        mag = np.sqrt(fx**2 + fy**2)
        active_mask = mag > 1.0 
        if np.any(active_mask):
            med_x = np.median(fx[active_mask])
            med_y = np.median(fy[active_mask])
        else:
            med_x = 0.0
            med_y = 0.0
            
        # Zoom
        flow_left = fx[:, :mid_x]
        flow_right = fx[:, mid_x:]
        mask_l = np.abs(flow_left) > 0.5
        mask_r = np.abs(flow_right) > 0.5
        val_l = np.median(flow_left[mask_l]) if np.any(mask_l) else 0.0
        val_r = np.median(flow_right[mask_r]) if np.any(mask_r) else 0.0
        zoom_score = val_r - val_l
        
        # Fuel Matches
        fuel_matches = 0
        if fuel_des is not None:
            kp_frame, des_frame = orb.detectAndCompute(curr_roi, None)
            if des_frame is not None:
                matches = bf.match(fuel_des, des_frame)
                good_matches = [m for m in matches if m.distance < 60]
                fuel_matches = len(good_matches)

        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Flow_X": med_x,
            "Flow_Y": med_y,
            "Zoom_Score": zoom_score,
            "Fuel_Matches": fuel_matches
        })
        
        prev_roi = curr_roi
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            progress_callback(frame_idx / total_frames)

    cap.release()
    return pd.DataFrame(telemetry_data), fps, width, height

# --- PASS 2: Analysis ---
def analyze_states(df, fps):
    window_slow = 15
    window_fast = 5
    
    if len(df) > window_slow:
        df['X_Smooth'] = savgol_filter(df['Flow_X'], window_slow, 3)
        df['Y_Smooth'] = savgol_filter(df['Flow_Y'], window_fast, 3) 
        df['Zoom_Smooth'] = savgol_filter(df['Zoom_Score'], window_fast, 3)
        df['Fuel_Smooth'] = savgol_filter(df['Fuel_Matches'], window_slow, 3)
    else:
        df['X_Smooth'] = df['Flow_X']
        df['Y_Smooth'] = df['Flow_Y']
        df['Zoom_Smooth'] = df['Zoom_Score']
        df['Fuel_Smooth'] = df['Fuel_Matches']

    df['Zoom_Velocity'] = np.gradient(df['Zoom_Smooth'])

    # 1. PIT STOP
    x_mag = df['X_Smooth'].abs()
    MOVE_THRESH = x_mag.max() * 0.3 
    STOP_THRESH = x_mag.max() * 0.05 
    
    peaks, _ = find_peaks(x_mag, height=MOVE_THRESH, distance=fps*5)
    t_start, t_end = None, None
    
    if len(peaks) >= 2:
        arrival_idx = peaks[0]
        depart_idx = peaks[-1]
        
        start_idx = arrival_idx
        for i in range(arrival_idx, depart_idx):
            if x_mag.iloc[i] < STOP_THRESH:
                start_idx = i
                break
        
        end_idx = depart_idx
        for i in range(depart_idx, start_idx, -1):
            if x_mag.iloc[i] < STOP_THRESH:
                end_idx = i
                break
        
        t_start = df.iloc[start_idx]['Time']
        t_end = df.iloc[end_idx]['Time']
    
    # 2. JACKS
    t_up, t_down = None, None
    
    if t_start and t_end:
        creep_idx = end_idx
        CREEP_THRESH = STOP_THRESH * 0.5
        for i in range(end_idx, start_idx, -1):
            if x_mag.iloc[i] < CREEP_THRESH:
                creep_idx = i
                break
        t_creep_limit = df.iloc[creep_idx]['Time']
        
        stop_window = df[(df['Time'] >= t_start) & (df['Time'] <= t_creep_limit)]
        
        if not stop_window.empty:
            z_pos = stop_window['Zoom_Smooth'].values
            times = stop_window['Time'].values
            
            peaks_up, _ = find_peaks(z_pos, height=0.2, distance=fps)
            if len(peaks_up) > 0:
                t_up = times[peaks_up[0]]
                base_zoom = np.min(z_pos)
                max_zoom = np.max(z_pos)
                lift_amplitude = max_zoom - base_zoom
            else:
                t_up = t_start
                lift_amplitude = 0.5 
                
            neg_vel = -stop_window['Zoom_Velocity'].values
            cand_drops, _ = find_peaks(neg_vel, height=0.1, distance=fps*0.5)
            valid_drops = []
            
            for p in cand_drops:
                t_p = times[p]
                if t_p <= t_up + 1.0: continue
                
                idx_before = max(0, p - int(fps*0.3))
                idx_after = min(len(z_pos)-1, p + int(fps*0.3))
                drop_dist = z_pos[idx_before] - z_pos[idx_after]
                
                if drop_dist > (lift_amplitude * 0.2):
                    valid_drops.append(t_p)
            
            if valid_drops:
                t_down = valid_drops[-1]
            else:
                t_down = t_creep_limit

    # 3. FUELING
    t_fuel_start, t_fuel_end = None, None
    
    if t_start and t_end:
        fuel_window = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        if not fuel_window.empty:
            f_sig = fuel_window['Fuel_Smooth'].values
            times = fuel_window['Time'].values
            max_matches = np.max(f_sig)
            
            if max_matches > 10:
                fuel_thresh = max_matches * 0.5
                is_fueling = f_sig > fuel_thresh
                start_indices = np.where(is_fueling)[0]
                if len(start_indices) > 0:
                    t_fuel_start = times[start_indices[0]]
                    t_fuel_end = times[start_indices[-1]]
                    if t_fuel_end > t_end - 0.5:
                        t_fuel_end = t_end - 0.5

    if t_up is None: t_up = t_start
    if t_down is None: t_down = t_end

    return (t_start, t_end), (t_up, t_down), (t_fuel_start, t_fuel_end)

# --- PASS 3: Render ---
def render_overlay(input_path, pit_times, tire_times, fuel_times, fps, width, height, progress_callback):
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
        
        # Pit Timer
        if t_start and current_time >= t_start:
            if t_end and current_time >= t_end:
                val_pit = t_end - t_start
                col_pit = (0,0,255)
            else:
                val_pit = current_time - t_start
                col_pit = (0,255,0)
        else:
            val_pit, col_pit = 0.0, (200,200,200)

        # Tire Timer
        if t_up and current_time >= t_up:
            if t_down and current_time >= t_down:
                val_tire = t_down - t_up
                col_tire = (0,0,255)
            else:
                val_tire = current_time - t_up
                col_tire = (0,255,255)
        else:
            val_tire, col_tire = 0.0, (200,200,200)
            
        # Fuel Timer
        if t_f_start and current_time >= t_f_start:
            if t_f_end and current_time >= t_f_end:
                val_fuel = t_f_end - t_f_start
                col_fuel = (0,0,255)
            else:
                val_fuel = current_time - t_f_start
                col_fuel = (255,165,0)
        else:
            val_fuel, col_fuel = 0.0, (200,200,200)
        
        # Draw UI
        cv2.rectangle(frame, (width-450, 0), (width, 240), (0,0,0), -1)
        
        cv2.putText(frame, "PIT STOP", (width-430, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{val_pit:.2f}s", (width-180, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col_pit, 3)
        
        cv2.putText(frame, "FUELING", (width-430, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{val_fuel:.2f}s", (width-180, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col_fuel, 3)

        cv2.putText(frame, "TIRES", (width-430, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{val_tire:.2f}s", (width-180, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col_tire, 3)

        cv2.putText(frame, "V25: Full Suite (Pit/Fuel/Tires)", (width-430, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            progress_callback(frame_idx / total_frames)
            
    cap.release()
    out.release()
    return temp_output.name

# --- Main ---
def main():
    st.title("🏁 Pit Stop Analyzer V25")
    st.markdown("### Full Analysis Suite")
    st.info("Tracks Pit Stop (Motion), Tires (Zoom/Drop), and Fuel (Visual Matching).")

    probe_path = os.path.join(BASE_DIR, "refs", "probein")
    if not os.path.exists(probe_path):
        st.warning("⚠️ 'refs/probein' not found. Fuel detection will fail.")

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
            st.write("Step 1: Extraction...")
            df, fps, w, h = extract_telemetry(tfile.name, bar.progress)
            
            st.write("Step 2: Logic Analysis...")
            pit_t, tire_t, fuel_t = analyze_states(df, fps)
            
            if pit_t[0] is None:
                st.error("Could not detect Stop.")
            else:
                st.write("Step 3: Rendering Video...")
                # CORRECTED CALL: Passing tuples separately
                vid_path = render_overlay(tfile.name, pit_t, tire_t, fuel_t, fps, w, h, bar.progress)
                
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
        
        if fuel_t[0] and fuel_t[1]:
            c2.metric("Fueling Time", f"{fuel_t[1] - fuel_t[0]:.2f}s")
        else:
            c2.metric("Fueling Time", "Not Detected")

        if tire_t[0] and tire_t[1]:
            c3.metric("Tire Change Time", f"{tire_t[1] - tire_t[0]:.2f}s")
        
        st.subheader("📊 Telemetry")
        base = alt.Chart(df).encode(x='Time')
        
        fuel_chart = base.mark_area(color='orange', opacity=0.3).encode(y=alt.Y('Fuel_Smooth', title='Fuel Matches'))
        zoom_chart = base.mark_line(color='magenta').encode(y=alt.Y('Zoom_Smooth', title='Zoom'))
        
        st.altair_chart((fuel_chart + zoom_chart).interactive(), use_container_width=True)
        
        st.subheader("Video Result")
        c1, c2 = st.columns([3,1])
        with c1:
            if os.path.exists(vid_path): st.video(vid_path)
        with c2:
            if os.path.exists(vid_path):
                with open(vid_path, 'rb') as f:
                    st.download_button("Download MP4", f, file_name="pitstop_full.mp4")

if __name__ == "__main__":
    main()
