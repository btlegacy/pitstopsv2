import streamlit as st
import cv2
import numpy as np
import tempfile
import os
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

# --- Helper: Feature Matching ---
def get_ref_features(ref_image):
    if ref_image is None: return None, None
    orb = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(ref_image, None)
    return kp, des

def match_features(frame_gray, ref_des, orb_detector):
    if ref_des is None: return False, []
    kp_frame, des_frame = orb_detector.detectAndCompute(frame_gray, None)
    if des_frame is None: return False, []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(ref_des, des_frame)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50]
    src_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return len(good_matches) > 15, src_pts

# --- PASS 1: Data Extraction ---
def extract_telemetry(video_path, ref_img_path, progress_callback):
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load Ref
    ref_des, orb = None, None
    if ref_img_path and os.path.exists(ref_img_path):
        ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
        if ref_img is not None:
            orb = cv2.ORB_create(nfeatures=1000)
            _, ref_des = get_ref_features(ref_img)
    
    telemetry_data = []
    prev_gray = None
    
    # --- ZONE CONFIGURATION ---
    # EXPANDED AREA: Only ignore the very bottom 10% of the screen
    # This ensures we catch the car even if it occupies the lower half.
    detection_limit_y = int(height * 0.90) 
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # 1. Global Motion (Fallback Metric)
        motion_score = 0.0
        if prev_gray is not None:
            delta = cv2.absdiff(prev_gray, gray_blur)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_score = np.sum(thresh) / 255.0
        prev_gray = gray_blur
        
        # 2. AI Detection (Primary Metric)
        # conf=0.15 allows low-confidence detections (good for fisheye/distortion)
        results = model.track(frame, persist=True, classes=[2], verbose=False, conf=0.15)
        car_detected = False
        cx, cy = np.nan, np.nan
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            # Filter: Ignore only if center is in the bottom 10%
            valid_indices = [i for i, box in enumerate(boxes) if box[1] < detection_limit_y]
            if valid_indices:
                valid_boxes = boxes[valid_indices]
                areas = valid_boxes[:, 2] * valid_boxes[:, 3]
                largest_idx = np.argmax(areas)
                bx, by, bw, bh = valid_boxes[largest_idx]
                cx, cy = bx, by
                car_detected = True

        # Ref Match Fallback
        if not car_detected and ref_des is not None:
            match_found, points = match_features(gray, ref_des, orb)
            if match_found:
                mean_y = np.mean(points[:, 0, 1])
                if mean_y < detection_limit_y:
                    cx = np.mean(points[:, 0, 0])
                    cy = mean_y
                    car_detected = True
        
        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Car_X": cx,
            "Car_Y": cy,
            "Motion_Intensity": motion_score,
            "Detected": car_detected
        })

        frame_idx += 1
        if frame_idx % 20 == 0:
            progress_callback(frame_idx / total_frames)

    cap.release()
    return pd.DataFrame(telemetry_data), fps, width, height

# --- Timing Analysis ---
def analyze_timings_robust(df, fps):
    """
    Dual-Layer Logic:
    1. Try X-Velocity (Interpolated).
    2. If that fails, use Motion Valley.
    """
    
    # --- PRE-PROCESSING: Interpolation ---
    # Interpolate up to 1.5 seconds of missing data (crew covering car)
    df['Car_X'] = df['Car_X'].interpolate(method='linear', limit=int(fps*1.5))
    df['Car_Y'] = df['Car_Y'].interpolate(method='linear', limit=int(fps*1.5))
    
    # Calculate Velocity
    window = 15
    if len(df) > window:
        df['Car_X'] = df['Car_X'].fillna(method='bfill').fillna(method='ffill')
        df['Car_Y'] = df['Car_Y'].fillna(method='bfill').fillna(method='ffill')
        
        # 1st Derivative (Velocity)
        df['Vel_X'] = savgol_filter(df['Car_X'], window, 3, deriv=1)
        df['Vel_Y'] = savgol_filter(df['Car_Y'], window, 3, deriv=1)
        
        df['Vel_X'] = df['Vel_X'].abs()
        df['Vel_Y'] = df['Vel_Y'].abs()
    else:
        df['Vel_X'] = 0
        df['Vel_Y'] = 0

    # --- LAYER 1: Horizontal Stop Detection ---
    t_start, t_end = None, None
    
    # Stop Threshold
    stop_thresh = 1.5 
    is_stopped = df['Vel_X'] < stop_thresh
    
    df['block'] = (is_stopped != is_stopped.shift()).cumsum()
    blocks = df[is_stopped].groupby('block')
    
    if len(blocks) > 0:
        largest_block = blocks.size().idxmax()
        block_data = df[df['block'] == largest_block]
        duration = block_data.iloc[-1]['Time'] - block_data.iloc[0]['Time']
        
        # Valid pit stop > 2.0 seconds
        if duration > 2.0:
            t_start = block_data.iloc[0]['Time']
            t_end = block_data.iloc[-1]['Time']

    # --- LAYER 2: Fallback ---
    used_fallback = False
    if t_start is None:
        used_fallback = True
        df['Motion_Smooth'] = savgol_filter(df['Motion_Intensity'], 25, 3)
        max_mot = df['Motion_Smooth'].max()
        high_thresh = max_mot * 0.25
        low_thresh = max_mot * 0.10
        
        peaks, _ = find_peaks(df['Motion_Smooth'], height=high_thresh, distance=fps*5)
        
        if len(peaks) >= 2:
            start_idx = peaks[0]
            while start_idx < len(df)-1 and df['Motion_Smooth'].iloc[start_idx] > low_thresh:
                start_idx += 1
            
            end_idx = peaks[-1]
            while end_idx > 0 and df['Motion_Smooth'].iloc[end_idx] > low_thresh:
                end_idx -= 1
                
            t_start = df.iloc[start_idx]['Time']
            t_end = df.iloc[end_idx]['Time']

    # --- DETECT JACKS ---
    t_up, t_down = None, None
    
    if t_start and t_end:
        window_df = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        
        if not window_df.empty:
            y_signal = window_df['Vel_Y'].values
            if used_fallback:
                y_signal = window_df['Motion_Intensity'].values
            
            prominence = np.max(y_signal) * 0.2
            peaks, _ = find_peaks(y_signal, prominence=prominence, distance=fps*0.5)
            
            peak_times = window_df.iloc[peaks]['Time'].values
            if len(peak_times) > 0:
                t_up = peak_times[0]
                t_down = peak_times[-1] if len(peak_times) > 1 else t_end

    return t_start, t_end, (t_up, t_down), used_fallback

# --- PASS 2: Video Rendering ---
def render_final_video(input_path, timings, fps, width, height, progress_callback):
    t_start, t_end, t_up, t_down = timings
    cap = cv2.VideoCapture(input_path)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
    
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    detection_limit_y = int(height * 0.90) # Update visual line to 90%
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        current_time = frame_idx / fps
        
        # Pit Timer
        if t_start and current_time >= t_start:
            if t_end and current_time >= t_end:
                val_pit = t_end - t_start
                color_pit = (0, 0, 255)
            else:
                val_pit = current_time - t_start
                color_pit = (0, 255, 0)
        else:
            val_pit = 0.0
            color_pit = (200, 200, 200) 

        # Tire Timer
        if t_up and current_time >= t_up:
            if t_down and current_time >= t_down:
                val_tire = t_down - t_up
                color_tire = (0, 0, 255)
            else:
                val_tire = current_time - t_up
                color_tire = (0, 255, 255)
        else:
            val_tire = 0.0
            color_tire = (200, 200, 200)

        # Overlay
        box_w, box_h = 400, 140
        cv2.rectangle(frame, (width - box_w, 0), (width, box_h), (0, 0, 0), -1)
        
        cv2.putText(frame, "PIT STOP (H-Stop)", (width - 380, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{val_pit:.2f}s", (width - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_pit, 3)

        cv2.putText(frame, "TIRES (V-Jolt)", (width - 380, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{val_tire:.2f}s", (width - 150, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_tire, 3)
        
        # Draw New Limit Line
        cv2.line(frame, (0, detection_limit_y), (width, detection_limit_y), (0,0,100), 1)
        cv2.putText(frame, "Limit (10%)", (10, detection_limit_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,100), 1)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            progress_callback(frame_idx / total_frames)
            
    cap.release()
    out.release()
    return temp_output.name

# --- Main UI ---
def main():
    st.title("🏁 Pit Stop AI Analyzer V11")
    st.markdown("### Robust Detection (Wide Area)")
    st.info("Expanded monitoring to 90% of the screen height to ensure car is detected.")

    default_ref_path = os.path.join(BASE_DIR, "refs", "car.png")
    ref_path = default_ref_path if os.path.exists(default_ref_path) else None

    if 'analysis_done' not in st.session_state:
        st.session_state.update({'analysis_done': False, 'df': None, 'video_path': None, 'timings': None})

    uploaded_file = st.file_uploader("Upload Overhead Video", type=["mp4", "mov", "avi"])

    if uploaded_file and st.button("Start Analysis", type="primary"):
        st.session_state['analysis_done'] = False
        
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile_in.write(uploaded_file.read())
        tfile_in.flush()

        try:
            progress_bar = st.progress(0)
            status = st.empty()
            
            status.write("Pass 1/2: Extracting Telemetry (Wide Scan)...")
            df, fps, w, h = extract_telemetry(tfile_in.name, ref_path, progress_bar.progress)
            
            status.write("Calculating Events...")
            t_start, t_end, (t_up, t_down), fallback = analyze_timings_robust(df, fps)
            timings = (t_start, t_end, t_up, t_down)
            
            if t_start is None:
                st.error("Critical Failure: Could not identify any stop event.")
            else:
                status.write("Pass 2/2: Rendering Overlay Video...")
                vid_path = render_final_video(tfile_in.name, timings, fps, w, h, progress_bar.progress)
                
                st.session_state.update({
                    'df': df,
                    'video_path': vid_path,
                    'timings': timings,
                    'fallback': fallback,
                    'analysis_done': True
                })
            
            progress_bar.empty()
            status.empty()
            
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists(tfile_in.name): os.remove(tfile_in.name)

    if st.session_state['analysis_done']:
        df = st.session_state['df']
        vid_path = st.session_state['video_path']
        t_start, t_end, t_up, t_down = st.session_state['timings']
        fallback = st.session_state.get('fallback', False)

        st.divider()
        
        if fallback:
            st.warning("⚠️ Note: Used Motion Intensity backup. (Check video to confirm).")
        
        m1, m2, m3, m4 = st.columns(4)
        if t_start and t_end:
            m1.metric("Pit Stop Duration", f"{(t_end - t_start):.2f}s")
            m3.metric("Stop Time", f"{t_start:.2f}s")
            m4.metric("Go Time", f"{t_end:.2f}s")
        
        if t_up and t_down:
            m2.metric("Tire Change", f"{(t_down - t_up):.2f}s")
        else:
            m2.metric("Tire Change", "--")

        st.subheader("📊 Telemetry")
        
        base = alt.Chart(df).encode(x='Time')
        
        # Velocity X
        l1 = base.mark_line(color='cyan').encode(y=alt.Y('Vel_X', title='Horizontal Vel'))
        
        # Velocity Y / Motion
        if fallback:
            l2 = base.mark_area(color='gray', opacity=0.3).encode(y=alt.Y('Motion_Intensity', title='Global Motion (Fallback)'))
        else:
            l2 = base.mark_line(color='orange').encode(y=alt.Y('Vel_Y', title='Vertical Vel'))
            
        if t_start and t_end:
            rect = alt.Chart(pd.DataFrame({'s': [t_start], 'e': [t_end]})).mark_rect(color='green', opacity=0.1).encode(x='s', x2='e')
            st.altair_chart((l1 + l2 + rect).interactive(), use_container_width=True)
        else:
            st.altair_chart((l1 + l2).interactive(), use_container_width=True)
        
        st.subheader("👁️ Final Video")
        c1, c2 = st.columns([3,1])
        with c1:
            if os.path.exists(vid_path): st.video(vid_path)
        with c2:
            if os.path.exists(vid_path):
                with open(vid_path, 'rb') as f:
                    st.download_button("Download Video", f, file_name="analyzed_pitstop.mp4")

if __name__ == "__main__":
    main()
