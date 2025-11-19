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
    """
    Pass 1: Analyze video to get Motion and Object Detection data.
    Does NOT generate the output video yet.
    """
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
    
    # Zones
    stall_x_start = int(width * 0.20) 
    stall_x_end = int(width * 0.80)
    detection_limit_y = int(height * 0.70)
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Motion
        motion_score = 0.0
        if prev_gray is not None:
            delta = cv2.absdiff(prev_gray, gray_blur)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_score = np.sum(thresh) / 255.0
        prev_gray = gray_blur

        # AI Detection
        results = model.track(frame, persist=True, classes=[2], verbose=False, conf=0.15)
        car_detected = False
        car_center_x = -1
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            # Filter bottom 30%
            valid_indices = [i for i, box in enumerate(boxes) if box[1] < detection_limit_y]
            if valid_indices:
                valid_boxes = boxes[valid_indices]
                areas = valid_boxes[:, 2] * valid_boxes[:, 3]
                largest_idx = np.argmax(areas)
                bx, by, bw, bh = valid_boxes[largest_idx]
                car_center_x = bx
                car_detected = True

        # Ref Match
        ref_match_detected = False
        if not car_detected and ref_des is not None:
            match_found, points = match_features(gray, ref_des, orb)
            if match_found:
                mean_y = np.mean(points[:, 0, 1])
                if mean_y < detection_limit_y:
                    ref_match_detected = True
                    car_center_x = np.mean(points[:, 0, 0])
        
        # Stall Logic
        in_stall = False
        if car_detected or ref_match_detected:
            if stall_x_start < car_center_x < stall_x_end:
                in_stall = True

        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Motion_Intensity": motion_score,
            "In_Stall": in_stall
        })

        frame_idx += 1
        if frame_idx % 20 == 0:
            progress_callback(frame_idx / total_frames)

    cap.release()
    return pd.DataFrame(telemetry_data), fps, width, height

# --- Timing Analysis ---
def analyze_timings_valley(df, fps):
    """Calculate Start/End times based on motion valley."""
    df['Motion_Smooth'] = savgol_filter(df['Motion_Intensity'], 15, 3)
    max_motion = df['Motion_Smooth'].max()
    threshold_high = max_motion * 0.25
    threshold_low = max_motion * 0.10
    
    peaks, _ = find_peaks(df['Motion_Smooth'], height=threshold_high, distance=fps*5)
    t_start, t_end = None, None
    
    if len(peaks) >= 2:
        arrival_idx = peaks[0]
        departure_idx = peaks[-1]
        
        start_idx = arrival_idx
        for i in range(arrival_idx, len(df)):
            if df['Motion_Smooth'].iloc[i] < threshold_low:
                start_idx = i
                break
        
        end_idx = departure_idx
        for i in range(departure_idx, start_idx, -1):
            if df['Motion_Smooth'].iloc[i] < threshold_low:
                end_idx = i
                break
        
        t_start = df.iloc[start_idx]['Time']
        t_end = df.iloc[end_idx]['Time']
    else:
        # Fallback
        mid_df = df.iloc[int(len(df)*0.1):int(len(df)*0.9)]
        mask = mid_df['Motion_Smooth'] < threshold_low
        mid_df = mid_df.copy()
        mid_df['block'] = (mask != mask.shift()).cumsum()
        blocks = mid_df[mask].groupby('block')
        if len(blocks) > 0:
            largest = blocks.size().idxmax()
            segment = mid_df[mid_df['block'] == largest]
            t_start = segment['Time'].min()
            t_end = segment['Time'].max()

    # Tire Change
    t_up, t_down = t_start, t_end
    if t_start and t_end:
        stop_window = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        if not stop_window.empty:
            curve = stop_window['Motion_Intensity'].values
            j_peaks, _ = find_peaks(curve, prominence=np.max(curve)*0.15)
            if len(j_peaks) > 0:
                times = stop_window.iloc[j_peaks]['Time'].values
                t_up = times[0]
                t_down = times[-1]

    return t_start, t_end, (t_up, t_down)

# --- PASS 2: Video Rendering ---
def render_final_video(input_path, timings, fps, width, height, progress_callback):
    """
    Pass 2: Re-read video and overlay the calculated timers.
    """
    t_start, t_end, t_up, t_down = timings
    
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
        
        # --- CALCULATE TIMERS ---
        
        # 1. Pit Stop Timer
        if t_start and current_time >= t_start:
            if t_end and current_time >= t_end:
                val_pit = t_end - t_start
                color_pit = (0, 0, 255) # Red (Stopped)
            else:
                val_pit = current_time - t_start
                color_pit = (0, 255, 0) # Green (Running)
        else:
            val_pit = 0.0
            color_pit = (200, 200, 200) # Gray (Waiting)

        # 2. Tire Change Timer
        if t_up and current_time >= t_up:
            if t_down and current_time >= t_down:
                val_tire = t_down - t_up
                color_tire = (0, 0, 255)
            else:
                val_tire = current_time - t_up
                color_tire = (0, 255, 255) # Yellow (Running)
        else:
            val_tire = 0.0
            color_tire = (200, 200, 200)

        # --- DRAW OVERLAY ---
        
        # Background Box (Top Right)
        box_w, box_h = 350, 130
        cv2.rectangle(frame, (width - box_w, 0), (width, box_h), (0, 0, 0), -1)
        
        # Draw Pit Stop
        cv2.putText(frame, "PIT STOP", (width - 330, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"{val_pit:.2f}s", (width - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_pit, 3)

        # Draw Tires
        cv2.putText(frame, "TIRES", (width - 330, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"{val_tire:.2f}s", (width - 160, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_tire, 3)

        # Exclusion Line (Visual Reference)
        detection_limit_y = int(height * 0.70)
        cv2.line(frame, (0, detection_limit_y), (width, detection_limit_y), (0, 0, 100), 1)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            progress_callback(frame_idx / total_frames)
            
    cap.release()
    out.release()
    return temp_output.name

# --- Main UI ---
def main():
    st.title("🏁 Pit Stop AI Analyzer V7")
    st.markdown("### Automated Timing & Overlay")
    st.info("Two-pass analysis: 1. Detect Motion/Events -> 2. Generate Video with synchronized timers.")

    # Ref Image
    default_ref_path = os.path.join(BASE_DIR, "refs", "car.png")
    ref_path = default_ref_path if os.path.exists(default_ref_path) else None
    if not ref_path: st.warning("Reference image not found. Using AI only.")

    # Session
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
            
            # PASS 1
            status.write("Pass 1/2: Analyzing Motion & Detections...")
            df, fps, w, h = extract_telemetry(tfile_in.name, ref_path, progress_bar.progress)
            
            # CALCULATION
            status.write("Calculating Timings...")
            t_start, t_end, (t_up, t_down) = analyze_timings_valley(df, fps)
            timings = (t_start, t_end, t_up, t_down)
            
            if t_start is None:
                st.error("Could not identify pit stop window.")
            else:
                # PASS 2
                status.write("Pass 2/2: Rendering Overlay Video...")
                vid_path = render_final_video(tfile_in.name, timings, fps, w, h, progress_bar.progress)
                
                st.session_state.update({
                    'df': df,
                    'video_path': vid_path,
                    'timings': timings,
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

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pit Stop Duration", f"{(t_end - t_start):.2f}s")
        m2.metric("Tire Change", f"{(t_down - t_up):.2f}s")
        m3.metric("Stop Time", f"{t_start:.2f}s")
        m4.metric("Go Time", f"{t_end:.2f}s")

        st.subheader("📊 Motion Telemetry")
        base = alt.Chart(df).encode(x='Time')
        area = base.mark_area(color='gray', opacity=0.3).encode(y=alt.Y('Motion_Intensity'))
        line = base.mark_line(color='blue', opacity=0.5).encode(y='Motion_Smooth')
        stop_rect = alt.Chart(pd.DataFrame({'start': [t_start], 'end': [t_end]})).mark_rect(color='green', opacity=0.1).encode(x='start', x2='end')
        rules = pd.DataFrame([{"Time": t_start, "Event": "Start", "Color": "green"}, {"Time": t_end, "Event": "End", "Color": "red"}])
        rule_chart = alt.Chart(rules).mark_rule(strokeWidth=3).encode(x='Time', color=alt.Color('Color', scale=None))
        st.altair_chart((area + line + stop_rect + rule_chart).interactive(), use_container_width=True)
        
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
