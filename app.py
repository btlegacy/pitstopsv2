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
    Pass 1: Extract X and Y coordinates of the car specifically.
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
    
    # Tracking state
    prev_x, prev_y = None, None
    
    # Filter settings: Ignore detection in bottom 30% (Pit Wall/Crew area)
    detection_limit_y = int(height * 0.70) 
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # AI Detection
        # Lower confidence to catch car through fisheye distortion
        results = model.track(frame, persist=True, classes=[2], verbose=False, conf=0.15)
        car_detected = False
        cx, cy = np.nan, np.nan
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            # Filter out boxes where center Y is too low
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
        
        # Calculate Instantaneous Velocity (Pixels/Frame)
        vel_x = 0.0
        vel_y = 0.0
        
        if car_detected:
            if prev_x is not None:
                vel_x = abs(cx - prev_x)
                vel_y = abs(cy - prev_y)
            prev_x, prev_y = cx, cy
        else:
            # If occluded, assume 0 velocity (Stopped)
            # We do not update prev_x/prev_y to avoid jump spikes on re-acquisition
            vel_x = 0.0
            vel_y = 0.0

        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Vel_X": vel_x,
            "Vel_Y": vel_y,
            "Car_X": cx,
            "Car_Y": cy
        })

        frame_idx += 1
        if frame_idx % 20 == 0:
            progress_callback(frame_idx / total_frames)

    cap.release()
    return pd.DataFrame(telemetry_data), fps, width, height

# --- Timing Analysis ---
def analyze_timings_xy(df, fps):
    """
    Determines events based on X and Y velocity separation.
    Includes 'Gap Filling' to ignore bounding box jitter.
    """
    # 1. Smooth Velocities
    df['Vel_X'] = df['Vel_X'].fillna(0)
    df['Vel_Y'] = df['Vel_Y'].fillna(0)
    
    # Increased window size for smoother data
    window = 25 
    if len(df) > window:
        df['Vel_X_Smooth'] = savgol_filter(df['Vel_X'], window, 3)
        df['Vel_Y_Smooth'] = savgol_filter(df['Vel_Y'], window, 3)
    else:
        df['Vel_X_Smooth'] = df['Vel_X']
        df['Vel_Y_Smooth'] = df['Vel_Y']
    
    # 2. Find Pit Stop (Horizontal Stop)
    # Increased threshold to 3.0px to ignore jitter
    move_thresh_x = 3.0 
    
    # Boolean mask: Is Stopped?
    is_stopped_x = df['Vel_X_Smooth'] < move_thresh_x
    
    # --- GAP FILLING ---
    # If we have [Stop, Stop, Move, Stop, Stop], treat 'Move' as Stop if it's short
    # We allow a gap of up to 0.5 seconds (fps/2) of "noise"
    gap_limit = int(fps / 2)
    
    # Identify gaps
    df['stopped_int'] = is_stopped_x.astype(int)
    # Find sequences of 0s (moving) between 1s (stopped)
    # This is a bit complex in pandas, simple iteration is safer for logic:
    
    clean_stopped = df['stopped_int'].values.copy()
    last_stop_idx = -1
    
    for i in range(len(clean_stopped)):
        if clean_stopped[i] == 1:
            # We are stopped
            if last_stop_idx != -1:
                # Check gap size
                gap = i - last_stop_idx - 1
                if 0 < gap < gap_limit:
                    # Fill gap
                    clean_stopped[last_stop_idx+1 : i] = 1
            last_stop_idx = i
            
    df['Is_Stopped_Clean'] = clean_stopped.astype(bool)
    
    # Find blocks
    df['block'] = (df['Is_Stopped_Clean'] != df['Is_Stopped_Clean'].shift()).cumsum()
    blocks = df[df['Is_Stopped_Clean']].groupby('block')
    
    t_start, t_end = None, None
    
    if len(blocks) > 0:
        # Find longest stopped block
        largest_block_idx = blocks.size().idxmax()
        block_data = df[df['block'] == largest_block_idx]
        
        # Must be at least 2 seconds
        duration = block_data.iloc[-1]['Time'] - block_data.iloc[0]['Time']
        if duration > 2.0:
            t_start = block_data.iloc[0]['Time']
            t_end = block_data.iloc[-1]['Time']

    # 3. Find Jacks (Vertical Movement)
    t_up, t_down = None, None
    
    if t_start and t_end:
        stop_window = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)].copy()
        
        if not stop_window.empty:
            y_curve = stop_window['Vel_Y_Smooth'].values
            
            # Peak detection
            peaks, _ = find_peaks(y_curve, height=1.0, distance=fps*1.0)
            peak_times = stop_window.iloc[peaks]['Time'].values
            
            if len(peak_times) >= 1:
                t_up = peak_times[0] 
                if len(peak_times) >= 2:
                    t_down = peak_times[-1]
                else:
                    # If only one spike, assume it was the UP, and DOWN happens at departure
                    t_down = t_end
            else:
                # Fallback: If no V-spikes found, maybe Y movement was too subtle
                t_up = t_start
                t_down = t_end

    return t_start, t_end, (t_up, t_down)

# --- PASS 2: Video Rendering ---
def render_final_video(input_path, timings, fps, width, height, progress_callback):
    t_start, t_end, t_up, t_down = timings
    cap = cv2.VideoCapture(input_path)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        current_time = frame_idx / fps
        
        # Logic for timers
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
        
        # Draw Ignore Line
        cv2.line(frame, (0, int(height*0.7)), (width, int(height*0.7)), (0,0,100), 1)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            progress_callback(frame_idx / int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            
    cap.release()
    out.release()
    return temp_output.name

# --- Main UI ---
def main():
    st.title("🏁 Pit Stop AI Analyzer V9")
    st.markdown("### X/Y Velocity Logic (Robust)")
    st.info("Detects horizontal stops even with camera/detection jitter.")

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
            
            status.write("Pass 1/2: Tracking X/Y Velocity...")
            df, fps, w, h = extract_telemetry(tfile_in.name, ref_path, progress_bar.progress)
            
            status.write("Calculating Events (Gap Filling & Smoothing)...")
            t_start, t_end, (t_up, t_down) = analyze_timings_xy(df, fps)
            timings = (t_start, t_end, t_up, t_down)
            
            if t_start is None:
                st.error("Could not detect a horizontal stop. (Try a clearer video or check if car detected).")
                # Visualize data anyway for debugging
                base = alt.Chart(df).encode(x='Time')
                line_x = base.mark_line(color='cyan').encode(y=alt.Y('Vel_X_Smooth', title='Horizontal Speed (X)'))
                st.altair_chart(line_x.interactive(), use_container_width=True)
            else:
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
        if t_start and t_end:
            m1.metric("Pit Stop Duration", f"{(t_end - t_start):.2f}s")
            m3.metric("Stop Time (H-Stop)", f"{t_start:.2f}s")
            m4.metric("Go Time (H-Start)", f"{t_end:.2f}s")
        
        if t_up and t_down:
            m2.metric("Tire Change", f"{(t_down - t_up):.2f}s")
        else:
            m2.metric("Tire Change", "--")

        st.subheader("📊 Velocity Telemetry")
        
        base = alt.Chart(df).encode(x='Time')
        
        line_x = base.mark_line(color='cyan').encode(
            y=alt.Y('Vel_X_Smooth', title='Horizontal Speed (X)'),
            tooltip=['Time', 'Vel_X_Smooth']
        )
        
        line_y = base.mark_line(color='orange').encode(
            y=alt.Y('Vel_Y_Smooth', title='Vertical Speed (Y)'),
            tooltip=['Time', 'Vel_Y_Smooth']
        )
        
        if t_start and t_end:
            rect = alt.Chart(pd.DataFrame({'s': [t_start], 'e': [t_end]})).mark_rect(color='green', opacity=0.1).encode(x='s', x2='e')
            st.altair_chart((line_x + line_y + rect).interactive(), use_container_width=True)
        else:
            st.altair_chart((line_x + line_y).interactive(), use_container_width=True)
        
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
