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

# --- Load AI Model ---
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

# --- Helper: Feature Matching (Reference Image) ---
def get_ref_features(ref_image):
    """Compute ORB keypoints and descriptors for the reference image."""
    if ref_image is None: return None, None
    orb = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(ref_image, None)
    return kp, des

def match_features(frame_gray, ref_des, orb_detector):
    """Check if reference features exist in the current frame."""
    if ref_des is None: return False, []
    
    kp_frame, des_frame = orb_detector.detectAndCompute(frame_gray, None)
    if des_frame is None: return False, []
    
    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(ref_des, des_frame)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Keep top matches
    good_matches = matches[:50]
    
    # Extract points
    src_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return len(good_matches) > 15, src_pts # Threshold of 15 matches

# --- Analysis Engine ---
def process_video_with_ai(video_path, ref_img_path, progress_callback):
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load Reference Image if provided
    ref_des = None
    orb = None
    if ref_img_path:
        ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create(nfeatures=1000)
        _, ref_des = get_ref_features(ref_img)

    # Setup output
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    telemetry_data = []
    prev_gray = None
    
    # Zones
    stall_x_start = int(width * 0.20) # Widen slightly (20% to 80%) to catch entry
    stall_x_end = int(width * 0.80)
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # 1. Motion Score
        motion_score = 0.0
        if prev_gray is not None:
            delta = cv2.absdiff(prev_gray, gray_blur)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_score = np.sum(thresh) / 255.0
        prev_gray = gray_blur

        # 2. AI Detection (YOLO)
        results = model.track(frame, persist=True, classes=[2], verbose=False, conf=0.15) # Lower conf to catch fisheye car
        
        car_detected = False
        car_center_x = -1
        
        if results[0].boxes.id is not None:
            # Get largest car box
            boxes = results[0].boxes.xywh.cpu().numpy()
            # Find box with largest area (closest/main car)
            areas = boxes[:, 2] * boxes[:, 3]
            largest_idx = np.argmax(areas)
            bx, by, bw, bh = boxes[largest_idx]
            
            car_center_x = bx
            car_detected = True

        # 3. Reference Match Fallback
        ref_match_detected = False
        if not car_detected and ref_des is not None:
            match_found, points = match_features(gray, ref_des, orb)
            if match_found:
                ref_match_detected = True
                # Calculate centroid of matched points
                car_center_x = np.mean(points[:, 0, 0])
        
        # 4. In Stall Logic
        in_stall = False
        if car_detected or ref_match_detected:
            if stall_x_start < car_center_x < stall_x_end:
                in_stall = True

        # Draw Overlay
        annotated_frame = results[0].plot()
        
        # Draw Stall Zone
        color = (0,255,0) if in_stall else (255,255,0)
        cv2.line(annotated_frame, (stall_x_start, 0), (stall_x_start, height), color, 2)
        cv2.line(annotated_frame, (stall_x_end, 0), (stall_x_end, height), color, 2)
        
        status_txt = "CAR FOUND" if (car_detected or ref_match_detected) else "SEARCHING..."
        cv2.putText(annotated_frame, status_txt, (stall_x_start, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        out.write(annotated_frame)

        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Motion_Intensity": motion_score,
            "In_Stall": in_stall,
            "Car_Center_X": car_center_x
        })

        frame_idx += 1
        if frame_idx % 20 == 0:
            progress_callback(frame_idx / total_frames)

    cap.release()
    out.release()
    
    return pd.DataFrame(telemetry_data), temp_output.name, fps

def analyze_timings_valley(df, fps):
    """
    Robust Valley Detection:
    1. Identify the "Arrival Peak" (High motion).
    2. Identify the "Departure Peak" (High motion).
    3. The pit stop is the 'Quiet Zone' in between.
    """
    # Smooth motion
    df['Motion_Smooth'] = savgol_filter(df['Motion_Intensity'], 15, 3)
    
    # Heuristic: A "Spike" is motion > 20% of max motion
    max_motion = df['Motion_Smooth'].max()
    threshold_high = max_motion * 0.25  # Threshold for Arrival/Departure
    threshold_low = max_motion * 0.10   # Threshold for "Stopped"
    
    # Find peaks (Arrival and Departure)
    # We look for peaks with a minimum distance between them (e.g., 5 seconds * fps)
    peaks, properties = find_peaks(df['Motion_Smooth'], height=threshold_high, distance=fps*5)
    
    t_start, t_end = None, None
    
    if len(peaks) >= 2:
        # Assume First Peak is Arrival, Last Peak is Departure
        arrival_peak_idx = peaks[0]
        departure_peak_idx = peaks[-1]
        
        # REFINEMENT: Find exact Start (when motion DROPS after arrival)
        # Scan forward from arrival peak until motion < threshold_low
        start_idx = arrival_peak_idx
        for i in range(arrival_peak_idx, len(df)):
            if df['Motion_Smooth'].iloc[i] < threshold_low:
                start_idx = i
                break
                
        # REFINEMENT: Find exact End (when motion RISES before departure)
        # Scan backward from departure peak
        end_idx = departure_peak_idx
        for i in range(departure_peak_idx, start_idx, -1):
            if df['Motion_Smooth'].iloc[i] < threshold_low:
                end_idx = i
                break
        
        t_start = df.iloc[start_idx]['Time']
        t_end = df.iloc[end_idx]['Time']
    else:
        # Fallback if peaks aren't distinct: Use simple thresholding logic
        # Find longest contiguous block of "Low Motion" in the middle 80% of video
        mid_df = df.iloc[int(len(df)*0.1):int(len(df)*0.9)]
        low_motion_mask = mid_df['Motion_Smooth'] < threshold_low
        
        # Identify blocks
        mid_df = mid_df.copy() # avoid setting on copy warning
        mid_df['block'] = (low_motion_mask != low_motion_mask.shift()).cumsum()
        blocks = mid_df[low_motion_mask].groupby('block')
        
        if len(blocks) > 0:
            largest_block = blocks.size().idxmax()
            segment = mid_df[mid_df['block'] == largest_block]
            t_start = segment['Time'].min()
            t_end = segment['Time'].max()

    # Tire Change Detection (Jacks)
    # Look for spikes inside the determined window
    t_up, t_down = t_start, t_end
    if t_start and t_end:
        stop_window = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        if not stop_window.empty:
            motion_curve = stop_window['Motion_Intensity'].values
            # Find internal peaks (crew activity)
            # Use a lower prominence to catch jacks
            jacks_peaks, _ = find_peaks(motion_curve, prominence=np.max(motion_curve)*0.15)
            
            if len(jacks_peaks) > 0:
                times = stop_window.iloc[jacks_peaks]['Time'].values
                t_up = times[0]
                t_down = times[-1]

    return t_start, t_end, (t_up, t_down)

# --- Main UI ---
def main():
    st.title("🏁 Pit Stop AI Analyzer V4")
    st.markdown("### Hybrid Motion & Reference Matching")
    st.info("Uses Global Motion to find the 'Quiet Valley' (Stop) between Arrival/Departure spikes. Can use Reference Image to aid detection.")

    # Sidebar for Reference Image
    st.sidebar.header("Configuration")
    ref_file = st.sidebar.file_uploader("Upload Car Reference Image (Optional)", type=['jpg', 'png'])
    
    # Try to find default ref if not uploaded
    default_ref_path = os.path.join("files", "refs", "car.jpg") # Adjust based on your repo structure if known
    ref_path = None

    if ref_file:
        # Save uploaded ref
        tref = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        tref.write(ref_file.read())
        tref.close()
        ref_path = tref.name
        st.sidebar.success("Reference Image Loaded")
    elif os.path.exists(default_ref_path):
        ref_path = default_ref_path
        st.sidebar.info(f"Using repository reference: {default_ref_path}")

    # Session State
    if 'analysis_done' not in st.session_state:
        st.session_state.update({
            'analysis_done': False, 
            'df': None, 
            'video_path': None, 
            'timings': None
        })

    uploaded_file = st.file_uploader("Upload Overhead Video", type=["mp4", "mov", "avi"])

    if uploaded_file and st.button("Start Analysis", type="primary"):
        st.session_state['analysis_done'] = False
        
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile_in.write(uploaded_file.read())
        tfile_in.flush()

        try:
            progress_bar = st.progress(0)
            status = st.empty()
            status.write("Analyzing Motion Profile & Detecting Object...")
            
            df, vid_path, fps = process_video_with_ai(tfile_in.name, ref_path, progress_bar.progress)
            
            status.write("Identifying Pit Stop Window...")
            t_start, t_end, (t_up, t_down) = analyze_timings_valley(df, fps)
            
            if t_start is None:
                st.error("Could not identify a clear pit stop window (Arrival -> Stop -> Departure).")
            else:
                st.session_state.update({
                    'df': df,
                    'video_path': vid_path,
                    'timings': (t_start, t_end, t_up, t_down),
                    'analysis_done': True
                })
            
            progress_bar.empty()
            status.empty()
            
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists(tfile_in.name): os.remove(tfile_in.name)
            if ref_path and ref_file: os.remove(ref_path) # clean up uploaded ref

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
        
        # Chart
        base = alt.Chart(df).encode(x='Time')
        
        # Motion Area
        area = base.mark_area(color='gray', opacity=0.3).encode(
            y=alt.Y('Motion_Intensity', title='Motion Intensity')
        )
        
        # Smooth line
        line = base.mark_line(color='blue', opacity=0.5).encode(
            y='Motion_Smooth'
        )

        # Stop Window Highlight
        stop_rect = alt.Chart(pd.DataFrame({'start': [t_start], 'end': [t_end]})).mark_rect(
            color='green', opacity=0.1
        ).encode(
            x='start', x2='end'
        )
        
        rules = pd.DataFrame([
            {"Time": t_start, "Event": "Stop Start", "Color": "green"},
            {"Time": t_end, "Event": "Stop End", "Color": "red"},
        ])
        
        rule_chart = alt.Chart(rules).mark_rule(strokeWidth=3).encode(
            x='Time', color=alt.Color('Color', scale=None), tooltip=['Event', 'Time']
        )
        
        st.altair_chart((area + line + stop_rect + rule_chart).interactive(), use_container_width=True)
        
        st.subheader("👁️ Analysis Overlay")
        c1, c2 = st.columns([3,1])
        with c1:
            if os.path.exists(vid_path): st.video(vid_path)
        with c2:
            if os.path.exists(vid_path):
                with open(vid_path, 'rb') as f:
                    st.download_button("Download Video", f, file_name="analyzed_pitstop.mp4")

if __name__ == "__main__":
    main()
