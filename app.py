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

# --- Analysis Engine ---
def process_video_with_ai(video_path, progress_callback):
    """
    Runs YOLOv8 tracking.
    Records Position (X,Y) and Velocity to determine if car is in the box.
    """
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup output video writer
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    telemetry_data = []
    
    # Tracking variables
    prev_car_center = None
    prev_gray = None
    
    # Define the "Pit Stall Zone" (The middle 50% of the screen width)
    stall_x_start = int(width * 0.25)
    stall_x_end = int(width * 0.75)
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. AI Inference
        results = model.track(frame, persist=True, classes=[0, 2], verbose=False, conf=0.3)
        
        # 2. Extract Data
        car_center = None
        crew_count = 0
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            
            for box, cls in zip(boxes, class_ids):
                x, y, w, h = box
                if int(cls) == 0: # Person
                    crew_count += 1
                elif int(cls) == 2: # Car
                    car_center = (x, y)

        # 3. Calculate Car Velocity & State
        velocity = 0.0
        in_stall = False
        
        if car_center is not None:
            # Check if center of car is in the "Stall Zone"
            if stall_x_start < car_center[0] < stall_x_end:
                in_stall = True
                
            if prev_car_center is not None:
                dx = car_center[0] - prev_car_center[0]
                dy = car_center[1] - prev_car_center[1]
                velocity = np.sqrt(dx**2 + dy**2)
        
        prev_car_center = car_center

        # 4. Calculate Pixel Motion (Secondary signal for Jacks)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        motion_score = 0.0
        
        if prev_gray is not None:
            delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_score = np.sum(thresh) / 255.0
        
        prev_gray = gray

        # 5. Draw Overlay
        annotated_frame = results[0].plot()
        
        # Draw Pit Stall Lines (The "Ground Lines" reference)
        cv2.line(annotated_frame, (stall_x_start, 0), (stall_x_start, height), (255, 255, 0), 2)
        cv2.line(annotated_frame, (stall_x_end, 0), (stall_x_end, height), (255, 255, 0), 2)
        cv2.putText(annotated_frame, "PIT STALL ZONE", (stall_x_start + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Status Text
        # We use a visual threshold of 2.0 for the overlay text
        status_color = (0, 255, 0) if in_stall and velocity < 2.0 else (0, 0, 255)
        status_text = "STOPPED IN BOX" if in_stall and velocity < 2.0 else "MOVING / OUT OF BOX"
        cv2.putText(annotated_frame, status_text, (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
        
        out.write(annotated_frame)

        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Velocity": velocity,
            "Car_X": car_center[0] if car_center else -1,
            "In_Stall": in_stall,
            "Motion_Intensity": motion_score
        })

        frame_idx += 1
        if frame_idx % 20 == 0:
            progress_callback(frame_idx / total_frames)

    cap.release()
    out.release()
    
    return pd.DataFrame(telemetry_data), temp_output.name, fps

def analyze_timings(df, fps):
    # 1. Smooth Velocity
    if len(df) > 15:
        df['Velocity_Smooth'] = savgol_filter(df['Velocity'], 15, 3)
    else:
        df['Velocity_Smooth'] = df['Velocity']

    # 2. Detect VALID Stop State
    # Condition: Velocity is Low AND Car is physically inside the Pit Stall Zone
    stop_threshold = 2.0 
    df['Valid_Stop'] = (df['Velocity_Smooth'] < stop_threshold) & (df['In_Stall'] == True)
    
    # Find the longest continuous block of "Valid_Stop"
    df['block'] = (df['Valid_Stop'] != df['Valid_Stop'].shift()).cumsum()
    blocks = df[df['Valid_Stop']].groupby('block')
    
    if len(blocks) == 0:
        return None, None, None

    largest_block = blocks.size().idxmax()
    stop_segment = df[df['block'] == largest_block]
    
    start_frame = stop_segment['Frame'].min()
    end_frame = stop_segment['Frame'].max()
    
    t_start = start_frame / fps
    t_end = end_frame / fps
    
    # 3. Detect Jacks (Peaks inside the valid stop window)
    stop_window_df = df[(df['Frame'] >= start_frame) & (df['Frame'] <= end_frame)]
    motion_curve = stop_window_df['Motion_Intensity'].values
    
    if len(motion_curve) > 0:
        peaks, _ = find_peaks(motion_curve, prominence=np.max(motion_curve)*0.2, distance=fps/2)
        if len(peaks) > 0:
            peak_times = stop_window_df.iloc[peaks]['Time'].values
            t_jacks_up = peak_times[0]
            t_jacks_down = peak_times[-1]
        else:
            t_jacks_up = t_start
            t_jacks_down = t_end
    else:
        t_jacks_up = t_start
        t_jacks_down = t_end

    return t_start, t_end, (t_jacks_up, t_jacks_down)

# --- Main UI ---
def main():
    st.title("🏁 Pit Stop AI Analyzer")
    st.markdown("### Automated Analysis with Stall Detection")
    st.info("The timer only starts when the car is **Stopped** inside the **Center Zone** of the video.")

    # Session State
    if 'analysis_done' not in st.session_state:
        st.session_state['analysis_done'] = False
        st.session_state['df'] = None
        st.session_state['video_path'] = None
        st.session_state['timings'] = None

    uploaded_file = st.file_uploader("Upload Overhead Video", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        if st.button("Start AI Analysis", type="primary"):
            st.session_state['analysis_done'] = False
            
            tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
            tfile_in.write(uploaded_file.read())
            tfile_in.flush()

            try:
                progress_bar = st.progress(0)
                status = st.empty()
                status.write("Initializing AI... Detecting Stall Zone...")
                
                df, vid_path, fps = process_video_with_ai(tfile_in.name, progress_bar.progress)
                
                status.write("Calculating Valid Stop Times...")
                t_start, t_end, (t_up, t_down) = analyze_timings(df, fps)
                
                if t_start is None:
                    st.error("No valid pit stop detected. (Car did not stop in the center zone).")
                else:
                    st.session_state['df'] = df
                    st.session_state['video_path'] = vid_path
                    st.session_state['timings'] = (t_start, t_end, t_up, t_down)
                    st.session_state['analysis_done'] = True
                
                progress_bar.empty()
                status.empty()
                
            except Exception as e:
                st.error(f"Analysis Error: {e}")
            finally:
                try:
                    os.remove(tfile_in.name)
                except:
                    pass

    if st.session_state['analysis_done']:
        df = st.session_state['df']
        vid_path = st.session_state['video_path']
        t_start, t_end, t_up, t_down = st.session_state['timings']

        # Results
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        
        m1.metric("Pit Stop Duration", f"{(t_end - t_start):.2f}s")
        m2.metric("Tire Change", f"{(t_down - t_up):.2f}s")
        m3.metric("Stop Time", f"{t_start:.2f}s")
        m4.metric("Go Time", f"{t_end:.2f}s")

        # Chart
        st.subheader("📊 Telemetry")
        base = alt.Chart(df).encode(x='Time')
        
        line_velocity = base.mark_line(color='cyan').encode(
            y=alt.Y('Velocity_Smooth', title='Car Speed'),
            tooltip=['Time', 'Velocity_Smooth']
        )
        
        # --- FIX: Correctly implemented area highlight using transform_filter ---
        stop_area = base.mark_area(color='green', opacity=0.1).transform_filter(
            alt.datum.Valid_Stop == True
        ).encode(
            y=alt.value(0),
            y2=alt.value(400) # Height in pixels to cover background
        )

        line_motion = base.mark_area(opacity=0.3, color='gray').encode(
            y=alt.Y('Motion_Intensity', title='Activity'),
            tooltip=['Time', 'Motion_Intensity']
        )
        
        rules = pd.DataFrame([
            {"Time": t_start, "Event": "Stop (In Box)", "Color": "green"},
            {"Time": t_end, "Event": "Go", "Color": "red"},
            {"Time": t_up, "Event": "Jacks Up", "Color": "orange"},
            {"Time": t_down, "Event": "Jacks Down", "Color": "orange"}
        ])
        
        rule_chart = alt.Chart(rules).mark_rule(strokeWidth=2).encode(
            x='Time',
            color=alt.Color('Color', scale=None),
            tooltip=['Event', 'Time']
        )
        
        st.altair_chart((stop_area + line_motion + line_velocity + rule_chart).interactive(), use_container_width=True)
        
        st.subheader("👁️ AI Overlay")
        col_v1, col_v2 = st.columns([3, 1])
        with col_v1:
            if os.path.exists(vid_path):
                st.video(vid_path)
        with col_v2:
            if os.path.exists(vid_path):
                with open(vid_path, 'rb') as f:
                    st.download_button("Download Video", f, file_name="analyzed_pitstop.mp4")

if __name__ == "__main__":
    main()
