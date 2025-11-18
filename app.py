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
    # YOLOv8 Nano is fast enough for CPU inference
    return YOLO('yolov8n.pt')

# --- Analysis Engine ---
def process_video_with_ai(video_path, progress_callback):
    """
    Runs YOLOv8 tracking to detect Car and Crew.
    Calculates Car Speed to determine Stop/Go timing.
    Calculates Optical Motion to determine Jacks Up/Down.
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
    # MP4V codec usually works well for temp files, H264 is better for web but requires ffmpeg
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    telemetry_data = []
    
    # Tracking variables
    prev_car_center = None
    prev_gray = None
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. AI Inference (Track Car & Persons)
        # classes: 0=person, 2=car
        results = model.track(frame, persist=True, classes=[0, 2], verbose=False, conf=0.3)
        
        # 2. Extract Data
        car_center = None
        crew_count = 0
        
        # Parse YOLO results
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            
            for box, cls, tid in zip(boxes, class_ids, track_ids):
                x, y, w, h = box
                if int(cls) == 0: # Person
                    crew_count += 1
                elif int(cls) == 2: # Car
                    # Assume the largest car or the one in the center is the race car
                    # Simple logic: just take the first car found
                    car_center = (x, y)

        # 3. Calculate Car Velocity (Pixels per frame)
        velocity = 0.0
        if car_center is not None and prev_car_center is not None:
            dx = car_center[0] - prev_car_center[0]
            dy = car_center[1] - prev_car_center[1]
            velocity = np.sqrt(dx**2 + dy**2)
        
        prev_car_center = car_center

        # 4. Calculate Pixel Motion (for Jacks detection)
        # We use this as a secondary signal because AI doesn't detect "Jacks" well
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        motion_score = 0.0
        
        if prev_gray is not None:
            delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_score = np.sum(thresh) / 255.0
        
        prev_gray = gray

        # 5. Draw Overlay on Video
        annotated_frame = results[0].plot()
        
        # Add custom telemetry text
        cv2.putText(annotated_frame, f"Speed: {velocity:.2f} px/f", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Crew: {crew_count}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(annotated_frame)

        # Store Data
        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Velocity": velocity,
            "Crew_Count": crew_count,
            "Motion_Intensity": motion_score
        })

        frame_idx += 1
        if frame_idx % 20 == 0:
            progress_callback(frame_idx / total_frames)

    cap.release()
    out.release()
    
    return pd.DataFrame(telemetry_data), temp_output.name, fps

def analyze_timings(df, fps):
    """
    Post-process the telemetry data to find Stop/Go events.
    """
    # 1. Smooth Velocity to remove jitter
    if len(df) > 15:
        df['Velocity_Smooth'] = savgol_filter(df['Velocity'], 15, 3)
    else:
        df['Velocity_Smooth'] = df['Velocity']

    # 2. Detect Stop State
    # Logic: Velocity < Threshold for significant duration
    # Pixels per frame threshold depends on resolution, but 2.0 is usually safe for "stopped"
    stop_threshold = 2.0 
    
    df['Is_Stopped'] = df['Velocity_Smooth'] < stop_threshold
    
    # Find the longest continuous block of "Is_Stopped"
    # This handles cases where the car slows down entering pit lane but hasn't stopped yet
    df['block'] = (df['Is_Stopped'] != df['Is_Stopped'].shift()).cumsum()
    blocks = df[df['Is_Stopped']].groupby('block')
    
    if len(blocks) == 0:
        return None, None, None

    # Get the largest stopped block
    largest_block = blocks.size().idxmax()
    stop_segment = df[df['block'] == largest_block]
    
    start_frame = stop_segment['Frame'].min()
    end_frame = stop_segment['Frame'].max()
    
    t_start = start_frame / fps
    t_end = end_frame / fps
    
    # 3. Detect Jacks (Motion Peaks inside the stop window)
    # We look at "Motion_Intensity" strictly inside the stop window
    stop_window_df = df[(df['Frame'] >= start_frame) & (df['Frame'] <= end_frame)]
    
    # Normalize motion
    motion_curve = stop_window_df['Motion_Intensity'].values
    if len(motion_curve) > 0:
        # Find peaks
        peaks, _ = find_peaks(motion_curve, prominence=np.max(motion_curve)*0.2, distance=fps/2)
        
        if len(peaks) > 0:
            # Map back to global time
            peak_times = stop_window_df.iloc[peaks]['Time'].values
            t_jacks_up = peak_times[0] # First big spike
            t_jacks_down = peak_times[-1] # Last big spike
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
    st.markdown("Uses **YOLOv8 Object Detection** to identify the car and crew, and calculates timing based on car velocity.")

    uploaded_file = st.file_uploader("Upload Overhead Video", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Save input
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile_in.write(uploaded_file.read())
        tfile_in.flush()

        if st.button("Start AI Analysis", type="primary"):
            progress_bar = st.progress(0)
            status = st.empty()
            
            status.write("Initializing AI Model & Tracking Car...")
            
            # Run Pipeline
            try:
                df, processed_vid_path, fps = process_video_with_ai(tfile_in.name, progress_bar.progress)
                
                status.write("Calculating Timings...")
                t_start, t_end, (t_up, t_down) = analyze_timings(df, fps)
                
                progress_bar.empty()
                status.empty()

                # --- Layout Results ---
                
                # 1. Metrics
                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                
                pit_duration = t_end - t_start if t_start else 0
                tire_duration = t_down - t_up if t_up else 0
                
                m1.metric("Pit Stop Duration", f"{pit_duration:.2f}s")
                m2.metric("Tire Change", f"{tire_duration:.2f}s")
                m3.metric("Stop Time", f"{t_start:.2f}s")
                m4.metric("Go Time", f"{t_end:.2f}s")

                # 2. Interactive Chart
                st.subheader("📊 Telemetry")
                
                # Prepare chart data
                # Scale velocity for visualization alongside motion
                base = alt.Chart(df).encode(x='Time')
                
                line_velocity = base.mark_line(color='cyan').encode(
                    y=alt.Y('Velocity_Smooth', title='Car Speed'),
                    tooltip=['Time', 'Velocity_Smooth']
                )
                
                line_motion = base.mark_area(opacity=0.3, color='gray').encode(
                    y=alt.Y('Motion_Intensity', title='Activity (Motion)'),
                    tooltip=['Time', 'Motion_Intensity']
                )
                
                # Add rule lines for detected events
                rules = pd.DataFrame([
                    {"Time": t_start, "Event": "Stop", "Color": "green"},
                    {"Time": t_end, "Event": "Go", "Color": "red"},
                    {"Time": t_up, "Event": "Jacks Up", "Color": "orange"},
                    {"Time": t_down, "Event": "Jacks Down", "Color": "orange"}
                ])
                
                rule_chart = alt.Chart(rules).mark_rule(strokeWidth=2).encode(
                    x='Time',
                    color=alt.Color('Color', scale=None),
                    tooltip=['Event', 'Time']
                )
                
                st.altair_chart((line_motion + line_velocity + rule_chart).interactive(), use_container_width=True)
                
                # 3. Processed Video
                st.subheader("👁️ AI Overlay")
                
                # We must convert the cv2 written video to a format streamlit likes (h264)
                # Since we can't rely on ffmpeg being installed on Streamlit Cloud,
                # we display the raw file. Browsers sometimes struggle with mp4v, 
                # but let's try displaying the processed path.
                
                # If browser fails to play 'mp4v', we fallback to showing the original
                # and asking user to trust the chart, OR we rely on the user downloading it.
                
                st.video(processed_vid_path)
                
                with open(processed_vid_path, 'rb') as f:
                    st.download_button("Download Analyzed Video", f, file_name="analyzed_pitstop.mp4")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.error("Note: AI analysis on CPU can be memory intensive. Try a shorter clip if this crashes.")

            finally:
                # Clean up inputs
                try:
                    os.remove(tfile_in.name)
                except:
                    pass

if __name__ == "__main__":
    main()
