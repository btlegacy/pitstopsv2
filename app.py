import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
import altair as alt
from scipy.signal import savgol_filter, find_peaks

# --- Configuration ---
st.set_page_config(page_title="Pit Stop Analytics AI", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- PASS 1: Extraction (Hybrid Presence + Motion) ---
def extract_hybrid_telemetry(video_path, progress_callback):
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    telemetry_data = []
    
    # 1. ESTABLISH BASELINE (Empty Pit Stall)
    # We grab the first 10 frames to define "Empty".
    background_frames = []
    for _ in range(10):
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        background_frames.append(gray)
    
    if len(background_frames) > 0:
        baseline_frame = np.median(background_frames, axis=0).astype(dtype=np.uint8)
    else:
        return pd.DataFrame(), fps, width, height

    # Reset video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # ROI: Focus on the center box where the car sits
    roi_x1, roi_x2 = int(width * 0.2), int(width * 0.8)
    roi_y1, roi_y2 = int(height * 0.2), int(height * 0.8)
    
    base_roi = baseline_frame[roi_y1:roi_y2, roi_x1:roi_x2]
    prev_gray_roi = base_roi.copy()
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        curr_roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # --- METRIC A: PRESENCE (Difference from Empty) ---
        # Even if lines are visible, the car changes the pixel colors (livery).
        diff = cv2.absdiff(curr_roi, base_roi)
        _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        # Score 0.0 = Empty, 1.0 = Completely Changed
        presence_score = np.sum(diff_thresh) / diff_thresh.size

        # --- METRIC B: MOTION (Optical Flow) ---
        # Farneback Dense Optical Flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray_roi, curr_roi, None, 
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        fx = flow[..., 0]
        fy = flow[..., 1]
        
        # Magnitude mask (ignore static asphalt noise)
        mag = np.sqrt(fx**2 + fy**2)
        active_pixels = mag > 1.0
        
        # Calculate MEDIAN flow of active pixels
        # Median filters out the chaotic crew movement, focusing on the heavy car body.
        if np.any(active_pixels):
            median_flow_x = np.median(fx[active_pixels])
            median_flow_y = np.median(fy[active_pixels])
        else:
            median_flow_x = 0.0
            median_flow_y = 0.0

        telemetry_data.append({
            "Frame": frame_idx,
            "Time": frame_idx / fps,
            "Presence": presence_score,
            "Flow_X": median_flow_x,
            "Flow_Y": median_flow_y
        })
        
        prev_gray_roi = curr_roi
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            progress_callback(frame_idx / total_frames)

    cap.release()
    return pd.DataFrame(telemetry_data), fps, width, height

# --- PASS 2: Analysis Logic ---
def analyze_hybrid_timings(df, fps):
    # Smooth data
    df['Presence_Smooth'] = savgol_filter(df['Presence'], 15, 3)
    df['Flow_X_Smooth'] = savgol_filter(df['Flow_X'], 15, 3)
    df['Flow_Y_Smooth'] = savgol_filter(df['Flow_Y'], 15, 3)
    
    # --- 1. PIT STOP DETECTION ---
    # Condition:
    # 1. Car must be PRESENT (Presence Score > threshold)
    # 2. Car must be STOPPED (Horizontal Flow approx 0)
    
    PRESENCE_THRESH = 0.15 # 15% of pixels are different from empty
    MOTION_THRESH_X = 0.5  # Pixels/frame horizontal movement allowed
    
    is_present = df['Presence_Smooth'] > PRESENCE_THRESH
    is_still = df['Flow_X_Smooth'].abs() < MOTION_THRESH_X
    
    # Valid Stop = Present AND Still
    valid_stop = is_present & is_still
    
    # Gap Filling (Ignore brief crew occlusions/jitters)
    clean_stop = valid_stop.astype(int).values
    gap_limit = int(fps * 0.5)
    last_idx = -1
    for i in range(len(clean_stop)):
        if clean_stop[i] == 1:
            if last_idx != -1:
                if (i - last_idx) < gap_limit:
                    clean_stop[last_idx:i] = 1
            last_idx = i
            
    df['Is_Stopped'] = clean_stop.astype(bool)
    
    # Find longest block
    blocks = df[df['Is_Stopped']].groupby((df['Is_Stopped'] != df['Is_Stopped'].shift()).cumsum())
    
    t_start, t_end = None, None
    
    if len(blocks) > 0:
        largest_block = blocks.size().idxmax()
        block_data = df.loc[blocks.groups[largest_block]]
        
        duration = block_data.iloc[-1]['Time'] - block_data.iloc[0]['Time']
        if duration > 2.0:
            t_start = block_data.iloc[0]['Time']
            t_end = block_data.iloc[-1]['Time']

    # --- 2. TIRE CHANGE DETECTION (Vertical Jolt) ---
    t_up, t_down = None, None
    
    if t_start and t_end:
        stop_window = df[(df['Time'] >= t_start) & (df['Time'] <= t_end)]
        
        if not stop_window.empty:
            # Look for vertical flow spikes inside the stop
            y_sig = stop_window['Flow_Y_Smooth'].values
            
            # Find peaks
            # Since Y direction depends on camera orientation, we look for largest deviations
            # Positive Peaks
            peaks_pos, _ = find_peaks(y_sig, height=0.3, distance=fps)
            # Negative Peaks
            peaks_neg, _ = find_peaks(-y_sig, height=0.3, distance=fps)
            
            # Collect events
            events = []
            for p in peaks_pos: events.append((stop_window.iloc[p]['Time'], y_sig[p]))
            for p in peaks_neg: events.append((stop_window.iloc[p]['Time'], y_sig[p]))
            
            # Sort by time
            events.sort(key=lambda x: x[0])
            
            if len(events) >= 1:
                t_up = events[0][0]
                if len(events) >= 2:
                    t_down = events[-1][0]
                else:
                    t_down = t_end

    return t_start, t_end, (t_up, t_down)

# --- PASS 3: Render ---
def render_overlay(input_path, timings, fps, width, height, progress_callback):
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
        
        # TIMERS
        val_pit, col_pit = (0.0, (200,200,200))
        if t_start and current_time >= t_start:
            if t_end and current_time >= t_end:
                val_pit = t_end - t_start
                col_pit = (0,0,255)
            else:
                val_pit = current_time - t_start
                col_pit = (0,255,0)

        val_tire, col_tire = (0.0, (200,200,200))
        if t_up and current_time >= t_up:
            if t_down and current_time >= t_down:
                val_tire = t_down - t_up
                col_tire = (0,0,255)
            else:
                val_tire = current_time - t_up
                col_tire = (0,255,255)
        
        # UI
        cv2.rectangle(frame, (width-450, 0), (width, 180), (0,0,0), -1)
        
        cv2.putText(frame, "PIT STOP", (width-430, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{val_pit:.2f}s", (width-180, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col_pit, 3)
        
        cv2.putText(frame, "TIRES (Jacks)", (width-430, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"{val_tire:.2f}s", (width-180, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col_tire, 3)

        # Add logic explanation
        cv2.putText(frame, "Logic: Presence (High) + Horiz Flow (0)", (width-430, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

        # Draw Ignore Line for reference (bottom 10%)
        cv2.line(frame, (0, int(height*0.9)), (width, int(height*0.9)), (0,0,100), 1)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            progress_callback(frame_idx / total_frames)
            
    cap.release()
    out.release()
    return temp_output.name

# --- Main ---
def main():
    st.title("🏁 Pit Stop Analyzer V14")
    st.markdown("### Hybrid (Presence + Zero-Velocity)")
    st.info("Detects when the car is **Present** (different from empty) AND **Stationary** (Zero Median Horizontal Flow).")

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
            st.write("Step 1: Calculating Presence & Flow Metrics...")
            df, fps, w, h = extract_hybrid_telemetry(tfile.name, bar.progress)
            
            st.write("Step 2: Analyzing Stop/Jacks Logic...")
            t_start, t_end, (t_up, t_down) = analyze_hybrid_timings(df, fps)
            timings = (t_start, t_end, t_up, t_down)
            
            if t_start is None:
                st.error("Could not detect a valid stop. (Check if video starts empty).")
            else:
                st.write("Step 3: Rendering Overlay...")
                vid_path = render_overlay(tfile.name, timings, fps, w, h, bar.progress)
                
                st.session_state.update({
                    'df': df, 'video_path': vid_path, 'timings': timings, 'analysis_done': True
                })
            bar.empty()
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists(tfile.name): os.remove(tfile.name)

    if st.session_state['analysis_done']:
        df = st.session_state['df']
        vid_path = st.session_state['video_path']
        t_start, t_end, t_up, t_down = st.session_state['timings']
        
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Pit Stop Time", f"{t_end - t_start:.2f}s")
        if t_up and t_down:
            c2.metric("Tire Change Time", f"{t_down - t_up:.2f}s")
        
        st.subheader("📊 Hybrid Telemetry")
        base = alt.Chart(df).encode(x='Time')
        
        # Presence (Purple Area)
        pres = base.mark_area(color='purple', opacity=0.3).encode(
            y=alt.Y('Presence_Smooth', title='Car Presence Score')
        )
        
        # Horizontal Flow (Blue Line)
        flow = base.mark_line(color='cyan').encode(
            y=alt.Y('Flow_X_Smooth', title='Horiz. Speed (Median)')
        )
        
        # Vertical Flow (Orange Line)
        vflow = base.mark_line(color='orange').encode(
            y=alt.Y('Flow_Y_Smooth', title='Vert. Speed (Jacks)')
        )
        
        rules = pd.DataFrame([{'t': t_start, 'c': 'green'}, {'t': t_end, 'c': 'red'}])
        rule_chart = alt.Chart(rules).mark_rule(strokeWidth=2).encode(x='t', color=alt.Color('c', scale=None))
        
        st.altair_chart((pres + flow + vflow + rule_chart).interactive(), use_container_width=True)
        
        st.subheader("Video Result")
        c1, c2 = st.columns([3,1])
        with c1:
            if os.path.exists(vid_path): st.video(vid_path)
        with c2:
            if os.path.exists(vid_path):
                with open(vid_path, 'rb') as f:
                    st.download_button("Download MP4", f, file_name="pitstop_hybrid.mp4")

if __name__ == "__main__":
    main()
