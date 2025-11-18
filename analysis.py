import cv2
import numpy as np
import mmcv
from mmdet.apis import init_detector, inference_detector
import tempfile
import os
import streamlit as st

# --- Model Loading ---
@st.cache_resource
def get_mmdet_model():
    """
    Initializes and returns the MMDetection model.
    """
    # Using a popular and effective model: Faster R-CNN
    config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # The checkpoint file will be downloaded automatically by MMDetection the first time.
    checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    # MMDetection needs the config and checkpoint files locally. We'll download them.
    # Base URL for MMDetection model zoo
    base_url = 'https://raw.githubusercontent.com/open-mmlab/mmdetection/v2.20.0/'

    # Create directories if they don't exist
    os.makedirs('configs/faster_rcnn', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Download config if it doesn't exist
    if not os.path.exists(config_file):
        print(f"Downloading {config_file}...")
        os.system(f"wget -O {config_file} {base_url}{config_file}")

    # Download checkpoint if it doesn't exist
    if not os.path.exists(checkpoint_file):
        print(f"Downloading {checkpoint_file}...")
        os.system(f"wget -O {checkpoint_file} https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth")

    # Initialize the detector
    model = init_detector(config_file, checkpoint_file, device='cpu')
    return model

# --- Car Detection ---
def get_car_centroid(model, frame, prev_centroid):
    """
    Detects the car in the frame using MMDetection and returns its centroid.
    """
    # MMDetection's inference function returns results for all 80 COCO classes.
    result = inference_detector(model, frame)
    
    # The class index for 'car' in the COCO dataset is 2.
    car_class_index = 2
    
    # Get the bounding boxes for the 'car' class
    car_boxes = result[car_class_index]

    if car_boxes.shape[0] == 0:
        return prev_centroid # Return previous if no car is detected

    # Find the car with the highest confidence score
    best_car_index = np.argmax(car_boxes[:, 4]) # Confidence score is the 5th element
    best_car_box = car_boxes[best_car_index]
    
    # Filter out low-confidence detections
    if best_car_box[4] < 0.6:
        return prev_centroid

    x1, y1, x2, y2, _ = best_car_box
    
    # Calculate the centroid of the bounding box
    cX = int((x1 + x2) / 2)
    cY = int((y1 + y2) / 2)

    return (cX, cY)

# --- Pit Stop Calculation (Main Logic) ---
def calculate_pit_stop_time(video_path, stop_threshold=3, move_threshold=5, buffer_frames=5):
    """
    Calculates the total pit stop time from a video using MMDetection.
    """
    model = get_mmdet_model()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_centroid_x = None
    stopped_frames_count = 0
    moving_frames_count = 0
    
    pit_stop_started = False
    start_frame = 0
    end_frame = 0
    
    frame_number = 0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    current_centroid = (width // 2, height // 2)

    st_progress_bar = st.progress(0)
    st_text = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        
        # Update progress bar
        progress = frame_number / total_frames
        st_progress_bar.progress(progress)
        st_text.text(f"Processing frame {frame_number}/{int(total_frames)}")
        
        current_centroid = get_car_centroid(model, frame, current_centroid)
        
        if prev_centroid_x is not None:
            delta_x = abs(current_centroid[0] - prev_centroid_x)

            if not pit_stop_started:
                if delta_x < stop_threshold:
                    stopped_frames_count += 1
                else:
                    stopped_frames_count = 0 

                if stopped_frames_count >= buffer_frames:
                    pit_stop_started = True
                    start_frame = frame_number - buffer_frames
                    st_text.text("Pit stop start detected!")
                    stopped_frames_count = 0

            else:
                if delta_x > move_threshold:
                    moving_frames_count += 1
                else:
                    moving_frames_count = 0

                if moving_frames_count >= buffer_frames:
                    end_frame = frame_number - buffer_frames
                    cap.release()
                    st_progress_bar.empty()
                    st_text.empty()
                    total_time_seconds = (end_frame - start_frame) / fps
                    return total_time_seconds, start_frame, end_frame, fps
        
        prev_centroid_x = current_centroid[0]

    cap.release()
    st_progress_bar.empty()
    st_text.empty()
    raise ValueError("Pit stop start or end could not be determined.")
