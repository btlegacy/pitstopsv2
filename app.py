import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

# --- Model Loading ---
@st.cache_resource
def get_predictor():
    """
    Initializes and returns the Detectron2 predictor.
    """
    cfg = get_cfg()
    # This config file is compatible with the older detectron2 version
    config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.DEVICE = "cpu"
    
    predictor = DefaultPredictor(cfg)
    return predictor, MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

# --- Car Detection ---
def get_car_centroid(predictor, metadata, frame, prev_centroid):
    """
    Detects the car in the frame using Detectron2 and returns its centroid.
    """
    outputs = predictor(frame)
    
    instances = outputs["instances"]
    car_class_index = metadata.thing_classes.index("car")
    
    car_boxes = instances[instances.pred_classes == car_class_index].pred_boxes.tensor.numpy()
    car_scores = instances[instances.pred_classes == car_class_index].scores.numpy()

    if len(car_boxes) == 0:
        return prev_centroid

    best_car_index = np.argmax(car_scores)
    box = car_boxes[best_car_index]
    
    x1, y1, x2, y2 = box
    cX = int((x1 + x2) / 2)
    cY = int((y1 + y2) / 2)

    return (cX, cY)

# --- Pit Stop Calculation (Main Logic) ---
def calculate_pit_stop_time(video_path, stop_threshold=3, move_threshold=5, buffer_frames=5):
    """
    Calculates the total pit stop time from a video using Detectron2.
    """
    # Note: I'm adding a decorator to the get_predictor function to cache it with Streamlit
    # This is a performance improvement. Let's make sure your app.py calls it.
    predictor, metadata = get_predictor()
    
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        
        if frame_number < total_frames * 0.1:
            continue
        
        current_centroid = get_car_centroid(predictor, metadata, frame, current_centroid)
        
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
                    stopped_frames_count = 0

            else:
                if delta_x > move_threshold:
                    moving_frames_count += 1
                else:
                    moving_frames_count = 0

                if moving_frames_count >= buffer_frames:
                    end_frame = frame_number - buffer_frames
                    cap.release()
                    total_time_seconds = (end_frame - start_frame) / fps
                    return total_time_seconds, start_frame, end_frame, fps
        
        prev_centroid_x = current_centroid[0]

    cap.release()
    raise ValueError("Pit stop start or end could not be determined.")
