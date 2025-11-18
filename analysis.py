import cv2
import numpy as np

def get_car_centroid(frame, prev_centroid):
    """
    Detect the car in the frame and return its centroid.
    This is a simplified detection method assuming the car is the largest moving object.
    """
    # Convert to grayscale and apply blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # A simple threshold to find large bright objects (like the car)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    
    # Dilate to fill in holes
    dilated = cv2.dilate(thresh, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return prev_centroid # Return previous if no contour found

    # Find the largest contour, which we assume is the car
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Check if the contour is reasonably large
    if cv2.contourArea(largest_contour) < 5000: # Threshold for car size
        return prev_centroid

    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return prev_centroid
        
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return (cX, cY)

def calculate_pit_stop_time(video_path, stop_threshold=2, move_threshold=5, buffer_frames=5):
    """
    Calculates the total pit stop time from a video.

    Args:
        video_path (str): The path to the video file.
        stop_threshold (int): The maximum pixel change in x-axis to be considered 'stopped'.
        move_threshold (int): The minimum pixel change in x-axis to be considered 'moving'.
        buffer_frames (int): Number of consecutive frames needed to confirm a state change.

    Returns:
        A tuple containing (total_time, start_frame, end_frame, fps).
        Raises an exception if a pit stop cannot be determined.
    """
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
    
    # Initialize centroid to the center of the frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    current_centroid = (width // 2, height // 2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        
        # We only start processing in the middle part of the video to avoid initial/final noise
        if frame_number < total_frames * 0.1:
            continue

        current_centroid = get_car_centroid(frame, current_centroid)
        
        if prev_centroid_x is not None:
            delta_x = abs(current_centroid[0] - prev_centroid_x)

            # --- State Machine ---
            if not pit_stop_started:
                # Check for car stopping
                if delta_x < stop_threshold:
                    stopped_frames_count += 1
                else:
                    stopped_frames_count = 0 # Reset if it moves

                if stopped_frames_count >= buffer_frames:
                    pit_stop_started = True
                    start_frame = frame_number - buffer_frames
                    stopped_frames_count = 0 # Reset counter

            else: # Pit stop has started, now check for car moving again
                if delta_x > move_threshold:
                    moving_frames_count += 1
                else:
                    moving_frames_count = 0 # Reset if it stops moving

                if moving_frames_count >= buffer_frames:
                    end_frame = frame_number - buffer_frames
                    cap.release()
                    total_time_seconds = (end_frame - start_frame) / fps
                    return total_time_seconds, start_frame, end_frame, fps
        
        prev_centroid_x = current_centroid[0]

    cap.release()
    raise ValueError("Pit stop start or end could not be determined in the video.")
