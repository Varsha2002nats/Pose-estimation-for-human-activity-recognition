import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import defaultdict

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize action logs
action_log = []  # To store action logs with start and stop times
action_frequency = defaultdict(int)  # To store the count of each action
current_action = None  # Tracks the current action
action_start_time = None  # Start time of the current action

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

# Function to classify actions based on landmarks
def classify_action(landmarks, height_threshold, prev_positions):
    if landmarks:
        # Extract key points
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

        # Calculate key angles
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

        # Calculate wrist movement for waving and punching
        wrist_movement = max(
            abs(left_wrist[1] - prev_positions.get("left_wrist", left_wrist)[1]),
            abs(right_wrist[1] - prev_positions.get("right_wrist", right_wrist)[1])
        )

        # Update previous positions
        prev_positions["left_wrist"] = left_wrist
        prev_positions["right_wrist"] = right_wrist

        # Sitting detection
        if left_knee_angle < 120 and right_knee_angle < 120 and left_hip_angle > 90 and right_hip_angle > 90:
            return "Sitting"

        # Waving detection
        if wrist_movement > 30 and (left_wrist[1] < left_shoulder[1] or right_wrist[1] < right_shoulder[1]):
            return "Waving"

        # Punching detection
        if left_wrist[0] > left_shoulder[0] + 50 or right_wrist[0] > right_shoulder[0] + 50:
            return "Punching"

        # Default to standing or moving
        if nose[1] > height_threshold:
            return "Standing"
        else:
            return "Moving"

    return "Unknown"

# Function to detect actions
def detect_action(frame, prev_positions):
    global action_log  # To store action logs with start and stop times
    global action_frequency   # To store the count of each action
    global current_action   # Tracks the current action
    global action_start_time   # Start time of the current action
    global current_action
    
    frame_height, frame_width = frame.shape[:2]
    height_threshold = frame_height * 0.5

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    results = pose.process(rgb_frame)

    # Extract landmarks
    landmarks = None
    if results.pose_landmarks:
        landmarks = [
            (lm.x * frame_width, lm.y * frame_height)
            for lm in results.pose_landmarks.landmark
        ]

    # Classify action
    action = classify_action(landmarks, height_threshold, prev_positions)

     # Log actions into action_log and action_frequency
    current_time = time.time()
    if action != current_action:
        # Log the previous action
        if current_action is not None:
            stop_time = current_time
            action_log.append({
                "action": current_action,
                "start": action_start_time,
                "stop": stop_time
            })
            action_frequency[current_action] += 1

        # Update the current action and start time
        current_action = action
        action_start_time = current_time

    return action, action_log, dict(action_frequency)