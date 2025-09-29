import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Key body parts and indices
key_body_parts = {
    "head": [0],  # Nose
    "left_hand": [11, 13, 15],
    "right_hand": [12, 14, 16],
    "left_leg": [23, 25, 27],
    "right_leg": [24, 26, 28],
}

# Velocity threshold
agitation_threshold = 1.0

# --- Helper functions ---

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

def classify_action(landmarks, height_threshold, prev_positions):
    if landmarks:
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

        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

        wrist_movement = max(
            abs(left_wrist[1] - prev_positions.get("left_wrist", left_wrist)[1]),
            abs(right_wrist[1] - prev_positions.get("right_wrist", right_wrist)[1])
        )
        prev_positions["left_wrist"] = left_wrist
        prev_positions["right_wrist"] = right_wrist

        if left_knee_angle < 120 and right_knee_angle < 120 and left_hip_angle > 90 and right_hip_angle > 90:
            return "Sitting"
        if left_wrist[0] > left_shoulder[0] + 50 or right_wrist[0] > right_shoulder[0] + 50:
            return "Punching"
        kick_threshold = 160
        if left_knee_angle < kick_threshold or right_knee_angle < kick_threshold:
            return "Kicking"
        if nose[1] > height_threshold:
            return "Standing"
        else:
            return "Moving"
    return "Unknown"

def get_keypoints(pose_landmarks):
    keypoints = {}
    for part, indices in key_body_parts.items():
        keypoints[part] = [
            (pose_landmarks.landmark[i].x, pose_landmarks.landmark[i].y)
            for i in indices
        ]
    return keypoints

def calculate_average_velocity(curr_keypoints, prev_keypoints, time_diff):
    velocities = {}
    for part, curr_pts in curr_keypoints.items():
        total = 0
        for i in range(len(curr_pts)):
            curr = np.array(curr_pts[i])
            prev = np.array(prev_keypoints[part][i])
            total += np.linalg.norm(curr - prev) / time_diff
        velocities[part] = total / len(curr_pts)
    return velocities

# --- Main script ---

cap = cv2.VideoCapture(0)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
prev_positions = {}
prev_keypoints = None
prev_time = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    landmarks = None
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            black_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in results.pose_landmarks.landmark]

        height_threshold = frame_height * 0.5
        action = classify_action(landmarks, height_threshold, prev_positions)

        # Velocity analysis
        curr_time = time.time()
        curr_keypoints = get_keypoints(results.pose_landmarks)
        velocities = {}
        if prev_keypoints is not None and prev_time is not None:
            time_diff = curr_time - prev_time
            velocities = calculate_average_velocity(curr_keypoints, prev_keypoints, time_diff)

            for idx, (part, vel) in enumerate(velocities.items()):
                color = (0, 255, 0) if vel < agitation_threshold else (0, 0, 255)
                cv2.putText(
                    black_frame, f"{part}: {vel:.2f}",
                    (10, 60 + idx * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

            if any(v > agitation_threshold for v in velocities.values()):
                cv2.putText(
                    black_frame, "Probable Agitation Detected!",
                    (300, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2
                )

        # Show action label
        cv2.putText(black_frame, f'Action: {action}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        prev_keypoints = curr_keypoints
        prev_time = curr_time

    combined_view = np.hstack((frame, black_frame))
    cv2.imshow("Real View + Stick Figure with Velocity", combined_view)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()