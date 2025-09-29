import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between 3 points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# Function to classify action
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

        # Angles
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

        # Wrist movement
        wrist_movement = max(
            abs(left_wrist[1] - prev_positions.get("left_wrist", left_wrist)[1]),
            abs(right_wrist[1] - prev_positions.get("right_wrist", right_wrist)[1])
        )
        prev_positions["left_wrist"] = left_wrist
        prev_positions["right_wrist"] = right_wrist

        # Rules
        if left_knee_angle < 120 and right_knee_angle < 120 and left_hip_angle > 90 and right_hip_angle > 90:
            return "Sitting"
        if left_wrist[0] > left_shoulder[0] + 50 or right_wrist[0] > right_shoulder[0] + 50:
            return "Punching"
        kick_angle_threshold = 160
        if left_knee_angle < kick_angle_threshold or right_knee_angle < kick_angle_threshold:
            return "Kicking"
        if nose[1] > height_threshold:
            return "Standing"
        else:
            return "Moving"
    return "Unknown"

# Setup webcam
cap = cv2.VideoCapture(0)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
prev_positions = {}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

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
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
        )

        landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in results.pose_landmarks.landmark]

    # Determine the action
    height_threshold = frame_height * 0.5
    action = classify_action(landmarks, height_threshold, prev_positions)

    # Display action on black frame
    cv2.rectangle(black_frame, (0, 0), (300, 50), (0, 0, 0), -1)
    cv2.putText(black_frame, f'Action: {action}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Combine both views side-by-side
    combined_view = np.hstack((frame, black_frame))

    # Show the combined output
    cv2.imshow('Live View + Stick Figure View', combined_view)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()