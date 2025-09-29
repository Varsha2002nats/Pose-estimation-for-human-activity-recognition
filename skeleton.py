import cv2
import numpy as np
import mediapipe as mp
from tracker import Tracker  

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Initialize Tracker
tracker = Tracker(dist_thresh=0.4, max_humans=10)

def get_skeletons(results, frame_width, frame_height):
    """Extract skeletons from MediaPipe results."""
    skeletons = []
    if results.pose_landmarks:
        # Extract all landmarks into a single list
        skeleton = []
        for landmark in results.pose_landmarks.landmark:
            x, y, visibility = landmark.x, landmark.y, landmark.visibility
            skeleton.extend([x * frame_width, y * frame_height, visibility])
        skeletons.append(skeleton)  # Append the processed skeleton to the list
    return skeletons

# Start Webcam Feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    frame_height, frame_width, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect Pose
    results = pose.process(rgb_frame)

    # Extract skeletons and track them
    skeletons = get_skeletons(results, frame_width, frame_height)
    tracked_skeletons = tracker.track(skeletons)

    # Draw skeletons and IDs on the frame
    for human_id, skeleton in tracked_skeletons.items():
        # Draw skeleton
        for i in range(0, len(skeleton), 3):
            x, y, visibility = skeleton[i], skeleton[i+1], skeleton[i+2]
            if visibility > 0.5:  # Only draw visible keypoints
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Display human ID near the neck
        neck_x, neck_y = skeleton[2], skeleton[3]
        cv2.putText(frame, f'ID: {human_id}', (int(neck_x), int(neck_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the video feed
    cv2.imshow("Real-Time Skeleton Tracking", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
