from flask import Flask, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import time
import json
from flask_cors import CORS
from stick_base import detect_action

app = Flask(__name__)

# Enable CORS
CORS(app)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

key_body_parts = {
    "head": [0],  # Nose
    "left_hand": [11, 13, 15],  # Left shoulder, elbow, wrist
    "right_hand": [12, 14, 16],  # Right shoulder, elbow, wrist
    "left_leg": [23, 25, 27],  # Left hip, knee, ankle
    "right_leg": [24, 26, 28],  # Right hip, knee, ankle
}
data = {}
prev_positions = {}  # For storing previous positions for action detection

# Function to process pose landmarks into a dictionary of key body parts
def get_keypoints(pose_landmarks):
    keypoints = {part: [] for part in key_body_parts}
    for part, indices in key_body_parts.items():
        keypoints[part] = [
            (
                pose_landmarks.landmark[i].x,
                pose_landmarks.landmark[i].y,
                pose_landmarks.landmark[i].visibility,
            )
            for i in indices
        ]
    return keypoints


def calculate_average_velocity(curr_keypoints, prev_keypoints, time_diff):
    avg_velocity = {}
    for part, curr_pos_set in curr_keypoints.items():
        total_velocity = 0
        count = 0
        for i, curr_pos in enumerate(curr_pos_set):
            prev_pos = prev_keypoints[part][i]
            velocity = (
                np.linalg.norm(np.array(curr_pos[:2]) - np.array(prev_pos[:2]))
                / time_diff
            )
            total_velocity += velocity
            count += 1
        avg_velocity[part] = total_velocity / count if count else 0
    return avg_velocity


# Threshold for velocity that indicates probable agitation or fast movement
agitation_threshold = 1.5


# SSE stream generator
def generate_stream():
    global data
    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    prev_keypoints = None
    prev_time = None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB for pose estimation
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                # Draw pose landmarks on the frame for visualization
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                curr_time = time.time()
                curr_keypoints = get_keypoints(results.pose_landmarks)

                if prev_keypoints is not None and prev_time is not None:
                    time_diff = curr_time - prev_time

                    # Calculate velocities
                    velocities = calculate_average_velocity(
                        curr_keypoints, prev_keypoints, time_diff
                    )

                    # Detect probable agitation
                    probable_agitation = [
                        part
                        for part, velocity in velocities.items()
                        if velocity > agitation_threshold
                    ]
                    
                    # Detect action using stick_base
                    action, action_log, action_frequency  = detect_action(frame, prev_positions)

                    # Display information on the frame
                    for part, velocity in velocities.items():
                        cv2.putText(
                            frame,
                            f"{part}: {velocity:.2f}",
                            (10, 30 + 20 * list(key_body_parts).index(part)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )

                    if probable_agitation:
                        cv2.putText(
                            frame,
                            "Probable Agitation Detected!",
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    # Create SSE-compatible JSON
                    data = {
                        "timestamp": curr_time,
                        "velocities": velocities,
                        "probable_agitation": probable_agitation, 
                        "action": action,
                        "action_log": action_log,
                        "action_frequency": dict(action_frequency),                       
                    }

                    # # Stream data to the client
                    # yield f"data:{json.dumps(data)}\n\n"

                # Update previous keypoints and time
                prev_keypoints = curr_keypoints
                prev_time = curr_time

            # Display the video feed locally
            cv2.imshow("Pose Estimation", frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


# Flask routes
@app.route("/stream")
def stream():
    return Response(generate_stream(), content_type="text/event-stream")


@app.route("/data")
def return_data():
    return jsonify(data)


@app.route("/")
def index():
    return jsonify({"message": "Server is running. Access the stream at /stream"})


if __name__ == "__main__":
    app.run(debug=True)
