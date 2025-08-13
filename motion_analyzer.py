# --- FINAL SCRIPT (YOLOv8-POSE ENGINE) ---
import sys, os, subprocess, collections
import cv2
import numpy as np
from ultralytics import YOLO

# --- Landmark indices for YOLOv8-Pose (different from MediaPipe) ---
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

# --- Define the skeleton structure for YOLOv8-Pose ---
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
LEFT_CONNECTIONS = [(L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE)]
RIGHT_CONNECTIONS = [(R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE)]
CENTER_CONNECTIONS = [(L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP), (L_SHOULDER, R_HIP), (R_SHOULDER, L_HIP)]

def draw_polished_skeleton(keypoints, image):
    height, width, _ = image.shape
    
    # YOLO keypoints are already in pixel coordinates [x, y, confidence]
    landmarks_np = np.array(keypoints)
    
    color_left = (255, 100, 0); color_right = (0, 100, 255); color_center = (200, 200, 200)
    connection_groups = [(LEFT_CONNECTIONS, color_left), (RIGHT_CONNECTIONS, color_right), (CENTER_CONNECTIONS, color_center)]

    for connections, color in connection_groups:
        for connection in connections:
            start_idx, end_idx = connection
            start_pos = (int(landmarks_np[start_idx][0]), int(landmarks_np[start_idx][1]))
            end_pos = (int(landmarks_np[end_idx][0]), int(landmarks_np[end_idx][1]))
            cv2.line(image, start_pos, end_pos, color, 2)
            
    for i, landmark in enumerate(landmarks_np):
        pos = (int(landmark[0]), int(landmark[1]))
        cv2.circle(image, pos, 4, (0, 255, 0), -1) # Green joints
    return image

def analyze_video(video_path, output_video_path):
    print(f"[Analyzer] Starting analysis of {video_path}")
    
    # Load the YOLOv8-Pose model
    model = YOLO('yolov8n-pose.pt')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError("Cannot open video file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))
    
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame_count += 1
        
        # Run pose estimation on the frame
        results = model(frame, verbose=False)
        
        # Check if any poses were detected
        if results and results[0].keypoints and len(results[0].keypoints.xy) > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy() # Get the first detected person's keypoints
            
            black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            skeleton_frame = draw_polished_skeleton(keypoints, black_frame)
            video_writer.write(skeleton_frame)
        else:
            # If no person is detected, write a blank frame
            black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            video_writer.write(black_frame)
            
    print(f"[Analyzer] Processed {frame_count} frames.")
    cap.release()
    video_writer.release()
    print(f"[Analyzer] Analysis complete. Final skeleton video saved to: {output_video_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    analyze_video(sys.argv[1], "skeleton_video_final.mp4")
