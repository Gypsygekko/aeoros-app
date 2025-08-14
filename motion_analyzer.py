# motion_analyzer.py with heartbeat prints
import sys, os, subprocess, collections
# ... (rest of the script is the same, this is just to show where to add the prints) ...

def analyze_video(video_path, output_video_path):
    print("[ANALYZER HEARTBEAT] Script started. Inside analyze_video function.") # <<< DEBUG PRINT
    # Load the YOLOv8-Pose model
    print("[ANALYZER HEARTBEAT] Loading YOLO model...") # <<< DEBUG PRINT
    model = YOLO('yolov8n-pose.pt')
    print("[ANALYZER HEARTBEAT] YOLO model loaded.") # <<< DEBUG PRINT
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ANALYZER HEARTBEAT] FATAL: Could not open video file.")
        raise ValueError("Cannot open video file")
# --- FINAL SCRIPT (SELF-INSTALLING YOLOv8 ENGINE) ---
import sys, os, subprocess, collections

# This block ensures all dependencies are installed when the script runs
try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    print("[Analyzer] Dependencies already satisfied.")
except ImportError:
    print("[Analyzer] Dependencies not found. Installing ultralytics and opencv-python...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics', 'opencv-python-headless'], check=True)
        import cv2
        import numpy as np
        from ultralytics import YOLO
        print("[Analyzer] Dependencies installed successfully.")
    except Exception as e:
        print(f"[Analyzer] FATAL ERROR during dependency installation: {e}")
        sys.exit(1)

# --- Landmark indices and connection definitions for YOLOv8-Pose ---
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
LEFT_CONNECTIONS = [(L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE)]
RIGHT_CONNECTIONS = [(R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE)]
CENTER_CONNECTIONS = [(L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP), (L_SHOULDER, R_HIP), (R_SHOULDER, L_HIP)]

def draw_polished_skeleton(keypoints, image):
    height, width, _ = image.shape
    landmarks_np = np.array(keypoints)
    color_left = (255, 100, 0); color_right = (0, 100, 255); color_center = (200, 200, 200)
    connection_groups = [(LEFT_CONNECTIONS, color_left), (RIGHT_CONNECTIONS, color_right), (CENTER_CONNECTIONS, color_center)]

    for connections, color in connection_groups:
        for connection in connections:
            start_idx, end_idx = connection
            # Check if keypoints are detected with sufficient confidence
            if landmarks_np[start_idx][2] > 0.1 and landmarks_np[end_idx][2] > 0.1:
                start_pos = (int(landmarks_np[start_idx][0]), int(landmarks_np[start_idx][1]))
                end_pos = (int(landmarks_np[end_idx][0]), int(landmarks_np[end_idx][1]))
                cv2.line(image, start_pos, end_pos, color, 2)
            
    for i, landmark in enumerate(landmarks_np):
        if landmark[2] > 0.1: # Only draw confident joints
            pos = (int(landmark[0]), int(landmark[1]))
            cv2.circle(image, pos, 4, (0, 255, 0), -1) # Green joints
    return image

def analyze_video(video_path, output_video_path):
    print(f"[Analyzer] Starting analysis of {video_path}")
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
        
        results = model(frame, verbose=False)
        
        black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        if results and hasattr(results[0], 'keypoints') and results[0].keypoints and len(results[0].keypoints.xy) > 0:
            keypoints = results[0].keypoints.cpu().numpy()[0] # Get the first detected person's keypoints
            skeleton_frame = draw_polished_skeleton(keypoints, black_frame)
            video_writer.write(skeleton_frame)
        else:
            video_writer.write(black_frame)
            
    print(f"[Analyzer] Processed {frame_count} frames.")
    cap.release()
    video_writer.release()
    print(f"[Analyzer] Analysis complete. Final skeleton video saved to: {output_video_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    analyze_video(sys.argv[1], "skeleton_video_final.mp4")

