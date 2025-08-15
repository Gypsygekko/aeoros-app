# --- FINAL SCRIPT (VERBOSE DEBUGGER for YOLOv8) ---
import sys, os, subprocess, collections

try:
    import cv2, numpy as np
    from ultralytics import YOLO
    print("[Analyzer] Dependencies already satisfied.")
except ImportError:
    print("[Analyzer] Dependencies not found. Installing...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics', 'opencv-python-headless'], check=True)
        import cv2, numpy as np
        from ultralytics import YOLO
        print("[Analyzer] Dependencies installed successfully.")
    except Exception as e:
        print(f"[Analyzer] FATAL ERROR during dependency installation: {e}")
        sys.exit(1)

# --- Landmark indices and connection definitions ---
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
            if landmarks_np[start_idx][2] > 0.1 and landmarks_np[end_idx][2] > 0.1:
                start_pos = (int(landmarks_np[start_idx][0]), int(landmarks_np[start_idx][1]))
                end_pos = (int(landmarks_np[end_idx][0]), int(landmarks_np[end_idx][1]))
                cv2.line(image, start_pos, end_pos, color, 2)
            
    for i, landmark in enumerate(landmarks_np):
        if landmark[2] > 0.1:
            pos = (int(landmark[0]), int(landmark[1]))
            cv2.circle(image, pos, 4, (0, 255, 0), -1)
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
        
        # <<< MODIFIED: Run the model with verbose=True to see its output >>>
        print(f"\n--- Analyzing Frame {frame_count} ---")
        results = model(frame, verbose=True)
        
        # Print the contents of the results for the first 5 frames for debugging
        if frame_count <= 5:
            print(f"DEBUG: Results object for frame {frame_count}: {results[0]}")
        
        black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        # The condition to check for keypoints has been made more robust
        if results and hasattr(results[0], 'keypoints') and results[0].keypoints is not None and results[0].keypoints.shape[1] > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            skeleton_frame = draw_polished_skeleton(keypoints, black_frame)
            video_writer.write(skeleton_frame)
        else:
            print(f"WARNING: No person detected in frame {frame_count}.")
            video_writer.write(black_frame)
            
    print(f"[Analyzer] Processed {frame_count} frames.")
    cap.release()
    video_writer.release()
    print(f"[Analyzer] Analysis complete. Final skeleton video saved to: {output_video_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    analyze_video(sys.argv[1], "skeleton_video_final.mp4")
