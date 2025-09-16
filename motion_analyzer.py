import sys, os, subprocess, math, collections
import cv2
import numpy as np
import mediapipe as mp

L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 23, 24, 25, 26, 27, 28
LEFT_CONNECTIONS = [(L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE)]
RIGHT_CONNECTIONS = [(R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE)]
CENTER_CONNECTIONS = [(L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP), (L_SHOULDER, R_HIP), (R_SHOULDER, L_HIP)]

def draw_polished_skeleton(landmarks, image):
    height, width, _ = image.shape
    landmarks_np = np.array(landmarks)
    color_left = (255, 100, 0); color_right = (0, 100, 255); color_center = (200, 200, 200)
    connection_groups = [(LEFT_CONNECTIONS, color_left), (RIGHT_CONNECTIONS, color_right), (CENTER_CONNECTIONS, color_center)]
    for connections, color in connection_groups:
        for connection in connections:
            start_idx, end_idx = connection
            if landmarks_np[start_idx][3] > 0.5 and landmarks_np[end_idx][3] > 0.5:
                start_z = landmarks_np[start_idx][2]
                brightness = max(0.3, 1 - (start_z + 0.5)); thickness = int(max(1, 4 * brightness))
                dynamic_color = tuple(int(c * brightness) for c in color)
                start_pos = (int(landmarks_np[start_idx][0] * width), int(landmarks_np[start_idx][1] * height))
                end_pos = (int(landmarks_np[end_idx][0] * width), int(landmarks_np[end_idx][1] * height))
                cv2.line(image, start_pos, end_pos, dynamic_color, thickness)
    for i, landmark in enumerate(landmarks_np):
        if landmark[3] > 0.5:
            pos = (int(landmark[0] * width), int(landmark[1] * height)); radius = 3 if i < 11 else 4
            joint_color = color_center
            if i in [11,13,15,17,19,21,23,25,27,29,31]: joint_color = color_left
            elif i in [12,14,16,18,20,22,24,26,28,30,32]: joint_color = color_right
            cv2.circle(image, pos, radius, joint_color, -1)
    return image

def analyze_video(video_path, output_video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError("Cannot open video file")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))
    smoother = collections.deque(maxlen=3)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); result = pose.process(image_rgb)
        black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        if result.pose_landmarks:
            landmarks_raw = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark])
            smoother.append(landmarks_raw)
            landmarks_smoothed = np.mean(smoother, axis=0).tolist()
            skeleton_frame = draw_polished_skeleton(landmarks_smoothed, black_frame)
            video_writer.write(skeleton_frame)
        else:
            video_writer.write(black_frame)
    cap.release(); video_writer.release()

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    analyze_video(sys.argv[1], "skeleton_video_final.mp4")
