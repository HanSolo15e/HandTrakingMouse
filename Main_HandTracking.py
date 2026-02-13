import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui as pygui
import math
import numpy as np
import time

# --- INITIALIZATION & CONFIG ---
SCREEN_WIDTH, SCREEN_HEIGHT = pygui.size()
NUM_POSITIONS = 10  
Mouse_state = 0
X_multi = 1.5
Y_multi = 1.5
Num_of_hands = 1

# Define Hand Connections manually (since mp.solutions is missing)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Pinky/Palm
]

# 1. Setup MediaPipe Tasks Detector
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=Num_of_hands,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

print("HEY! it started!")

# 2. Camera Setup (macOS specific)
is_macos = True 
if is_macos:
    # Use +0 if +1 doesn't work; CAP_AVFOUNDATION is best for Macs
    cap = cv2.VideoCapture(cv2.CAP_AVFOUNDATION + 0)
else:
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Helper for 3D distance
def distance_3d(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

positions = []

# --- MAIN LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Mirror and Resize
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape # Original camera resolution
    
    # MediaPipe requires RGB and a specific Image object
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detection (requires a millisecond timestamp)
    frame_timestamp_ms = int(time.time() * 1000)
    results = detector.detect_for_video(mp_image, frame_timestamp_ms)

    # Process Results
    if results.hand_landmarks:
        # We only care about the first hand detected
        hand_landmarks = results.hand_landmarks[0]
        
        # Grab specific tips for mouse logic
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        middle_tip = hand_landmarks[12]
        hand_root = hand_landmarks[9] # Using landmark 9 (middle finger root) for tracking

        # Logic Distances
        dis_point1 = distance_3d(index_tip, thumb_tip) * 100
        dis_point2 = distance_3d(index_tip, middle_tip) * 100
        dis_point3 = distance_3d(middle_tip, thumb_tip) * 100

        # Screen Space Translation with Sensitivity
        Hand_Root_Scr_x = hand_root.x * SCREEN_WIDTH
        Hand_Root_Scr_y = hand_root.y * SCREEN_HEIGHT

        Hand_Root_Scr_xNRM = (Hand_Root_Scr_x - SCREEN_WIDTH/2) * X_multi
        Hand_Root_Scr_yNRM = (SCREEN_HEIGHT/2 - Hand_Root_Scr_y) * Y_multi

        raw_x = Hand_Root_Scr_xNRM + SCREEN_WIDTH/2
        raw_y = SCREEN_HEIGHT/2 - Hand_Root_Scr_yNRM

        # Smoothing Logic
        positions.append((raw_x, raw_y))
        if len(positions) > NUM_POSITIONS:
            positions.pop(0)
        
        avg_x, avg_y = np.mean(positions, axis=0)

        # Click detection & Cursor Movement
        if dis_point1 < 4.5 and dis_point2 > 10: # Left Click (Index + Thumb)
            if Mouse_state != 1:
                pygui.leftClick(_pause=False)
                Mouse_state = 1
        elif dis_point3 < 4.5 and dis_point1 > 5: # Right Click (Middle + Thumb)
            if Mouse_state != 2:
                pygui.rightClick(_pause=False)
                Mouse_state = 2
        else:
            Mouse_state = 0
        
        pygui.moveTo(avg_x, avg_y, _pause=False)

        # --- DRAWING (Manual OpenCV fallback for drawing_utils) ---
        # Draw connections (Skeleton)
        for connection in HAND_CONNECTIONS:
            p1 = hand_landmarks[connection[0]]
            p2 = hand_landmarks[connection[1]]
            cv2.line(frame, (int(p1.x * w), int(p1.y * h)), 
                     (int(p2.x * w), int(p2.y * h)), (58, 145, 89), 2)

        # Draw dots (Landmarks)
        for lm in hand_landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (252, 123, 43), -1)

        # Visual Feedback Text
        status_text = f"Pos: {int(avg_x)}, {int(avg_y)} | State: {Mouse_state}"
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Evan's Fabulous Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()