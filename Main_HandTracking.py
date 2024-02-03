import cv2
import mediapipe as mp
import pyautogui as pygui
import math
import numpy as np

# Initialize Mediapipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = pygui.size()

# Constants
NUM_POSITIONS = 10  
Mouse_state = 0

# Mouse Sensitivity
X_multi = 1.5
Y_multi = 1.5

# set how manny hands it can detect at one time
Num_of_hands = 1

print("HEY! it started!")

# Opens the webcam, input 1 is not always the webcam, if it is not, the project will not launch
# you might need to play with the value to find your webcam of choice
is_macos = True  # Change this based on your platform detection logic
if is_macos:
    # For macOS, use AVFoundation for video capture
    cap = cv2.VideoCapture(cv2.CAP_AVFOUNDATION + 1)
else:
    # For other platforms like Windows, use the default capture device
    cap = cv2.VideoCapture(1)


# Define Distance calculation
def distance_3d(point1, point2):
    point1_np = np.array(point1)
    point2_np = np.array(point2)
    return np.linalg.norm(point2_np - point1_np)

def distance_2d(point1, point2):
    point1_np = np.array(point1)
    point2_np = np.array(point2)
    return math.dist(point2_np, point1_np)

# Settings for hand detection
with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=Num_of_hands) as hands:
    positions = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (SCREEN_WIDTH, SCREEN_HEIGHT))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if len(hand_landmarks.landmark) >= 21:

                    index_tip = hand_landmarks.landmark[8]
                    thumb_tip = hand_landmarks.landmark[4]
                    middle_tip = hand_landmarks.landmark[12]
                    hand_root = hand_landmarks.landmark[9]
                    hand_root_2 = hand_landmarks.landmark[13]

                    dis_point1 = distance_3d((index_tip.x, index_tip.y, index_tip.z),
                                              (thumb_tip.x, thumb_tip.y, thumb_tip.z)) * 100
                    dis_point2 = distance_3d((index_tip.x, index_tip.y, index_tip.z),
                                              (middle_tip.x, middle_tip.y, middle_tip.z)) * 100
                    dis_point3 = distance_3d((middle_tip.x, middle_tip.y, middle_tip.z),
                                              (thumb_tip.x, thumb_tip.y, thumb_tip.z)) * 100
                    

                    # Convert to screen space
                    Hand_Root_Scr_x, Hand_Root_Scr_y = (hand_root.x * SCREEN_WIDTH), (hand_root.y * SCREEN_HEIGHT)
                    ##Hand_Root2_Scr_x, Hand_Root2_Scr_y2= (hand_root_2.x * SCREEN_WIDTH), (hand_root_2.y * SCREEN_HEIGHT)

                    ##dis_point_2D = distance_2d((Hand_Root_Scr_x, Hand_Root_Scr_y),(Hand_Root2_Scr_x, Hand_Root2_Scr_y2))

                    ##print(dis_point_2D)

                    # check if hand is not pointing at camera 
                    ##if dis_point_2D < 20:
                        

                    # Translation to center the coordinates around (0, 0)
                    # and apply mouse sensitivity
                    Hand_Root_Scr_xNRM = (Hand_Root_Scr_x - SCREEN_WIDTH/2) * X_multi
                    Hand_Root_Scr_yNRM = (SCREEN_HEIGHT/2 - Hand_Root_Scr_y) * Y_multi

                    # Translation back to original coordinate system
                    Hand_Root_Scr_xMlt = Hand_Root_Scr_xNRM + SCREEN_WIDTH/2
                    Hand_Root_Scr_yMlt = SCREEN_HEIGHT/2 - Hand_Root_Scr_yNRM

                    # Makes mouse movement smooth
                    Mouse_pos = [Hand_Root_Scr_xMlt, Hand_Root_Scr_yMlt]
                    current_position = Mouse_pos
                    positions.append(current_position)

                    if len(positions) > NUM_POSITIONS:
                        positions.pop(0)

                    avg_x, avg_y = np.mean(positions, axis=0)

                    ##print(ix, iy)

                    # Click detection
                    if dis_point1 < 4.5 and dis_point2 > 10:
                        
                        if Mouse_state == 1:
                            Mouse_state = 1
                            pygui.moveTo(avg_x, avg_y, _pause=False)
                        else:
                            pygui.leftClick(_pause=False)
                            Mouse_state = 1
                            pygui.moveTo(avg_x, avg_y, _pause=False)
                    elif dis_point3 < 4.5 and dis_point1 > 5:
                        
                        if Mouse_state == 2:
                            Mouse_state = 2
                            pygui.moveTo(avg_x, avg_y, _pause=False)
                        else:
                            pygui.rightClick(_pause=False)
                            Mouse_state = 2
                            pygui.moveTo(avg_x, avg_y, _pause=False)
                    else:
                        Mouse_state = 0
                        
                        pygui.moveTo(avg_x, avg_y, _pause=False)

                        # draws hand bones / render text
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(252, 123, 43), thickness=2,
                                                                     circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(58, 145, 89), thickness=2,
                                                                     circle_radius=10))
                    text_screen = int(avg_x), int(avg_y)
                    text_screen = (str(text_screen) + "  " + str(Mouse_state))
                    Text_pos = (int(Hand_Root_Scr_x), int(Hand_Root_Scr_y))
                    cv2.putText(image, text_screen, Text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)

        cv2.imshow("Evan's Fabulous Hand Tracking", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
