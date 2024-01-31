import cv2
import mediapipe as mp
import pyautogui as pygui
import math

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

print("HEY! it started!")

# Opens the webcam, input 1 is not always the webcam, if it is not, the project will not launch
# you might need to play with the value to find your webcam of choice
cap = cv2.VideoCapture(1)

# Define Distance calculation
def distance_3d(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

# Settings for hand detection
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.7) as hands:
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
                    pinky_tip = hand_landmarks.landmark[12]
                    hand_root = hand_landmarks.landmark[9]

                    dis_point1 = distance_3d((index_tip.x, index_tip.y, index_tip.z),
                                              (thumb_tip.x, thumb_tip.y, thumb_tip.z)) * 100
                    dis_point2 = distance_3d((index_tip.x, index_tip.y, index_tip.z),
                                              (middle_tip.x, middle_tip.y, middle_tip.z)) * 100
                    dis_point3 = distance_3d((middle_tip.x, middle_tip.y, middle_tip.z),
                                              (thumb_tip.x, thumb_tip.y, thumb_tip.z)) * 100

                    # Initial calculations
                    ix, iy = (hand_root.x * SCREEN_WIDTH), (hand_root.y * SCREEN_HEIGHT)

                    # Translation to center the coordinates around (0, 0)
                    # and apply mouse sensitivity
                    ix_NRM = (ix - SCREEN_WIDTH/2) * X_multi
                    iy_NRM = (SCREEN_HEIGHT/2 - iy) * Y_multi

                    # Translation back to original coordinate system
                    ix = ix_NRM + SCREEN_WIDTH/2
                    iy = SCREEN_HEIGHT/2 - iy_NRM

                    Mouse_pos = [ix, iy]
                    current_position = Mouse_pos
                    positions.append(current_position)

                    if len(positions) > NUM_POSITIONS:
                        positions.pop(0)

                    avg_x = sum(pos[0] for pos in positions) / len(positions)
                    avg_y = sum(pos[1] for pos in positions) / len(positions)

                    print(ix, iy)

                    if dis_point1 < 4.5 and dis_point2 > 10:

                        if Mouse_state == 1:
                            Mouse_state = 1
                            pygui.moveTo(avg_x, avg_y, _pause=False)
                        else:
                            pygui.leftClick(_pause=False)
                            Mouse_state = 1
                            pygui.moveTo(avg_x, avg_y, _pause=False)
                    elif dis_point3 < 4.5 and dis_point1 > 1:
                        pygui.rightClick(_pause=False)
                        Mouse_state = 2
                        pygui.moveTo(avg_x, avg_y, _pause=False)
                    else:
                        Mouse_state = 0
                        pygui.moveTo(avg_x, avg_y, _pause=False)

                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(252, 123, 43), thickness=2,
                                                                     circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(58, 145, 89), thickness=2,
                                                                     circle_radius=10))
                    text_screen = int(avg_x), int(avg_y)
                    text_screen = (str(text_screen) + "  " + str(Mouse_state))
                    Text_pos = (int(hand_root.x * SCREEN_WIDTH), int(hand_root.y * SCREEN_HEIGHT))
                    cv2.putText(image, text_screen, Text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)

        cv2.imshow("Evan's Fabulous Hand Tracking For Mac", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
