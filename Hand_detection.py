import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Creating a black canvas
canvas = np.zeros((480, 640, 3), dtype="uint8")

# Opening the webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip to avoid lateral inversion

    # Converting image to RGB for mediapipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Getting index finger tip coordinates 
            index_finger_tip = hand_landmarks.landmark[8]
            h, w, c = img.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Drawing on the canvas with finger
            cv2.circle(canvas, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    # Including the real time vedio and canvas
    img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

    # Display 
    cv2.imshow("Air Canvas", img)

    # Exit code
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
