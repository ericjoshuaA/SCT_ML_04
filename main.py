import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# --- 1. Load the trained model and labels ---
try:
    model = load_model('models/gesture_model.h5')
    with open('models/gesture_labels.txt', 'r') as f:
        gesture_labels = [line.strip() for line in f.readlines()]
    print("Model and labels loaded successfully.")
except (IOError, OSError) as e:
    print(f"Error loading model or labels: {e}")
    print("Please make sure you have run 'model_trainer.py' first.")
    exit()

# --- 2. Setup Mediapipe and Webcam ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    
    recognized_gesture = "No Gesture"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))
            
            # --- 3. Predict the gesture using the trained model ---
            flat_landmarks = []
            for landmark in hand_landmarks.landmark:
                flat_landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            input_data = np.array(flat_landmarks).reshape(1, -1)
            
            # Make the prediction
            try:
                prediction = model.predict(input_data)
                predicted_class_index = np.argmax(prediction)
                recognized_gesture = gesture_labels[predicted_class_index]
            except Exception as e:
                # This could happen if the model's input shape doesn't match
                # the number of landmarks detected.
                recognized_gesture = f"Error: {e}"
            
            cv2.putText(frame, recognized_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
