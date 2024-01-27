import cv2
import numpy as np
import mediapipe as mp
import pickle
import pandas as pd

# Inisialisasi model Random Forest dari file PKL
with open('modelhurufbisindo.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = holistic.process(image_rgb)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        image_bgr.flags.writeable = True

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(65, 145, 151), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245, 252, 205), thickness=2, circle_radius=2))

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(65, 145, 151), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245, 252, 205), thickness=2, circle_radius=2))

        try:
            right_hand = results.right_hand_landmarks.landmark
            left_hand = results.left_hand_landmarks.landmark

            right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for
                                           landmark in right_hand]).flatten())
            left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for
                                          landmark in left_hand]).flatten())

            row = right_hand_row + left_hand_row

            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]  # Prediksi label
            body_language_prob = model.predict_proba(X)[0]
            
            coords = (30, 50)  # Koordinat untuk menampilkan tag

            # Membuat kotak untuk tulisan prediksi
            cv2.rectangle(image_bgr, (coords[0], coords[1] + 5), 
                          (coords[0] + len(f'Prediksi: {body_language_class}') * 20, coords[1] - 30),
                          (0, 255, 0), -1)
            
            # Menambahkan teks prediksi ke frame
            cv2.putText(image_bgr, f'Prediksi: {body_language_class}', coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        except:
            pass

        cv2.imshow('Kamera', image_bgr)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

