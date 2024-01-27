import cv2
import csv
import mediapipe as mp
import numpy as np
from collections import deque
import os

mp_draw = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
num_coords = 21 * 2

landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]


# Inisialisasi variabel
cvFpsCalc = cv2.TickMeter()

history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)

cap = cv2.VideoCapture(0)


file_path = 'datahuruf.csv'

class_label = input("Masukkan kelas: ")

recorded_data = 0

# Definisi landmarks jika belum ada
if not os.path.isfile(file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(landmarks)

# Membuat variabel untuk menandai sudah mencatat data atau belum
data_recorded = False

with open(file_path, mode='a', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
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
                mp_draw.draw_landmarks(image_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(65, 145, 151), thickness=2, circle_radius=4),
                                       mp_draw.DrawingSpec(color=(245, 252, 205), thickness=2, circle_radius=2))

            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(image_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(65, 145, 151), thickness=2, circle_radius=4),
                                       mp_draw.DrawingSpec(color=(245, 252, 205), thickness=2, circle_radius=2))

            try:
                right_hand = results.right_hand_landmarks.landmark
                left_hand = results.left_hand_landmarks.landmark
                
                right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for
                                                landmark in right_hand]).flatten())
                left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for
                                               landmark in left_hand]).flatten())

                row = right_hand_row + left_hand_row
                row.insert(0, class_label)
                
                writer.writerow(row)
                recorded_data += 1 

                if recorded_data >= 120:
                    print("Telah mencapai 120 data. Perekaman dihentikan.")
                    cap.release()  # Memberhentikan perekaman dengan melepaskan kamera
                    cv2.destroyAllWindows()  # Menutup jendela
                    break
            
            except:
                pass

            cv2.imshow('Kamera', image_bgr)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
