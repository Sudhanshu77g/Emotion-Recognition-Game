import cv2
from deepface import DeepFace
import random
import time


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)


emotions_list = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
score = 0
target_emotion = random.choice(emotions_list)
emotion_change_time = time.time()
emotion_duration = 5 

while True:
   
    ret, frame = cap.read()
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
   
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    detected_emotion = None

    for (x, y, w, h) in faces:
 
        face_roi = frame[y:y + h, x:x + w]

        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        detected_emotion = result[0]['dominant_emotion']

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f'Detected: {detected_emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)  # Black text

   
    if time.time() - emotion_change_time > emotion_duration:
        target_emotion = random.choice(emotions_list)
        emotion_change_time = time.time()  # Reset timer

    
    cv2.putText(frame, f'Guess: {target_emotion}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Black text

   
    if detected_emotion is not None:
        if detected_emotion == target_emotion:
            score += 1
            cv2.putText(frame, 'Correct!', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green for correct
        else:
            cv2.putText(frame, 'Try Again!', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red for try again

    
    cv2.putText(frame, f'Score: {score}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Black text

   
    cv2.imshow('Gamified Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
