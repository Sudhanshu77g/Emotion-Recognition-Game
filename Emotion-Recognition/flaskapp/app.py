from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace
import random
import time

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

emotions_list = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
score = 0
target_emotion = random.choice(emotions_list)
emotion_change_time = time.time()
emotion_duration = 5 

def generate_frames():
    global target_emotion, score, emotion_change_time

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
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

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
