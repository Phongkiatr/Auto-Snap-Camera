from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import threading
import os
import time

# ฟังก์ชันสำหรับเตรียมภาพ
def prepare_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # แปลงเป็น Grayscale
    img = cv2.resize(img, (48, 48))  # ปรับขนาดภาพให้ตรงกับโมเดล
    img = np.expand_dims(img, axis=-1)  # เพิ่มมิติของสี (1 ช่องสำหรับ grayscale)
    img = np.expand_dims(img, axis=0)  # เพิ่มมิติ batch ให้เป็น (1, 48, 48, 1)
    img = img / 255.0  # Normalization
    return img

# โหลดโมเดลที่เคยเทรนด์ไว้
model = tf.keras.models.load_model('model/CK_model.h5', compile=False)

# แปลงหมายเลขคลาสเป็นชื่ออารมณ์
emotion_labels = {
    0: 'angry',    
    1: 'contempt',   
    2: 'disgust',     
    3: 'fear',     
    4: 'happy',   
    5: 'sad',       
    6: 'surprise'   
}

# ฟังก์ชันทำนายอารมณ์
def predict_emotion(face):
    img = prepare_image(face)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    predicted_emotion = emotion_labels[predicted_class]
    return predicted_emotion

# โหลด CascadeClassifier ข้างนอก loop
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ฟังก์ชันสำหรับการจับภาพและทำนายอารมณ์
def capture_and_predict(cap, stop_event):
    if not os.path.exists('saved_images'):
        os.makedirs('saved_images')

    image_counter = 0
    last_predicted_emotion = None
    countdown = 0
    countdown_start_time = 0
    frame_counter = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip in horizontal
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        if frame_counter % 2 == 0:
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                predicted_emotion = predict_emotion(face)

                if countdown <= 0:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                if predicted_emotion == 'happy' and last_predicted_emotion != 'happy':
                    countdown = 3
                    countdown_start_time = time.time()

                if countdown > 0:
                    elapsed_time = time.time() - countdown_start_time
                    remaining_time = int(countdown - elapsed_time)

                    if remaining_time > 0:
                        cv2.putText(frame, f'Capturing in {remaining_time}...', (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    elif remaining_time <= 0:
                        image_counter += 1
                        image_filename = f'saved_images/happy_{image_counter}.jpg'
                        cv2.imwrite(image_filename, frame)
                        print(f'Image saved as {image_filename}')
                        countdown = -1

                last_predicted_emotion = predicted_emotion

        frame_counter += 1
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# หน้าเว็บหลัก
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# เส้นทางสำหรับการส่งภาพจากกล้อง
@app.route('/video')
def video():
    stop_event = threading.Event()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    capture_thread = threading.Thread(target=capture_and_predict, args=(cap, stop_event))
    capture_thread.start()

    return Response(capture_and_predict(cap, stop_event), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)