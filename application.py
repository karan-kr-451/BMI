from flask import Flask, render_template, Response
import cv2
import pymongo
from src.pipelines.prediction_pipeline import PredictPipeline


app = Flask(__name__)
camera = cv2.VideoCapture(0)
pipeline = PredictPipeline()
client = pymongo.MongoClient("mongodb+srv://karan:Tharki@cluster0.gfvk9lg.mongodb.net/?retryWrites=true&w=majority")
db = client['Face_Data']
user_collection = db['users']


def capture_frame():
    success, frame = camera.read()
    if not success:
        return None
    else:
        return frame

def detect_face(frame):
    detector = cv2.CascadeClassifier('Haarcascade/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 7)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        return face_roi
    return None

@app.route('/')
def index():
    return render_template('home.html')

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            face_roi = detect_face(frame)
            if face_roi is not None:
                # Make prediction on the face ROI
                pred = pipeline.predict(face_roi)
                # Draw prediction on the frame
                cv2.putText(frame, f'Prediction: {pred}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_image', methods=['POST'])
def capture_image():
    frame = capture_frame()
    if frame is not None:
        face_roi = detect_face(frame)
        if face_roi is not None:
            cv2.imwrite('captured_face.jpg', face_roi)
            pred = pipeline.predict(face_roi)[0][0]
            pred = pred.item()
            print(pred)

            user_data = {
                'face_data' : 'captured_face.jpg',
                'bmi_prediction': pred
            }
            user_collection.insert_one(user_data)

            return {'prediction': str(pred)}
        else:
            return {'error': 'No face detected'}
    else:
        return {'error': 'Failed to capture image'}

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=True)
