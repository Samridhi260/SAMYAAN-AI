from flask import Flask, render_template, Response, request, redirect, url_for, flash
import os, cv2, torch, pickle
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC

app = Flask(__name__)
app.secret_key = "secret"

UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'piyush_model ( 4 july ).pkl'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['recognizer'], model_data['label_encoder']

recognizer, le = load_model()

mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, post_process=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
normalizer = Normalizer(norm='l2')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/live')
def live():
    return render_template('live.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = mtcnn.detect(img)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]

                try:
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).resize((160, 160))
                    face_tensor = torch.tensor(np.array(face_pil).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                    embedding = resnet(face_tensor).detach().numpy()
                    embedding = normalizer.transform(embedding)
                    probs = recognizer.predict_proba(embedding)[0]
                    j = np.argmax(probs)

                    if probs[j] >= 0.12:
                        name = le.classes_[j]
                    else:
                        name = "Unknown"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except Exception as e:
                    continue

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files or request.files['image'].filename == '':
        flash('No file uploaded.')
        return redirect(url_for('index'))

    file = request.files['image']
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    img = Image.open(img_path).convert("RGB")
    boxes, _ = mtcnn.detect(img)

    name = "No face found"
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face_crop = img.crop((x1, y1, x2, y2)).resize((160, 160))
            face_tensor = torch.tensor(np.array(face_crop).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            embedding = resnet(face_tensor).detach().numpy()
            embedding = normalizer.transform(embedding)
            probs = recognizer.predict_proba(embedding)[0]
            j = np.argmax(probs)
            name = le.classes_[j]

            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv2, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            break

    result_filename = f"result_{file.filename}"
    result_path = os.path.join(UPLOAD_FOLDER, result_filename)
    cv2.imwrite(result_path, img_cv2)

    return render_template("index.html", result_img=f"uploads/{result_filename}", name=name)


if __name__ == '__main__':
    app.run(debug=True)
