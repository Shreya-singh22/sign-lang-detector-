import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import logging
import os
import base64
from flask import Flask, render_template, jsonify, request, send_from_directory

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load the model
model_dict = pickle.load(open(os.path.join('models', 'final_model.p'), 'rb'))
model = model_dict['model']

# Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, max_num_hands=1)

# Labels dictionary (27 classes: a-z, nothing, space)
labels_dict = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f',
    6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l',
    12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r',
    18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x',
    24: 'y', 25: 'z', 26: 'nothing', 27: 'space'
}

# Variable to store the previous prediction and time
last_detected_character = None
fixed_character = ""
delayCounter = 0
start_time = time.time()

# For storing predictions to send to the front-end
predicted_text = ""

@app.route('/')
def index():
    return render_template('index.html')  # Serve the main index.html

@app.route('/camera')
def camera():
    return render_template('camera.html')  # Serve the camera page

@app.route('/ping')
def ping():
    return jsonify({'status': 'ok'})

@app.route('/dataset')
def dataset():
    test_dir = os.path.join(os.path.dirname(__file__), 'Test')
    folders = []
    total_images = 0
    if os.path.exists(test_dir):
        for folder_name in sorted(os.listdir(test_dir)):
            folder_path = os.path.join(test_dir, folder_name)
            if os.path.isdir(folder_path):
                images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                count = len(images)
                total_images += count
                folders.append({'name': folder_name, 'count': count, 'images': sorted(images)})
    return render_template('dataset.html', folders=folders, total_images=total_images)

@app.route('/dataset_image/<folder>/<filename>')
def dataset_image(folder, filename):
    test_dir = os.path.join(os.path.dirname(__file__), 'Test')
    return send_from_directory(os.path.join(test_dir, folder), filename)

@app.route('/get_prediction')
def get_prediction():
    return jsonify({"prediction": predicted_text})

@app.route('/clear_text', methods=['POST'])
def clear_text():
    global predicted_text
    predicted_text = ""  # Reset the predicted text
    return jsonify({"status": "success"})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global predicted_text, last_detected_character, fixed_character, delayCounter, start_time

    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        current_character = None
        hold_progress = 0.0

        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                x1 = max(0, int(min(x_) * W) - 15)
                y1 = max(0, int(min(y_) * H) - 15)
                x2 = min(W, int(max(x_) * W) + 15)
                y2 = min(H, int(max(y_) * H) + 15)

                prediction = model.predict([np.asarray(data_aux)])
                predicted_index = int(prediction[0])

                if predicted_index in labels_dict:
                    current_character = labels_dict[predicted_index]

                if current_character:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (74, 144, 226), 3)
                    label = current_character.upper()
                    cv2.putText(frame, label, (x1, y1 - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (74, 144, 226), 3, cv2.LINE_AA)

                    now = time.time()
                    if current_character == last_detected_character:
                        elapsed = now - start_time
                        hold_progress = min(elapsed / 1.0, 1.0)
                        if elapsed >= 1.0 and delayCounter == 0:
                            predicted_text += current_character
                            delayCounter = 1
                    else:
                        start_time = now
                        last_detected_character = current_character
                        delayCounter = 0
                        hold_progress = 0.0

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'image': 'data:image/jpeg;base64,' + encoded_image,
            'prediction': predicted_text,
            'current_character': current_character,
            'hold_progress': hold_progress
        })
    except Exception as e:
        logging.error("process_frame error: %s", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
