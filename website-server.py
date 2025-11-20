import pickle
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import logging
import os
from flask import Flask, render_template, jsonify, Response

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

@app.route('/get_prediction')
def get_prediction():
    return jsonify({"prediction": predicted_text})

@app.route('/clear_text', methods=['POST'])
def clear_text():
    global predicted_text
    predicted_text = ""  # Reset the predicted text
    return jsonify({"status": "success"})

def gen_frames():
    global predicted_text, last_detected_character, fixed_character, delayCounter, start_time

    cap = cv2.VideoCapture(0)

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Make prediction using the model
                prediction = model.predict([np.asarray(data_aux)])
                predicted_index = int(prediction[0])

                # Filter out invalid predictions
                if predicted_index in labels_dict:
                    predicted_character = labels_dict[predicted_index]
                else:
                    predicted_character = None  # Ignore invalid predictions

                # Draw a rectangle and the predicted character on the frame
                if predicted_character:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)

                    current_time = time.time()

                    # Timer logic: Check if the predicted character is the same for more than 1 second
                    if predicted_character == last_detected_character:
                        if (current_time - start_time) >= 1.0:  # Class fixed after 1 second
                            fixed_character = predicted_character
                            if delayCounter == 0:  # Add character once after it stabilizes for 1 second
                                predicted_text += fixed_character
                                delayCounter = 1
                    else:
                        # Reset the timer when a new character is detected
                        start_time = current_time
                        last_detected_character = predicted_character
                        delayCounter = 0  # Reset delay counter for a new character

        # Encode frame as JPEG and yield as MJPEG stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start the camera in a separate thread
threading.Thread(target=gen_frames, daemon=True).start()

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
