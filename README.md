# Sign Language Detector

Real-time American Sign Language (ASL) detection in the browser — show a hand sign to your webcam and the model identifies the letter, live.

## 🔗 [sign-lang-ai.onrender.com](https://sign-lang-ai.onrender.com)

---

## How it works

1. The browser captures webcam frames and sends them to the Flask backend as base64 JPEG.
2. [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands) extracts 21 hand landmarks from each frame.
3. A Random Forest classifier (trained on the [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)) maps the landmark coordinates to one of 28 classes (A–Z, Nothing, Space).
4. The annotated frame (with landmarks and bounding box drawn) is returned and displayed in real-time.
5. Hold a sign still for **1 second** to confirm the letter — it gets added to the predicted text.

---

## Pages

| Route | Description |
|---|---|
| `/` | Landing page |
| `/camera` | Live webcam detection |
| `/dataset` | Dataset overview — browse all 28 classes with sample images |

---

## Tech stack

| Layer | Tools |
|---|---|
| Backend | Python, Flask, Gunicorn |
| Hand tracking | MediaPipe Hands |
| ML model | Scikit-learn Random Forest |
| Image processing | OpenCV (`opencv-python-headless`) |
| Frontend | HTML/CSS/JS, Bootstrap 5 |
| Deployment | [Render](https://render.com) |

---

## Dataset

**[ASL Alphabet — Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)** by Akash (grassknoted)

- 87,000 images across 29 classes (A–Z + Delete, Nothing, Space)
- 200 × 200 px, diverse lighting and hand positions
- 3,000 training images per class

The model uses MediaPipe landmarks instead of raw pixels — each sample is 42 normalised (x, y) coordinates — making it fast and lighting-invariant.

---

## Run locally

```bash
# Clone
git clone https://github.com/Shreya-singh22/sign-lang-detector-.git
cd sign-lang-detector-

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

Open **http://localhost:5001** in your browser and allow camera access.

> Camera access requires **HTTPS or localhost**. On a remote server, put the app behind an HTTPS proxy.

---

## Project structure

```
sign-lang-detector-/
├── app.py                  # Flask app — routes + MediaPipe + model inference
├── requirements.txt
├── Procfile                # Gunicorn entry point for Render
├── models/
│   └── final_model.p       # Trained Random Forest (pickle)
├── templates/
│   ├── index.html          # Landing page
│   ├── camera.html         # Live detection UI
│   └── dataset.html        # Dataset browser
├── static/
│   └── styles.css
└── Test/                   # Sample images (A–Z, Nothing, Space)
    ├── A/
    ├── B/
    └── ...
```

---

## Tips for best results

- Good lighting on your hand
- Keep your hand fully within the frame
- Plain or uncluttered background
- Hold each sign still for 1 second to confirm the letter
