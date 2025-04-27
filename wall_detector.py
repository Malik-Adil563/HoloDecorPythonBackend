from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import train_wall_model
from joblib import load

# Load the trained model and scaler
model, scaler = load("wall_detector_model.joblib")

app = Flask(__name__)


def extract_features_from_bytes(image_bytes):
    np_arr = np.frombuffer(base64.b64decode(image_bytes), np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (100, 100))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Texture (edge magnitude)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    texture_hist = np.histogram(magnitude, bins=8, range=(0, 255))[0]
    texture_hist = texture_hist / np.sum(texture_hist)  # normalize

    features = np.hstack([hist, texture_hist])
    return features


@app.route('/detect-wall', methods=['POST'])
def detect_wall():
    try:
        img_data = request.get_json()['image']
        features = extract_features_from_bytes(img_data)

        # Scale the features using the loaded scaler
        features = scaler.transform([features])

        prediction = model.predict(features)[0]

        # Log prediction
        if prediction:
            print("Wall detected: True")
        else:
            print("Wall detected: False")

        return jsonify({'wallDetected': bool(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
