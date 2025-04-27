import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
from sklearn.preprocessing import StandardScaler
import random


def augment_image(img):
    """Apply random augmentations like flipping and rotation."""
    if random.choice([True, False]):
        img = cv2.flip(img, 1)  # Horizontal flip
    angle = random.randint(-15, 15)
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img


def extract_features(img):
    """Extract improved color + texture features."""
    img = cv2.resize(img, (100, 100))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color histogram (HSV space)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Texture feature (Sobel edge magnitude)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    texture_hist = np.histogram(magnitude, bins=8, range=(0, 255))[0]
    texture_hist = texture_hist / np.sum(texture_hist)  # normalize

    # Combine color and texture
    features = np.hstack([hist, texture_hist])
    return features


data = []
labels = []

base_path = os.path.join(os.path.dirname(__file__), "wall_dataset")

for label, folder in enumerate(['walls', 'non_walls']):
    folder_path = os.path.join(base_path, folder)
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        continue
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(path)

                # 1 original + 2 augmented copies
                for _ in range(3):
                    if _ > 0:
                        img_aug = augment_image(img.copy())
                    else:
                        img_aug = img.copy()

                    features = extract_features(img_aug)
                    data.append(features)
                    labels.append(label)
            except Exception as e:
                print(f"Error processing {path}: {e}")

# Normalize features
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    labels,
                                                    test_size=0.2,
                                                    random_state=42)

clf = SVC(kernel='linear', probability=True, class_weight='balanced')
clf.fit(X_train, y_train)

# Evaluate
print("\nModel evaluation:\n")
print(classification_report(y_test, clf.predict(X_test)))

# Save model and scaler
dump((clf, scaler), 'wall_detector_model.joblib')
print("\nModel and scaler saved to wall_detector_model.joblib")
