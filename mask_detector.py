import os
import cv2
import numpy as np

try:
    from keras.models import load_model
except ImportError:
    from tensorflow.keras.models import load_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'mask_detector.keras')

# If model not found locally, check the Documents/Face_Mask_Detection directory
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.expanduser('~'), 'Documents', 'Face_Mask_Detection', 'mask_detector.keras')

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file not found. Please place 'mask_detector.keras' in:\n  {SCRIPT_DIR}")
    exit(1)

cnn = load_model(MODEL_PATH)
print("[INFO] Mask detection model loaded successfully.")

# --- Configuration ---
labels_dict = {0: 'Mask', 1: 'No Mask'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}  # Green=Mask, Red=No Mask
IMG_SIZE = 224
imgsize = 4  # Downscale factor for faster face detection

# --- Open camera ---
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("[ERROR] Could not open camera.")
    exit(1)

# --- Load Haar Cascade face detector ---
classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

print("[INFO] Starting detection. Press ESC to quit.")

while True:
    (rval, im) = camera.read()
    
    if not rval:
        print("[ERROR] Failed to read from camera.")
        break

    im = cv2.flip(im, 1, 1)  # Mirror the image

    imgs = cv2.resize(im, (im.shape[1] // imgsize, im.shape[0] // imgsize))
    face_rec = classifier.detectMultiScale(imgs)

    for i in face_rec:
        (x, y, l, w) = [v * imgsize for v in i]
        
        if y < 0 or x < 0 or y + w > im.shape[0] or x + l > im.shape[1]:
            continue

        face_img = im[y:y + w, x:x + l]

        if face_img.size == 0:
            continue
        
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
       
        resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, IMG_SIZE, IMG_SIZE, 3))
        reshaped = np.vstack([reshaped])

        result = cnn.predict(reshaped, verbose=0)
        label = np.argmax(result, axis=1)[0]
        confidence = float(np.max(result))

        cv2.rectangle(im, (x, y), (x + l, y + w), color_dict[label], 2)
        cv2.rectangle(im, (x, y - 40), (x + l, y), color_dict[label], -1)
        cv2.putText(
            im,
            f"{labels_dict[label]}: {confidence * 100:.1f}%",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    cv2.imshow('LIVE', im)
    key = cv2.waitKey(10)

    # Stop loop by ESC
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()

