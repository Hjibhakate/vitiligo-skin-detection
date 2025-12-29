import cv2
import os
import numpy as np
import requests
from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input

# ✅ Class labels
CLASS_LABELS = ['Normal Skin', 'Vitiligo Detected']

# ✅ Build model matching saved weights
def build_model(input_shape=(528, 528, 3), num_classes=2):
    base_model = EfficientNetB6(include_top=False, weights=None, input_tensor=Input(shape=input_shape))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=output)

# ✅ Predict ROI
def predict_vitiligo_roi(roi_image, model):
    resized = cv2.resize(roi_image, (528, 528)) / 255.0
    img_array = np.expand_dims(resized, axis=0)
    prediction = model.predict(img_array)[0]

    predicted_index = np.argmax(prediction)
    predicted_class = CLASS_LABELS[predicted_index]
    confidence = prediction[predicted_index]

    print(f"Prediction: {predicted_class}")
    print(f"Accuracy: {confidence * 100:.2f}%")

# ✅ IP webcam + ROI capture
def predict_with_manual_roi():
    ip_url = 'http://192.0.0.4:8080/shot.jpg'
    model_path = "efficientnetb6-vitiligo.h5"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    print("[INFO] Loading model weights...")
    model = build_model()
    model.load_weights(model_path)
    print("✅ Model loaded successfully.")

    while True:
        try:
            response = requests.get(ip_url, timeout=5)
            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, -1)

            if frame is None:
                continue

            frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)
            cv2.imshow("📷 IP Webcam - Press SPACE to Capture", frame)

            key = cv2.waitKey(1)

            if key == 27:  # ESC
                cv2.destroyAllWindows()
                break

            elif key == 32:  # SPACE
                cv2.destroyAllWindows()
                roi = cv2.selectROI("🖱️ Select ROI", frame, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("🖱️ Select ROI")

                if roi == (0, 0, 0, 0):
                    print("⚠️ No region selected.")
                    break

                x, y, w, h = roi
                cropped = frame[y:y+h, x:x+w]
                predict_vitiligo_roi(cropped, model)
                break

        except Exception as e:
            print(f"⚠️ Error: {str(e)}")
            break

# ✅ Run main
if __name__ == "__main__":
    predict_with_manual_roi()
