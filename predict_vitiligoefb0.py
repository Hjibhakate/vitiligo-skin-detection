import cv2
import os
import numpy as np
import requests
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras.models import load_model

# ✅ Class labels
CLASS_LABELS = ['Normal Skin', 'Vitiligo Detected']

# ✅ Build EfficientNetB0 model architecture
def build_model():
    input_tensor = Input(shape=(224, 224, 3))
    base_model = EfficientNetB0(include_top=False, weights=None, input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

# ✅ Predict on image ROI with EfficientNet preprocessing
def predict_vitiligo_roi(roi_image, model, threshold=0.60):
    resized = cv2.resize(roi_image, (224, 224))
    img_array = np.expand_dims(resized, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array, verbose=0)[0][0]
    print(f"\nRaw sigmoid score: {prediction:.4f}")

    if prediction >= threshold:
        predicted_class = CLASS_LABELS[1]
        confidence = prediction
        color = (0, 0, 255)  # Red
    else:
        predicted_class = CLASS_LABELS[0]
        confidence = 1 - prediction
        color = (0, 255, 0)  # Green

    print(f" Prediction: {predicted_class}")
    print(f" Confidence: {confidence * 100:.2f}%")

    display_img = cv2.resize(roi_image, (300, 300))
    label_text = f"{predicted_class} ({confidence * 100:.2f}%)"
    cv2.putText(display_img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("🔍 Prediction Result", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ✅ Load image from IP camera and do prediction
def predict_with_manual_roi():
    ip_url = 'http://192.0.0.4:8080/shot.jpg'
    model_path = "vitiligo_efficientnetb0 (1).h5"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f" Model file not found: {model_path}")

    print("[INFO] Building model...")
    model = build_model()

    print("[INFO] Loading model weights...")
    model.load_weights(model_path)
    print(" Model loaded successfully.")

    while True:
        try:
            response = requests.get(ip_url, timeout=5)
            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, -1)

            if frame is None:
                print("⚠️ Could not load frame.")
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

                if cropped.shape[0] < 50 or cropped.shape[1] < 50:
                    print("⚠️ ROI too small. Try selecting a larger region.")
                    continue

                predict_vitiligo_roi(cropped, model, threshold=0.60)
                break

        except Exception as e:
            print(f"⚠️ Error: {str(e)}")
            break

# ✅ Main execution
if __name__ == "__main__":
    predict_with_manual_roi()
