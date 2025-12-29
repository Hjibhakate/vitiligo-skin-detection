import cv2
import os
import numpy as np
import requests
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input

# ✅ Class labels
CLASS_LABELS = ['Normal Skin', 'Vitiligo Detected']
# resnet 50
# ✅ Build ResNet50 Model
def build_model(input_shape=(128, 128, 3), num_classes=2):
    base_model = ResNet50(include_top=False, weights=None, input_tensor=Input(shape=input_shape))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=output)

# ✅ Prediction from ROI
def predict_vitiligo_roi(roi_image, model):
    resized = cv2.resize(roi_image, (128, 128)) / 255.0
    img_array = np.expand_dims(resized, axis=0)
    prediction = model.predict(img_array)[0]

    predicted_index = np.argmax(prediction)
    predicted_class = CLASS_LABELS[predicted_index]
    confidence = prediction[predicted_index]

    # ✅ Flask-compatible print output
    print(f"Prediction: {predicted_class}")
    print(f"Accuracy: {confidence * 100:.2f}%")

# ✅ Main camera + ROI flow
def predict_with_manual_roi():
    ip_url = 'http://192.0.0.4:8080/shot.jpg'  # IP webcam snapshot URL
    model_path = "final_model_finetuned.h5"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = build_model()
    model.load_weights(model_path)
    model.make_predict_function()

    while True:
        try:
            # Fetch snapshot from IP webcam
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
            print("⚠️ Error accessing webcam snapshot. Retrying...")
            cv2.destroyAllWindows()
            break

# ✅ Entry
if __name__ == "__main__":
    predict_with_manual_roi()
