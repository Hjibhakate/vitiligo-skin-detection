import cv2
import torch
import numpy as np
from torchvision import transforms
import timm
import requests
import time
import os

# ✅ Class labels
CLASS_LABELS = ['Normal Skin', 'Vitiligo Detected']

# ✅ Known model accuracy (e.g., from test set evaluation)
MODEL_TEST_ACCURACY = 0.9583  # 95.83%

# ✅ Model path
model_path = 'efficientnetv2-vitiligo.keras'

# ✅ Check if weights exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found: {model_path}")

# ✅ Load model
model = timm.create_model('efficientnetv2_s', pretrained=False, num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# ✅ Image preprocessing
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ✅ IP webcam URL
ip_url = 'http://192.0.0.4:8080/shot.jpg'

# ✅ Prediction loop
def predict_with_manual_roi():
    while True:
        try:
            response = requests.get(ip_url, timeout=5)
            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, -1)

            if frame is None:
                print("⚠️ Frame decoding failed.")
                continue

            frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)
            cv2.imshow("📱 IP Webcam Feed - Press SPACE to Capture", frame)

            key = cv2.waitKey(1)

            if key % 256 == 32:  # SPACE to capture
                cv2.destroyAllWindows()
                roi = cv2.selectROI("🖼️ Select ROI", frame, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("🖼️ Select ROI")

                if roi == (0, 0, 0, 0):
                    print("⚠️ No ROI selected.")
                    continue

                x, y, w, h = roi
                cropped = frame[y:y+h, x:x+w]
                image_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

                input_tensor = preprocess(image_rgb).unsqueeze(0)

                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)[0]
                    pred_idx = torch.argmax(probs).item()
                    confidence = probs[pred_idx].item()

                print(f"Prediction: {CLASS_LABELS[pred_idx]}")
                print(f"Confidence: {confidence * 100:.2f}%")
                print(f"Model Test Accuracy: {MODEL_TEST_ACCURACY * 100:.2f}%")
                break

            elif key % 256 == 27:  # ESC to exit
                cv2.destroyAllWindows()
                break

        except Exception as e:
            print(f"⚠️ Error: {e}. Retrying in 1s...")
            time.sleep(1)

# ✅ Entry
if __name__ == "__main__":
    predict_with_manual_roi()
