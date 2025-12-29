import cv2
import torch
import numpy as np
from torchvision import transforms
import timm
import requests
import time

# ✅ Class labels (formatted for direct printing)
CLASS_LABELS = ['Normal Skin', 'Vitiligo Detected']

# ✅ Load ResNet-RS-50 model
model = timm.create_model('resnetrs50', pretrained=False, num_classes=2)
model.load_state_dict(torch.load('best_model_updated(1).pth', map_location=torch.device('cpu')))
model.eval()

# ✅ Image transform pipeline
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ✅ IP Webcam snapshot URL
ip_url = 'http://192.0.0.4:8080/shot.jpg'  # Replace with your IP camera URL

def predict_with_manual_roi():
    while True:
        try:
            # 🖼️ Fetch image from IP webcam
            img_resp = requests.get(ip_url, timeout=5)
            img_array = np.array(bytearray(img_resp.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, -1)

            # Resize & flip for user-friendly view
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)
            cv2.imshow("📱 IP Webcam Feed - Press SPACE to capture", frame)

            key = cv2.waitKey(1)

            if key % 256 == 32:  # SPACE key
                cv2.destroyAllWindows()
                roi = cv2.selectROI("🖼️ Select ROI", frame, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("🖼️ Select ROI")

                if roi == (0, 0, 0, 0):
                    return  # No region selected

                x, y, w, h = roi
                cropped = frame[y:y+h, x:x+w]
                image_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                input_tensor = preprocess(image_rgb).unsqueeze(0)

                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                    prediction = torch.argmax(probabilities).item()
                    confidence = probabilities[prediction].item()

                # ✅ Print both prediction and accuracy
                print(f"Prediction: {CLASS_LABELS[prediction]}")
                print(f"Accuracy: {confidence * 100:.2f}%")
                break

            elif key % 256 == 27:  # ESC key
                cv2.destroyAllWindows()
                break

        except Exception as e:
            time.sleep(1)

# ✅ Entry Point
if __name__ == "__main__":
    predict_with_manual_roi()
