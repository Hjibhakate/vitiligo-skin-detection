# import os
# import cv2
# import numpy as np
# import requests
# import tensorflow as tf
# from tensorflow.keras.applications.resnet50 import preprocess_input

# # ✅ Load the TensorFlow SavedModel
# MODEL_PATH = "resnet101_vitiligo_savedmodel"
# print("[INFO] Loading model...")
# try:
#     model = tf.saved_model.load(MODEL_PATH)
#     infer = model.signatures["serving_default"]
#     print("[INFO] Model loaded successfully.")
# except Exception as e:
#     print(f"[ERROR] Failed to load model: {e}")
#     exit(1)

# # ✅ Class Labels
# CLASS_LABELS = ['Normal Skin', 'Vitiligo Detected']

# # ✅ IP Webcam URL
# IP_WEBCAM_URL = 'http://10.190.134.52:8080/shot.jpg'

# # ✅ Preprocess ROI
# def preprocess_roi(roi):
#     roi = cv2.resize(roi, (224, 224))
#     roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#     roi = preprocess_input(roi.astype(np.float32))
#     roi = np.expand_dims(roi, axis=0)
#     return roi

# # ✅ Predict
# def predict_roi(roi):
#     try:
#         preprocessed = preprocess_roi(roi)
#         input_tensor = tf.convert_to_tensor(preprocessed)
#         outputs = infer(input_tensor)
#         prediction = list(outputs.values())[0].numpy()[0]
#         class_id = int(np.argmax(prediction))
#         confidence = float(np.max(prediction)) * 100
#         return CLASS_LABELS[class_id], confidence
#     except Exception as e:
#         print(f"[ERROR] Prediction failed: {e}")
#         return "Error", 0.0

# # ✅ Main Loop
# print("[INFO] Starting IP Webcam preview. Press SPACE to capture, select ROI, then press ENTER to detect.")
# print("[INFO] Press Q or ESC to quit.")

# while True:
#     try:
#         # Capture frame from IP camera
#         response = requests.get(IP_WEBCAM_URL, timeout=5)
#         img_array = np.array(bytearray(response.content), dtype=np.uint8)
#         frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#         if frame is None:
#             print("[WARNING] Empty frame. Retrying...")
#             continue

#         frame = cv2.resize(frame, (640, 480))
#         display_frame = frame.copy()

#         # Show camera feed
#         cv2.putText(display_frame, "Press SPACE to capture | ENTER to detect | Q/ESC to exit",
#                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#         cv2.imshow("Vitiligo Detection - Live", display_frame)

#         key = cv2.waitKey(1) & 0xFF

#         # Quit
#         if key == 27 or key == ord('q'):
#             print("[INFO] Exiting.")
#             break

#         # SPACE pressed → capture frame and ask for ROI
#         if key == 32:  # SPACE key
#             print("[INFO] Captured frame. Please select ROI with mouse.")
#             roi_coords = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
#             x, y, w, h = roi_coords

#             if w == 0 or h == 0:
#                 print("[WARNING] Invalid ROI selected.")
#                 continue

#             selected_region = frame[y:y+h, x:x+w]

#             # Wait for ENTER to trigger detection
#             print("[INFO] Press ENTER to detect, or ESC to cancel.")
#             while True:
#                 temp = cv2.waitKey(0) & 0xFF
#                 if temp == 13:  # ENTER key
#                     print("[INFO] Running detection...")
#                     label, confidence = predict_roi(selected_region)

#                     # Annotate result
#                     result_frame = frame.copy()
#                     cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#                     cv2.putText(result_frame, f"{label} ({confidence:.1f}%)", (x, y - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#                     cv2.imshow("Prediction Result", result_frame)
#                     print(f"[RESULT] {label} ({confidence:.2f}%)")
#                     break
#                 elif temp == 27:  # ESC
#                     print("[INFO] Detection cancelled.")
#                     break

#             cv2.destroyWindow("Select ROI")
#             cv2.waitKey(0)
#             cv2.destroyWindow("Prediction Result")

#     except requests.exceptions.RequestException as re:
#         print(f"[ERROR] IP webcam error: {re}")
#     except Exception as e:
#         print(f"[ERROR] Unexpected error: {e}")
#         break

# # ✅ Cleanup
# cv2.destroyAllWindows()





import os
import cv2
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load TensorFlow SavedModel
MODEL_PATH = "resnet101_vitiligo_savedmodel"
print("[INFO] Loading model...")
try:
    model = tf.saved_model.load(MODEL_PATH)
    infer = model.signatures["serving_default"]
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit(1)

# Class labels
CLASS_LABELS = ['Normal Skin', 'Vitiligo Detected']

# IP Webcam URL
IP_WEBCAM_URL = 'http://192.0.0.4:8080/shot.jpg'

def preprocess_roi(roi):
    roi = cv2.resize(roi, (224, 224))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = preprocess_input(roi.astype(np.float32))
    roi = np.expand_dims(roi, axis=0)
    return roi

def predict_roi(roi):
    try:
        preprocessed = preprocess_roi(roi)
        input_tensor = tf.convert_to_tensor(preprocessed)
        outputs = infer(input_tensor)
        prediction = list(outputs.values())[0].numpy()[0]
        class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100
        return CLASS_LABELS[class_id], confidence
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return "Error", 0.0

print("[INFO] Starting IP Webcam preview.")
print("[INFO] Press SPACE to capture & select ROI.")
print("[INFO] After ROI, press ENTER to detect or ESC to cancel.")
print("[INFO] Press Q or ESC anytime to quit.")

while True:
    try:
        # Fetch frame from IP webcam
        response = requests.get(IP_WEBCAM_URL, timeout=5)
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            print("[WARNING] Empty frame. Retrying...")
            continue

        frame = cv2.resize(frame, (640, 480))
        display_frame = frame.copy()

        # Show live feed
        cv2.putText(display_frame, "Press SPACE to capture | Q/ESC to exit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Vitiligo Detection - Live", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # Quit keys
        if key == 27 or key == ord('q'):
            print("[INFO] Exiting.")
            break

        # SPACE key to capture and select ROI
        if key == 32:  # SPACE pressed
            print("[INFO] Captured frame. Select ROI with mouse.")
            roi_coords = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
            x, y, w, h = roi_coords

            if w == 0 or h == 0:
                print("[WARNING] Invalid ROI selected. Returning to live view.")
                cv2.destroyWindow("Select ROI")
                continue

            selected_region = frame[y:y+h, x:x+w]

            print("[INFO] Press ENTER to detect, or ESC to cancel.")
            while True:
                temp = cv2.waitKey(0) & 0xFF
                if temp == 13:  # ENTER key
                    print("[INFO] Running detection...")

                    # Close all windows except result
                    cv2.destroyAllWindows()

                    label, confidence = predict_roi(selected_region)

                    result_frame = frame.copy()
                    cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(result_frame, f"{label} ({confidence:.1f}%)", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow("Prediction Result", result_frame)
                    print(f"[RESULT] {label} ({confidence:.2f}%)")

                    cv2.waitKey(0)  # Wait for any key press to close result
                    cv2.destroyAllWindows()
                    exit(0)  # Exit program after showing result

                elif temp == 27:  # ESC key cancels detection and returns to live feed
                    print("[INFO] Detection cancelled. Returning to live preview.")
                    cv2.destroyWindow("Select ROI")
                    break

    except requests.exceptions.RequestException as re:
        print(f"[ERROR] IP webcam connection error: {re}")
    except Exception as e:
        print(f"[ERROR] Unexpected runtime error: {e}")
        break

cv2.destroyAllWindows()
