import cv2
import numpy as np
import torch
import onnx
import onnxruntime as ort

# Load YuNet face detector (OpenCV)
model_path = "./face_detection_yunet_2023mar.onnx"
face_detector = cv2.FaceDetectorYN_create(model_path, "", (320, 320))

# Configure detection thresholds
face_detector.setScoreThreshold(0.7)  # Higher = fewer false positives
face_detector.setNMSThreshold(0.3)
face_detector.setTopK(5000)

def crop_face(image_path, output_face_path="cropped_face.jpg", output_bg_path="background.jpg"):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image '{image_path}'. Check the file path.")
        return
    
    h, w = img.shape[:2]
    scale = 640 / max(h, w)  # Preserve aspect ratio
    img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    
    face_detector.setInputSize((img_resized.shape[1], img_resized.shape[0]))
    faces = face_detector.detect(img_resized)

    if faces is not None and faces[1] is not None:
        x, y, w, h = map(int, faces[1][0][:4])
        
        # Ensure square bounding box
        max_side = max(w, h)
        padding_x = (max_side - w) // 2
        padding_y = (max_side - h) // 2
        x, y = max(x - padding_x, 0), max(y - padding_y, 0)

        face_crop = img_resized[y:y + max_side, x:x + max_side]

        # Blur face instead of blacking it out
        background = img_resized.copy()
        face_region = background[y:y + max_side, x:x + max_side]
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
        background[y:y + max_side, x:x + max_side] = blurred_face

        cv2.imwrite(output_face_path, face_crop)
        cv2.imwrite(output_bg_path, background)

        print(f"Face cropped and saved to {output_face_path}")
        print(f"Background saved to {output_bg_path}")
    else:
        print("No face detected!")

def export_face_detector_onnx(output_path="face_detector.onnx"):
    dummy_input = torch.randn(1, 3, 320, 320)
    torch.onnx.export(
        face_detector,
        dummy_input,
        output_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Face detection model exported to {output_path}")

# Example usage
crop_face("test.jpeg")
export_face_detector_onnx()
