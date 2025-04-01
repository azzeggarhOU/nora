import cv2
import numpy as np

# Load YuNet model
model_path = "./face_detection_yunet_2023mar.onnx"
face_detector = cv2.FaceDetectorYN_create(model_path, "", (640, 640))

# Tune detection thresholds
face_detector.setScoreThreshold(0.5)
face_detector.setNMSThreshold(0.3)
face_detector.setTopK(5000)

def crop_face(image_path, output_face_path="cropped_face.jpg", output_bg_path="background.jpg"):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image '{image_path}'. Check the file path.")
        return
    
    img = cv2.resize(img, (640, 640))  # Resize for better detection
    face_detector.setInputSize((640, 640))
    
    faces = face_detector.detect(img)
    print("Raw detection output:", faces)  # Debugging

    if faces is not None and faces[1] is not None:
        # Extract face bounding box (x, y, w, h)
        x, y, w, h = map(int, faces[1][0][:4])
        
        # Create a square bounding box around the face
        max_side = max(w, h)
        padding_x = (max_side - w) // 2
        padding_y = (max_side - h) // 2

        # Adjust x, y for the padding (ensure the face stays centered)
        x = max(x - padding_x, 0)
        y = max(y - padding_y, 0)

        # Ensure the square crop does not exceed image boundaries
        face_crop = img[y:y + max_side, x:x + max_side]

        # Create the background image (blackout the face part)
        background = img.copy()

        # Blackout the face region in the background image
        background[y:y + max_side, x:x + max_side] = 0  # Set the face area to black

        # Save the cropped face and background
        cv2.imwrite(output_face_path, face_crop)
        cv2.imwrite(output_bg_path, background)

        print(f"Face cropped and saved to {output_face_path}")
        print(f"Background saved to {output_bg_path}")
    else:
        print("No face detected!")

# Example usage
crop_face("test.jpeg")
