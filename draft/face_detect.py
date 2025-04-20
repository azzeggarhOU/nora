import cv2
import numpy as np
import onnxruntime as ort

# Initialize ONNX Runtime session for face recognition (ArcFace)
face_recognition_model_path = 'w600k_r50.onnx'
face_recognition_session = ort.InferenceSession(face_recognition_model_path)

# Initialize ONNX Runtime session for face detection (SCRFD)
scrfd_model_path = 'det_2.5g.onnx'
scrfd_session = ort.InferenceSession(scrfd_model_path)

# Function to preprocess the input image for face recognition
def preprocess_image_for_recognition(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to load at path: {image_path}")
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize to 112x112 as expected by ArcFace model
    image_resized = cv2.resize(image_rgb, (112, 112))
    # Normalize the image
    image_normalized = (image_resized / 255.0 - 0.5) * 2.0
    # Convert to float32 and transpose to (1, 3, 112, 112)
    image_transposed = np.transpose(image_normalized, (2, 0, 1)).astype(np.float32)
    # Add batch dimension to match the model's expected input shape
    image_batch = np.expand_dims(image_transposed, axis=0)
    return image_rgb, image_batch, image

# Function to preprocess the input image for face detection
def preprocess_image_for_detection(image):
    h, w, _ = image.shape
    image_resized = cv2.resize(image, (640, 640))
    image_normalized = (image_resized / 255.0).astype(np.float32)
    image_transposed = np.transpose(image_normalized, (2, 0, 1))[np.newaxis, :]
    return image_resized, image_transposed, (h, w)

# Function to perform face recognition
def recognize_face(image_batch):
    input_name = face_recognition_session.get_inputs()[0].name
    # Run inference
    outputs = face_recognition_session.run(None, {input_name: image_batch})
    # The output is a 1x512 vector representing the face embedding
    face_embedding = outputs[0][0]
    return face_embedding

# Function to perform face detection (SCRFD)
def detect_faces(image_transposed):
    input_name = scrfd_session.get_inputs()[0].name
    # Run inference
    detections = scrfd_session.run(None, {input_name: image_transposed})[0]
    return detections

# Function to draw bounding boxes and labels on the image
def draw_label_and_box(image, boxes, landmarks, label):
    for box, landmark in zip(boxes, landmarks):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Draw facial landmarks
        for (x, y) in landmark:
            cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)
    
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image

# Main function to process the image
def process_image(image_path):
    # Preprocess the image for face detection
    image_rgb, image_batch, original_image = preprocess_image_for_recognition(image_path)
    image_resized, image_transposed, (h, w) = preprocess_image_for_detection(original_image)
    
    # Perform face detection
    detections = detect_faces(image_transposed)
    
    # The detections array might be of shape (N, 6) where N is the number of detected faces
    boxes = detections[:, :4]  # Bounding boxes (x1, y1, x2, y2)
    scores = detections[:, 4]  # Confidence scores
    landmarks = detections[:, 5:].reshape(-1, 5, 2)  # Facial landmarks (5 points for each face)

    # Filter out low-confidence detections
    threshold = 0.5
    valid_indices = np.where(scores > threshold)[0]
    boxes = boxes[valid_indices]
    landmarks = landmarks[valid_indices]

    # Recognize face embeddings for each detected face
    for box in boxes:
        # Crop the face from the image
        x1, y1, x2, y2 = box
        cropped_face = image_rgb[int(y1):int(y2), int(x1):int(x2)]
        
        # Preprocess cropped face for recognition
        _, image_batch_for_recognition, _ = preprocess_image_for_recognition(image_path)
        
        # Perform face recognition (embedding generation)
        face_embedding = recognize_face(image_batch_for_recognition)
        label = f"Embedding: {face_embedding[:5]}"  # Display first 5 values

        # Draw bounding boxes, landmarks, and label
        image_with_label = draw_label_and_box(original_image.copy(), boxes, landmarks, label)
        
    # Convert RGB to BGR for displaying with OpenCV
    image_bgr = cv2.cvtColor(image_with_label, cv2.COLOR_RGB2BGR)
    # Display the image
    cv2.imshow('Face Recognition with Detection', image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Optionally, save the output image
    cv2.imwrite('output_image.jpg', image_bgr)

# Example usage
image_path = 'test.jpeg'
process_image(image_path)
