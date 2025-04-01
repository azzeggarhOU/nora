import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2

def preprocess_image(image_path, target_size):
    """
    Load and preprocess the image to match the model's input requirements.
    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Desired size (width, height) for the input image.

    Returns:
        np.ndarray: Preprocessed image ready for model input.
    """
    # Load the image
    image = Image.open(image_path).convert('RGBA')  # Convert to RGBA to ensure 4 channels

    # Resize the image to the target size using LANCZOS resampling
    image = image.resize(target_size, Image.Resampling.LANCZOS)

    # Convert image to numpy array and normalize pixel values to [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0

    # Transpose the array to match the model's input shape (channels, height, width)
    image_array = np.transpose(image_array, (2, 0, 1))

    # Add a batch dimension (1, channels, height, width)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

def main():
    # Path to your ONNX model
    model_path = 'nora_draft.onnx'

    # Path to the input image
    image_path = 'test.jpeg'

    # Load the ONNX model
    session = ort.InferenceSession(model_path)

    # Get model input details
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape

    # Determine the target size for the input image
    # Assuming the model expects input shape (batch_size, channels, height, width)
    target_size = (input_shape[3], input_shape[2])  # (width, height)

    # Preprocess the image
    input_data = preprocess_image(image_path, target_size)

    # Run inference
    outputs = session.run(None, {input_name: input_data})

    # Process and display the output
    # Assuming the model outputs an image; adjust this part based on your model's output
    output_image = outputs[0][0]  # Remove batch dimension

    # Transpose back to (height, width, channels)
    output_image = np.transpose(output_image, (1, 2, 0))

    # Denormalize pixel values from [0, 1] to [0, 255]
    output_image = (output_image * 255).astype(np.uint8)

    # Display the output image
    cv2.imshow('Output Image', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
