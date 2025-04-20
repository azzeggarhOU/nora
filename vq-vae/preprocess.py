import os
from PIL import Image

# Define input and output folders
input_folder = 'data/thumbnails128x128'
output_folder = 'data/thumbnails64x64'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all images in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # Resize to 64x64
        img_resized = img.resize((64, 64), Image.LANCZOS)

        # Save resized image
        output_path = os.path.join(output_folder, filename)
        img_resized.save(output_path)

print("Done resizing all thumbnails!")
