import os
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt

# --- Configurations ---
MODEL_PATH = './vqvae2_web.onnx'
DATA_DIR = './data/thumbnails64x64'
OUTPUT_DIR = './onnx_inference_outputs'
DEVICE = 'cpu'  # onnxruntime will use CPU or GPU if available

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Supported image extensions
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

# --- Preprocessing & Postprocessing ---
# Matches training normalization: (x - 0.5) / 0.5
transform = lambda img: (np.array(img).astype(np.float32) / 255.0 - 0.5) / 0.5
inv_transform = lambda tensor: ((tensor * 0.5 + 0.5) * 255.0).astype(np.uint8)

# --- Load ONNX model ---
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# --- Inference routine ---
def infer(image_path):
    # Load and preprocess
    img = Image.open(image_path).convert('RGB').resize((64, 64))
    x = transform(img)            # HWC
    x = np.transpose(x, (2, 0, 1)) # CHW
    x = np.expand_dims(x, 0)      # NCHW

    # Run ONNX Runtime
    recon = session.run([output_name], {input_name: x})[0]  # numpy array NCHW
    recon = recon[0]  # remove batch dim -> CHW

    # Postprocess
    recon = np.transpose(recon, (1, 2, 0))  # HWC
    recon_img = Image.fromarray(inv_transform(recon))
    return img, recon_img

# --- Run on sample images ---
if __name__ == '__main__':
    # Gather image paths
    img_paths = []
    for root, _, files in os.walk(DATA_DIR):
        for fname in sorted(files):
            if os.path.splitext(fname.lower())[1] in IMG_EXTS:
                img_paths.append(os.path.join(root, fname))

    # Process and time first 5 images or fewer
    timings = []
    for img_path in img_paths[:5]:
        rel_name = os.path.relpath(img_path, DATA_DIR).replace(os.sep, '_')
        out_path = os.path.join(OUTPUT_DIR, f'recon_{rel_name}')

        start = time.time()
        orig_img, recon_img = infer(img_path)
        elapsed = time.time() - start
        timings.append(elapsed)

        # Save side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(orig_img)
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[1].imshow(recon_img)
        axes[1].set_title('Reconstruction')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)

        print(f'Processed {rel_name} in {elapsed:.3f} seconds; saved to {out_path}')

    # Performance summary
    if timings:
        total = sum(timings)
        avg = total / len(timings)
        print(f'Processed {len(timings)} images in {total:.3f}s (avg {avg:.3f}s per image)')
    print('ONNX inference complete.')