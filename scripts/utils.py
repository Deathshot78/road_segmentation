import subprocess
import sys
import torch
import numpy as np
import cv2
import sknw
from skimage.morphology import skeletonize, remove_small_objects

def run_command(command, working_dir="."):
    """Helper function to run a command in the shell and handle errors."""
    print(f"\n> Executing: {' '.join(command)}")
    try:
        # Using check=True will raise an exception if the command fails
        subprocess.run(command, check=True, cwd=working_dir)
    except FileNotFoundError:
        print(f"Error: Command '{command[0]}' not found.")
        print("Please ensure the necessary programs (like 'aws' or 'tar') are installed and in your system's PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Return code: {e.returncode}")
        sys.exit(1)
        
def post_process_mask(
    prediction_mask_np,
    use_close=True,
    close_kernel_size=5,
    use_min_object_size=True,
    min_object_size=1500,
):
    """
    Applies a series of post-processing steps to a raw binary segmentation mask
    to clean noise, connect gaps, and produce a final road network skeleton.

    Args:
        prediction_mask_np (np.ndarray): The raw, "thick" prediction mask from the model.
        use_close (bool): If True, applies morphological closing.
        close_kernel_size (int): The size of the kernel for morphological closing.
        use_min_object_size (bool): If True, removes small objects from the thick mask.
        min_object_size (int): The minimum pixel area for an object to be kept.

    Returns:
        np.ndarray: The final, cleaned, 1-pixel wide skeletonized mask.
    """

    mask = prediction_mask_np.astype(np.uint8)

    # --- 1. Morphological closing ---
    if use_close:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- 2. Remove small objects ---
    if use_min_object_size:
        mask = remove_small_objects(mask.astype(bool), min_size=min_object_size).astype(np.uint8)

    if np.sum(mask) == 0:
        return mask  

    return mask

def visualize_image(tensor):
    """De-normalizes a tensor and prepares it for visualization."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.cpu().numpy().transpose(1, 2, 0)
    image = std * tensor + mean
    image = np.clip(image, 0, 1)
    return image
