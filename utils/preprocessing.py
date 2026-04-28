# ============================================================
# Image Preprocessing Utilities
# ============================================================

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

def preprocess_images(input_dir, output_dir, min_size=100, max_size=2000):
    """
    Preprocess images: validate, resize, normalize.
    
    Args:
        input_dir (str): Input directory
        output_dir (str): Output directory
        min_size (int): Minimum image dimension
        max_size (int): Maximum image dimension
        
    Returns:
        int: Number of processed images
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
    image_files = [
        f for f in os.listdir(input_dir)
        if Path(f).suffix.lower() in image_extensions
    ]
    
    processed_count = 0
    
    for filename in tqdm(image_files, desc="Preprocessing images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Open image
            img = Image.open(input_path).convert("RGB")
            
            # Check size
            if img.size[0] < min_size or img.size[1] < min_size:
                continue  # Skip too small images
            
            # Resize if too large
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Save
            img.save(output_path, quality=95)
            processed_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"✅ Processed {processed_count}/{len(image_files)} images")
    return processed_count

def augment_image(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """
    Data augmentation for image.
    
    Args:
        image: PIL Image
        brightness: Brightness augmentation factor
        contrast: Contrast augmentation factor
        saturation: Saturation augmentation factor
        hue: Hue augmentation factor
        
    Returns:
        PIL Image: Augmented image
    """
    from torchvision import transforms
    
    transform = transforms.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue
    )
    return transform(image)
