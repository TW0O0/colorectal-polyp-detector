import os
import shutil
import random
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

def create_directory_structure(base_dir):
    """
    Create the necessary directory structure for the dataset
    Args:
        base_dir: Base directory for the dataset
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # Create train, validation, and test directories
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Create class directories
        for class_name in ['normal', 'benign', 'cancerous']:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

def process_cvc_clinicdb(source_dir, target_base_dir):
    """
    Process the CVC-ClinicDB dataset
    Args:
        source_dir: Directory containing CVC-ClinicDB dataset
        target_base_dir: Base directory for processed dataset
    """
    print("Processing CVC-ClinicDB dataset...")
    
    # CVC-ClinicDB contains polyp images and masks
    # Images are in Original folder, masks in Ground Truth folder
    originals_dir = os.path.join(source_dir, 'Original')
    masks_dir = os.path.join(source_dir, 'Ground Truth')
    
    if not os.path.exists(originals_dir) or not os.path.exists(masks_dir):
        print(f"Error: CVC-ClinicDB dataset structure not found in {source_dir}")
        return
    
    # Get list of image files
    image_files = [f for f in os.listdir(originals_dir) if f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(image_files)
    
    # Calculate split sizes (70% train, 15% validation, 15% test)
    train_size = int(0.7 * len(image_files))
    val_size = int(0.15 * len(image_files))
    
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]
    
    # Process files for each split
    for file_list, split_name in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
        for file_name in tqdm(file_list, desc=f"Processing {split_name} set"):
            # Determine if the image contains a polyp by checking the corresponding mask
            mask_file = os.path.join(masks_dir, file_name)
            
            if os.path.exists(mask_file):
                # Load the mask to determine polyp size
                mask = Image.open(mask_file).convert('L')
                mask_array = np.array(mask)
                polyp_pixels = np.sum(mask_array > 0)
                
                # Determine class based on polyp size (simplified heuristic)
                image_class = 'benign'  # Default class for polyps
                
                # If polyp is large, mark as potentially cancerous (simplified rule)
                if polyp_pixels > 0.1 * mask_array.size:
                    image_class = 'cancerous'
            else:
                # If no mask exists or no polyp in mask, categorize as normal
                image_class = 'normal'
            
            # Copy the image to the appropriate directory
            source_file = os.path.join(originals_dir, file_name)
            target_dir = os.path.join(target_base_dir, split_name, image_class)
            
            # Convert image format if needed and save
            try:
                img = Image.open(source_file)
                # Save as jpg to standardize format
                target_file = os.path.join(target_dir, f"{os.path.splitext(file_name)[0]}.jpg")
                img.convert('RGB').save(target_file, 'JPEG')
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

def process_kvasir_seg(source_dir, target_base_dir):
    """
    Process the Kvasir-SEG dataset
    Args:
        source_dir: Directory containing Kvasir-SEG dataset
        target_base_dir: Base directory for processed dataset
    """
    print("Processing Kvasir-SEG dataset...")
    
    # Kvasir-SEG contains images in images folder and masks in masks folder
    images_dir = os.path.join(source_dir, 'images')
    masks_dir = os.path.join(source_dir, 'masks')
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"Error: Kvasir-SEG dataset structure not found in {source_dir}")
        return
    
    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(image_files)
    
    # Calculate split sizes (70% train, 15% validation, 15% test)
    train_size = int(0.7 * len(image_files))
    val_size = int(0.15 * len(image_files))
    
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]
    
    # Process files for each split
    for file_list, split_name in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
        for file_name in tqdm(file_list, desc=f"Processing {split_name} set"):
            # Check corresponding mask
            mask_file = os.path.join(masks_dir, file_name.replace('.jpg', '.png').replace('.png', '.png'))
            
            if os.path.exists(mask_file):
                # Load the mask to determine polyp characteristics
                mask = Image.open(mask_file).convert('L')
                mask_array = np.array(mask)
                polyp_pixels = np.sum(mask_array > 0)
                
                # Determine class based on polyp size and shape
                image_class =
