""" 
Converts all image files to .jpg format and removes other file formats. 
Non-image files are moved to misc folder.
Image files in non-jpg format are moved to delete folder.
"""

import os
import shutil
from PIL import Image, ExifTags
from pathlib import Path

# Source directory to process
source_directory = "/Users/bab226/Pictures/test_dataset/"

# Exclude these directories
excluded_dirs = {"misc", "delete"}

# Folder to move non-JPG files
delete_directory = os.path.join(source_directory, "delete")
misc_directory = os.path.join(source_directory, "misc")

# Create misc and delete_directory folders if they don't exist
os.makedirs(misc_directory, exist_ok=True)
os.makedirs(delete_directory, exist_ok=True)

def is_image_file(file_path):
    """Check if a file is an image based on its extension."""
    image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".heic", ".raw", ".cr2"}
    return file_path.suffix.lower() in image_extensions

def is_jpg_file(file_name):
    """Check if a file is an image JPEG based on its extension."""
    file_path = Path(file_name)
    image_extensions = {".jpg"}
    return file_path.suffix.lower() in image_extensions

def move_non_images():
    """Move non-image JPG files to the 'delete' directory."""
    for file_name in os.listdir(source_directory):
        file_path = os.path.join(source_directory, file_name)
        if os.path.isfile(file_path) and not is_jpg_file(file_name):
            shutil.move(file_path, os.path.join(delete_directory, file_name))
            print(f"Moved non-image file to delete folder: {file_name}")

def convert_to_jpg(file_path):
    """Convert an image to JPG format."""
    try:
        with Image.open(file_path) as img:
            # Extract metadata (if available)
            exif_data = img.info.get("exif")
            
            # Convert to RGB (required for JPG format)
            img = img.convert("RGB")
            
            # Save as JPG
            output_file = file_path.with_suffix(".jpg")
            if exif_data:
                img.save(output_file, "JPEG", quality=95, exif=exif_data)
            else:
                img.save(output_file, "JPEG", quality=95)
            
            print(f"Converted: {file_path} -> {output_file}")
            return output_file
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        return None

def process_directory(directory):
    """Process all files in a directory recursively, excluding certain folders."""
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        
        for file_name in files:
            file_path = Path(root) / file_name
            if file_path.suffix.lower() not in [".jpg"]:
                if is_image_file(file_path):
                    # Convert non-JPG images to JPG
                    converted_file = convert_to_jpg(file_path)
                    # if converted_file:
                        # Optionally delete the original non-JPG file
                        # file_path.unlink()  # Deletes the original file
                else:
                    # Move non-image files to the 'misc' folder
                    dest_path = Path(misc_directory) / file_path.name
                    shutil.move(str(file_path), str(dest_path))
                    print(f"Moved non-image file: {file_path} -> {dest_path}")
    

if __name__ == "__main__":
    # Run the script
    process_directory(source_directory)
    move_non_images()
    print("Processing complete!")