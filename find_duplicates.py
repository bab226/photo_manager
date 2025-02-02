""" Removve duplicate images from photos folder and move them to the duplicates folder.
    The script will find all images in the specified directory and its subdirectories, 
    compute an MD5 hash for each image, and then compare the hashes to find duplicates.
    The duplicates will be moved to a separate folder for review or deletion.
    Args:
        directory (str): The path to the directory containing the photos.
        Example: python find_duplicates.py /path/to/photos/"""

import os
import hashlib
import shutil

# Step 1: Function to load and read image files
def find_images(directory):
    image_files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append(os.path.join(dirpath, filename))
    return image_files

# Step 2: Function to generate image hashes
def hash_image(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Step 3: Function to compare image hashes and find duplicates
def find_duplicates(image_files):
    hashes = {}
    duplicates = []
    for filepath in image_files:
        image_hash = hash_image(filepath)
        if image_hash in hashes:
            duplicates.append((filepath, hashes[image_hash]))
        else:
            hashes[image_hash] = filepath
    return duplicates

# Step 4: Function to delete duplicate images
def delete_duplicates(duplicates):
    if not os.path.exists("./duplicates/"):
        os.makedirs("./duplicates/")
    for duplicate, original in duplicates:
        # Uncomment the next line to actually delete the file
        # os.remove(duplicate)
        print(f"Duplicate: {duplicate} -> Original: {original}")
        # Optionally, move duplicates to a separate folder for review
        shutil.move(duplicate, "./duplicates/")

# Main function to execute the program
def main(directory):
    image_files = find_images(directory)
    duplicates = find_duplicates(image_files)
    delete_duplicates(duplicates)

# Run the program
if __name__ == "__main__":
    main("/Users/bab226/Pictures/")