from PIL import Image
from glob import glob
from numpy import random
import pathlib

# Adjust the path and file pattern as needed
image_files = glob("/Users/bab226/Pictures/backup_galexy_s10/*.jpg")

# Choose 20 random images:
#Select random intager in range:
for img_file in image_files[:50]:
    random_int = random.randint(1, len(image_files))
    random_image_file = image_files[random_int]
    #with Image.open(random_image_file) as img:
        #print(f"{img_file}: {img.size}")  # prints (width, height)
    print(random_image_file)
    # # Copy image to new directory called test_dataset
    # new_dir = "/Users/bab226/Pictures/test_dataset/"
    # # Get image file name without paht:
    # path = pathlib.Path(random_image_file)
    # image_name = path.name
    # new_image_file = f"{new_dir}{image_name}"
    # Image.open(random_image_file).copy().save(new_image_file)

