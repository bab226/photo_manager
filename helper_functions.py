#!bin/python

# Import libraries
import os
import cv2
import dlib
import numpy as np
from PIL import Image, ImageEnhance
import face_recognition
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import random
import piexif
import json

# Loads images from a directory, detects faces, and encodes each face.

def detect_faces(image):
    return face_recognition.face_locations(image, model='hog', number_of_times_to_upsample=2)  # Using CNN for better accuracy.

def rotate_image(image, angle):
    return image.rotate(angle, expand=True)

def preprocess_image(image_path, output_path, contrast_factor=1.5, resize_dim=(256, 256)):
    """
    Preprocess the image by adjusting contrast, resizing, and aligning the face.
    
    :param image_path: Path to the input image.
    :param output_path: Path to save the preprocessed image.
    :param contrast_factor: Factor by which to increase the contrast (default is 1.5).
    :param resize_dim: Tuple for the target resize dimensions (default is (256, 256)).
    """

# Load dlib's face detector and shape predictor

def preprocess_image(image_path, contrast_factor=1.5, resize_dim=(256, 256)):
    """
    Preprocess the image by adjusting contrast and resizing with facial alignment.
    :param image_path: Path to the input image.
    :param contrast_factor: Factor by which to increase the contrast (default is 1.5).
    :param resize_dim: Tuple for the target resize dimensions (default is (256, 256)).
    :return: Preprocessed image.
    """
    
    # Determine the path to the shape_predictor_68_face_landmarks.dat file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    predictor_path = os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Load image using PIL
    img_pil = Image.open(image_path)
    
    # Adjust contrast
    img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast_factor)
    
    # Convert PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using dlib
    faces = detector(gray, 1)
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Get the aligned face using the landmarks
        aligned_face = dlib.get_face_chip(img_cv, landmarks)
        
        # Resize to target dimensions
        resized_face = cv2.resize(aligned_face, resize_dim)
        
        return resized_face
    
    # If no face detected, return None
    print("No faces detected...")
    return None

def tryVar(var):
    try:
        val = var
    except NameError:
        return None
    return val

def load_and_process_images(directory, contrast_factor=1.5):
    images_with_paths = []
    for filename in os.listdir(directory):
        print("Detecting faces in %s" %(filename))
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory, filename)  # Get file path
            preprocessed_image = preprocess_image(image_path, contrast_factor)
            if preprocessed_image is not None:
                images_with_paths.append((image_path, preprocessed_image))
    return images_with_paths

def extract_encodings_with_paths(images_with_paths):
    """
    Extract encodings for each preprocessed image.
    
    :param images_with_paths: List of tuples containing image path and preprocessed image.
    :return: List of tuples containing image path and encoding.
    """
    encodings_with_paths = []
    for image_path, image in images_with_paths:
        # Convert to RGB for face_recognition
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Extract face encodings
        encodings = face_recognition.face_encodings(rgb_image)
        if encodings:
            encodings_with_paths.append((image_path, encodings[0]))
    return encodings_with_paths

def find_optimal_cluster_and_plot(data, max_k=10):
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    plt.plot(range(1, max_k + 1), inertia, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.show()
    
    optimal_k = np.argmin(np.diff(inertia)) + 2
    return optimal_k

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def find_optimal_cluster_silhouette(data, max_k=10, output_dir='./plots'):
    """Find the optimal number of clusters using silhouette scores and save the plot."""
    n_samples = len(data)
    if n_samples < 2:
        raise ValueError("Number of samples must be at least 2.")
    
    max_k = min(max_k, n_samples - 1)
    
    silhouette_scores = []
    k_values = range(2, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    if not silhouette_scores:
        raise ValueError("Unable to calculate silhouette scores. Please check the input data.")
    
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal Number of Clusters')
    plt.grid(True)
    
    # Save the plot to a file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'silhouette_scores.png')
    plt.savefig(output_file)
    plt.close()
    
    optimal_k = k_values[np.argmax(silhouette_scores)]
    return optimal_k

def validate_clusters(data, k):
    """Ensure the number of clusters is valid."""
    if k >= len(data):
        raise ValueError(f"Number of clusters (k={k}) must be less than the number of samples (n={len(data)}).")
    return True

def visualize_cluster_images(images_with_clusters, num_samples=5):
    clusters = {}
    for image_path, cluster in images_with_clusters:
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(image_path)

    # Show a few samples from each cluster
    for cluster, image_paths in clusters.items():
        print(f"Cluster {cluster} contains {len(image_paths)} images")
        samples = random.sample(image_paths, min(num_samples, len(image_paths)))
        
        fig, ax = plt.subplots(1, len(samples), figsize=(15, 5))
        if len(samples) == 1:
            ax = [ax]  # Convert to list for consistency in iteration
            
        fig.suptitle(f"Cluster {cluster}")
        for i, img_path in enumerate(samples):
            img = Image.open(img_path)
            ax[i].imshow(img)
            ax[i].axis('off')
        plt.show()

    #         image = face_recognition.load_image_file(image_path)  # Load image
    #         pil_image = Image.fromarray(image)
    #         face_locations = detect_faces(image)  # Detect faces
    
    #         if face_locations:  # If face detected, append to list
    #             print("Face detected in picture %s" %(filename))
    #         else:
    #             # If no faces, try rotating the image by 90, 180, 270 degrees.
    #             angles = [0, 90, 180, 270]
    #             for angle in angles:
    #                 if face_locations:
    #                     print("Face detected in picture %s" %(filename))
    #                     break
    #                 else:
    #                     print("Face NOT detected. Trying again...")
    #                     rotated_image = rotate_image(pil_image, angle)
    #                     rotated_array = np.array(rotated_image)
    #                     face_locations = detect_faces(rotated_array)
                    
    #             print("Skipping picture %s..." %(filename))
                
    #         face_encodings = face_recognition.face_encodings(image, face_locations)
    #         images_with_faces.append((image_path, image, face_locations, face_encodings))
    # return images_with_faces

# def extract_encodings_with_paths(images_with_faces):
#     encodings_with_paths = []
#     for item in images_with_faces:
#         image_path, _, _, face_encodings = item
#         for encoding in face_encodings:
#             encodings_with_paths.append((image_path, list(encoding)))  # Ensure encoding is a 1D list
#     return encodings_with_paths

# def cluster_faces_with_paths(encodings_with_paths, n_clusters):
#     encodings = [encoding for _, encoding in encodings_with_paths]
#     kmeans = KMeans(n_clusters=n_clusters)
#     labels = kmeans.fit_predict(encodings)
    
#     clusters = {i: [] for i in range(n_clusters)}
#     for label, (image_path, _) in zip(labels, encodings_with_paths):
#         clusters[label].append(image_path)
    
#     return clusters, labels

# def plot_optimal_clusters(encodings, max_k):
#     """Find optimal clusters using elbow method."""
#     iters = range(1, max_k+1)
#     distortions = []
    
#     for k in iters:
#         kmeans = KMeans(n_clusters=k)
#         kmeans.fit(encodings)
#         distortions.append(kmeans.inertia_)
        
#     plt.figure(figsize=(8, 6))
#     plt.plot(iters, distortions, marker='o')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Inertia')
#     plt.title('Elbow method for determining optimal number of clusters')
#     plt.show()

def find_optimal_cluster(encodings, max_k):
    """Find cluster with the least inertia using Elbow method."""
    inertia_values = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(encodings)
        inertia_values.append(kmeans.inertia_)

    min_inertia = min(inertia_values)
    kmin = inertia_values.index(min_inertia) + 1  # Convert index to cluster number (since index starts from 0)
    return kmin

def visualize_and_tag_clusters(images_with_clusters, num_samples=5):
    clusters = {}
    for image_path, cluster in images_with_clusters:
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(image_path)

    cluster_tags = {}
    # Show a few samples from each cluster and get tags
    for cluster, image_paths in clusters.items():
        print(f"Cluster {cluster} contains {len(image_paths)} images")
        samples = random.sample(image_paths, min(num_samples, len(image_paths)))
        
        fig, ax = plt.subplots(1, len(samples), figsize=(15, 5))
        if len(samples) == 1:
            ax = [ax]  # Convert to list for consistency in iteration
            
        fig.suptitle(f"Cluster {cluster}")
        for i, img_path in enumerate(samples):
            img = Image.open(img_path)
            ax[i].imshow(img)
            ax[i].axis('off')
        plt.show()
        
        # Prompt user for a tag
        tag = input(f"Enter a tag for Cluster {cluster} (e.g., 'Person A'): ")
        cluster_tags[int(cluster)] = tag  # Ensure keys are regular integers
    return clusters, cluster_tags

def save_tags_to_file(cluster_tags, file_path='cluster_tags.json'):
    with open(file_path, 'w') as file:
        json.dump(cluster_tags, file, indent=4)

def add_tags_to_jpeg_images(clusters, cluster_tags):
    """
    Adds a tag to each JPEG image in the specified directory, using the cluster tags.
    Saves the modified images with the tags.
    
    Args:
        clusters (dict): Dictionary of cluster IDs to image paths.
        cluster_tags (dict): Dictionary of cluster IDs to tags.
    """
    for cluster_id, image_paths in clusters.items():
        tag = cluster_tags.get(cluster_id, 'untagged')
        for image_path in image_paths:
            image = Image.open(image_path)
            exif_dict = piexif.load(image.info['exif'])
            exif_dict['Exif'][piexif.ExifIFD.UserComment] = tag.encode('utf-8')
            exif_bytes = piexif.dump(exif_dict)
            image.save(image_path, 'jpeg', exif=exif_bytes)
            print(f"Saved tag '{tag}' to image {image_path}")

def search_images_by_tag(directory, search_tag):
    """
    Searches for JPEG images in the specified directory with the specified tag.
    Returns a list of image paths that match the tag.
    Args:
        directory (str): Path to the directory containing JPEG images.
        search_tag (str): Tag to search for in the images."""

    print(f"Searching for images with tag '{search_tag}'...")
    print("Found:")
    
    matching_images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            exif_dict = piexif.load(image.info['exif'])
            user_comment = exif_dict['Exif'].get(piexif.ExifIFD.UserComment, b'').decode('utf-8')
            if search_tag == user_comment:
                matching_images.append(image_path)
    return matching_images

def save_clusters_to_file(clusters, cluster_labels, directory):
    """Save clusters and inertia values to a csv file"""
    with open(os.path.join(directory, 'clusters.csv'), 'w') as file:
        file.write('Cluster ID,Label,Inertia\n')
        for cluster_id, label in zip(cluster_labels, clusters):
            inertia = KMeans(n_clusters=len(label)).fit(label).inertia_
            file.write(f'{cluster_id},{label[0]},{inertia}\n')

def plot_cluster_inertia(filename):
    """Plot cluster inertia values from csv file"""
    df = pd.read_csv(filename)
    sns.barplot(x='Cluster ID', y='Inertia', data=df)
    plt.title('Cluster Inertia')
    plt.show()