# To load images and cluster them based on similarity.
# Allows for manual tagging of clusters and assignment of names to identify people.
# These labels are saved in a separate file for each cluster to be used on future pictures.

import helper_functions as hf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json 

# Main
directory = "/Users/bab226/Pictures/test_photos/test_photos"

# Load and preprocess images
images_with_paths = hf.load_and_process_images(directory, 1.5)
max_k=len(images_with_paths) - 1  # Max clusters < number of images

# Extract facial encodings
encodings_with_paths = hf.extract_encodings_with_paths(images_with_paths)
all_encodings = [encoding for _, encoding in encodings_with_paths]

# Find the optimal number of clusters and plot inertia vs. number of clusters
optimal_k = hf.find_optimal_cluster_silhouette(all_encodings, max_k=max_k)
print(f"Optimal number of clusters: {optimal_k}")

# Perform clustering with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(all_encodings)

# Assign cluster labels to images
images_with_clusters = [(path, cluster) for (path, encoding), cluster in zip(encodings_with_paths, cluster_labels)]

# Print results and tag clusters
for path, cluster in images_with_clusters:
    print(f"Image: {path} is in Cluster {cluster}")

# Visualize some samples from each cluster and tag them
clusters, cluster_tags = hf.visualize_and_tag_clusters(images_with_clusters, num_samples=5)

# Print cluster tags
print("Cluster tags:")
for cluster_id, tag in cluster_tags.items():
    print(f"Cluster {cluster_id}: {tag}")

# Add tags to images
hf.add_tags_to_jpeg_images(clusters, cluster_tags)

# Optionally, save the cluster_tags dictionary to a file
with open('./cluster_tags.json', 'w') as f:
    json.dump(cluster_tags, f)

# Example verification with ExifTool
# Open Terminal and use ExifTool to verify tags
# exiftool -UserComment /path/to/your/image.jpg

# Example usage
# search_tag = 'alex'

# matching_images = search_images_by_tag(directory, search_tag)
# print(f"Images with tag '{search_tag}':")
# for image_path in matching_images:
#     print(image_path)




