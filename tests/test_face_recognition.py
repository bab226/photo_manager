import os
import sys
import unittest

# Ensure helper_functions.py is reachable by adding its directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import helper_functions as hf

class TestFaceRecognition(unittest.TestCase):

    def test_preprocess_image(self):
        # Test preprocessing function with a sample image path
        image_path = os.path.join(os.path.dirname(__file__), 'test_images', 'image1.jpg')
        preprocessed_image = hf.preprocess_image(image_path, contrast_factor=1.5)
        self.assertIsNotNone(preprocessed_image, "Preprocessing failed, got None")

    def test_extract_encodings(self):
        # Test encoding extraction
        image_paths_directory = os.path.join(os.path.dirname(__file__), 'test_images')
        images_with_paths = hf.load_and_process_images(image_paths_directory)
        encodings_with_paths = hf.extract_encodings_with_paths(images_with_paths)
        self.assertTrue(len(encodings_with_paths) > 0, "No encodings extracted")

    def test_clustering(self):
        # Test clustering function
        image_paths_directory = os.path.join(os.path.dirname(__file__), 'test_images')
        images_with_paths = hf.load_and_process_images(image_paths_directory)
        encodings_with_paths = hf.extract_encodings_with_paths(images_with_paths)
        all_encodings = [encoding for _, encoding in encodings_with_paths]
        self.assertGreater(len(all_encodings), 1, "Not enough samples for clustering")
        
        optimal_k = hf.find_optimal_cluster_silhouette(all_encodings, max_k=5)
        hf.validate_clusters(all_encodings, optimal_k)
        self.assertTrue(optimal_k > 0, "Optimal clusters could not be determined")

if __name__ == "__main__":
    unittest.main()
