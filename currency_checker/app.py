from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Create a Flask application instance
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Secret key for session management (used for flash messages)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the upload folder if it doesn't exist
os.makedirs(STATIC_FOLDER, exist_ok=True)  # Create the static folder if it doesn't exist

# Configure Matplotlib to use the 'Agg' backend
plt.switch_backend('Agg')

# Image processing functions
def preprocess_image(image_path, target_size=(500, 500)):
    # Load the image from the given path
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    # Resize the image while preserving aspect ratio
    height, width, _ = image.shape
    if height > width:
        new_height = target_size[0]
        new_width = int(width * (new_height / height))
    else:
        new_width = target_size[1]
        new_height = int(height * (new_width / width))

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Pad the resized image to reach the target size
    top_pad = (target_size[0] - new_height) // 2
    bottom_pad = target_size[0] - new_height - top_pad
    left_pad = (target_size[1] - new_width) // 2
    right_pad = target_size[1] - new_width - left_pad

    padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT,
                                      value=0)
    return padded_image

def detect_edges(image):
    # Convert the image to grayscale and detect edges using Canny
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return edges

def compare_color_histograms(image1, image2, bins=256):
    # Calculate and normalize color histograms for each channel in the BGR color space
    hist1 = [cv2.calcHist([image1], [i], None, [bins], [0, 256]) for i in range(3)]
    hist2 = [cv2.calcHist([image2], [i], None, [bins], [0, 256]) for i in range(3)]
    hist1 = [cv2.normalize(h, h) for h in hist1]
    hist2 = [cv2.normalize(h, h) for h in hist2]

    # Return histograms and dummy similarity (we'll compute similarity separately)
    return hist1, hist2, 0  # 0 is a dummy value for similarity

def plot_histograms(hist1, hist2):
    # Plot and save histograms
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        plt.plot(hist1[i], color=color, linestyle='dashed', label=f'Standard {color.upper()}')
        plt.plot(hist2[i], color=color, label=f'Test {color.upper()}')
        plt.xlim([0, 256])

    plt.title('Color Histograms')
    plt.xlabel('Pixel Value')
    plt.ylabel('Normalized Frequency')
    plt.legend()

    hist_path = os.path.join(STATIC_FOLDER, 'histogram.png')
    plt.savefig(hist_path)
    plt.clf()  # Clear the figure for the next plot
    return hist_path

def plot_edges(standard_edges, test_edges):
    # Plot and save edge images
    edge_standard_path = os.path.join(STATIC_FOLDER, 'standard_edges.png')
    edge_test_path = os.path.join(STATIC_FOLDER, 'test_edges.png')

    plt.imshow(standard_edges, cmap='gray')
    plt.title('Standard Image Edges')
    plt.axis('off')
    plt.savefig(edge_standard_path)
    plt.clf()

    plt.imshow(test_edges, cmap='gray')
    plt.title('Test Image Edges')
    plt.axis('off')
    plt.savefig(edge_test_path)
    plt.clf()

    return edge_standard_path, edge_test_path

def compare_with_standard(standard, test, color_threshold=0.85, edge_threshold=0.80):
    # Compare color histograms using Chi-Square instead of Correlation
    hist1, hist2, _ = compare_color_histograms(standard, test)
    color_similarity = sum(cv2.compareHist(hist1[i], hist2[i], cv2.HISTCMP_CHISQR) for i in range(3)) / 3
    print(f"Color similarity (Chi-Square): {color_similarity}")
    # color_similarity_before_normalized = color_similarity

    # Normalize the similarity to be between 0 and 1 (lower is better for Chi-Square)
    color_similarity = 1 - (color_similarity / (2 * color_threshold))
    color_similarity = max(0, color_similarity)  # Ensure it's non-negative
    print(f"Normalized color similarity: {color_similarity}")

    # Plot and save histograms
    hist_path = plot_histograms(hist1, hist2)

    # Compare edges
    standard_edges = detect_edges(standard)
    test_edges = detect_edges(test)
    edge_similarity = np.sum(standard_edges == test_edges) / standard_edges.size
    print(f"Edge similarity: {edge_similarity}")

    # Plot and save edge images
    edge_standard_path, edge_test_path = plot_edges(standard_edges, test_edges)

    return color_similarity, edge_similarity, hist_path, edge_standard_path, edge_test_path

# Define a route for the root URL
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'standard' not in request.files or 'test' not in request.files:
            flash('No file part')
            return redirect(request.url)

        standard_file = request.files['standard']
        test_file = request.files['test']

        if standard_file.filename == '' or test_file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        standard_path = os.path.join(UPLOAD_FOLDER, 'standard.jpg')
        test_path = os.path.join(UPLOAD_FOLDER, 'test.jpg')
        standard_file.save(standard_path)
        test_file.save(test_path)

        try:
            standard_note = preprocess_image(standard_path, target_size=(500, 500))
            test_note = preprocess_image(test_path, target_size=(500, 500))
            color_similarity, edge_similarity, hist_path, edge_standard_path, edge_test_path = compare_with_standard(
                standard_note, test_note)



            hist1, hist2, raw_color_similarity = compare_color_histograms(standard_note, test_note)
            raw_color_similarity = sum(cv2.compareHist(hist1[i], hist2[i], cv2.HISTCMP_CHISQR) for i in range(3)) / 3



            message = "The currency is genuine." if color_similarity > 0.90 and edge_similarity > 0.85 else "The currency is fake."
        except Exception as e:
            flash(f"An error occurred: {e}")
            return redirect(request.url)

        flash(message)
        return render_template('index.html', color_similarity=color_similarity, raw_color_similarity=raw_color_similarity, edge_similarity=edge_similarity,
                               hist_path=hist_path, edge_standard_path=edge_standard_path,
                               edge_test_path=edge_test_path, message=message)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
