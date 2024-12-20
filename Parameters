#Parameters Python file

from OpenCV import cv2
import numpy as np
from skimage import io, color
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


# Preprocess Image
def preprocess_image(image):
    """
    Preprocess the image by applying a threshold and binarizing.
    :param image: Grayscale image as a NumPy array.
    :return: Binary image as a NumPy array.
    """
    threshold = threshold_otsu(image)
    binary_image = image > threshold
    return binary_image.astype(np.uint8)

# Display Image
def show_image(image, title='Image'):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load, preprocess, and visualize
image = load_image('/path/to/image.png', grayscale=True)
binary_image = preprocess_image(image)
show_image(binary_image, title='Binary Image')


image = cv2.imread('/Users/noahweiler/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Y3/SWE/SWE project/Skeletonized_image.png', cv2.IMREAD_GRAYSCALE)  # Load in grayscale

#Skeletonizing the image for further processing into essentially a binary image where black is 0 and white is 1
skeleton = skeletonize(binary_image > 0)
show_image(skeleton, title='Skeletonized Image')
