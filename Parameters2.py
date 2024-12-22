import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
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

# Detect Hyphal Tips
def find_hyphal_tips(skeleton):
    """
    Detect the tips (endpoints) of hyphae in the skeletonized image.
    :param skeleton: Skeletonized binary image as a NumPy array.
    :return: List of (row, col) coordinates for hyphal tips.
    """
    # Define a 3x3 kernel to count neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    
    # Convolve the skeleton with the kernel
    convolved = convolve(skeleton, kernel, mode='constant', cval=0)
    
    # Detect endpoints: Pixel value = 11 (10 for itself + 1 neighbor)
    tips = np.argwhere(convolved == 11)
    
    return tips

# Load Image
image = cv2.imread('/Users/lindaschermeier/Desktop/Skel_Im.jpg', cv2.IMREAD_GRAYSCALE)  # Load in grayscale

# Check if the image is loaded successfully
if image is None:
    raise FileNotFoundError("The image file was not found. Check the file path.")

# Preprocess and Visualize
binary_image = preprocess_image(image)
show_image(binary_image, title='Binary Image')

# Skeletonize the Image
skeleton = skeletonize(binary_image > 0)
show_image(skeleton, title='Skeletonized Image')

# Find Hyphal Tips
tips = find_hyphal_tips(skeleton)

# Display Tips on the Image
skeleton_with_tips = skeleton.copy()
for (row, col) in tips:
    skeleton_with_tips[row, col] = 2  # Mark tips (value > 1 for visualization)

show_image(skeleton_with_tips, title='Skeleton with Tips')

# Print Tip Coordinates
print("Hyphal Tip Positions (row, col):")
for tip in tips:
    print(tip)
