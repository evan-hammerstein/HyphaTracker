import cv2
import numpy as np
from skimage import io, color
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

# Load the grayscale image
image = cv2.imread('/Users/noahweiler/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Noah SWE Project/HyphaTracker/Skeletonized_image.png', cv2.IMREAD_GRAYSCALE)

# Preprocess Image
def preprocess_image(image):
    """
    Preprocess the image by applying Otsu's thresholding and binarizing the image.
    :param image: Grayscale image as a NumPy array.
    :return: Binary image as a NumPy array (1 for foreground, 0 for background).
    """
    threshold = threshold_otsu(image)  # Compute optimal threshold using Otsu's method
    binary_image = image > threshold  # Binarize image using the threshold
    return binary_image.astype(np.uint8)  # Convert to uint8 for further processing

# Display Image
def show_image(image, title='Image'):
    """
    Display the given image using matplotlib.
    :param image: Image to display.
    :param title: Title of the image window.
    """
    plt.imshow(image, cmap='gray')  # Display image in grayscale
    plt.title(title)  # Set the title of the plot
    plt.axis('off')  # Hide the axes for better visualization
    plt.show()

# Skeletonize Image
def skeletonize_image(binary_image):
    """
    Skeletonize a binary image to reduce structures to 1-pixel-wide lines.
    :param binary_image: Binary image as input.
    :return: Skeletonized binary image.
    """
    skeleton = skeletonize(binary_image > 0)  # Convert to boolean and skeletonize
    return skeleton

# Remove small objects (e.g., spores or noise)
def filter_hyphae(binary_image, min_size=50):
    """
    Remove small connected components (e.g., spores or noise) to retain only large hyphae.
    :param binary_image: Binary image of the skeleton.
    :param min_size: Minimum size (in pixels) for connected components to retain.
    :return: Filtered binary image with small components removed.
    """
    labeled_image = label(binary_image)  # Label connected components in the image
    filtered_image = remove_small_objects(labeled_image, min_size=min_size)  # Remove small components
    return filtered_image > 0  # Return as binary image (True for retained components)

# Detect endpoints
def find_hyphal_endpoints(skeleton):
    """
    Detect endpoints of hyphae by identifying pixels with exactly one connected neighbor.
    :param skeleton: Skeletonized binary image.
    :return: List of (y, x) coordinates of detected endpoints.
    """
    # Define a 3x3 kernel to identify pixels with exactly one neighbor
    kernel = np.array([[1, 1, 1], 
                       [1, 10, 1], 
                       [1, 1, 1]])
    
    # Convolve the kernel with the skeleton to count neighbors for each pixel
    convolved = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
    
    # Identify pixels with exactly one neighbor (endpoints)
    endpoints = np.argwhere((convolved == 11))
    
    # Filter endpoints to ensure they belong to large hyphae components
    labeled_skeleton = label(skeleton)  # Label connected components in the skeleton
    valid_endpoints = []  # Initialize list to store valid endpoints
    for y, x in endpoints:
        if labeled_skeleton[y, x] > 0:  # Check if endpoint belongs to a labeled component
            valid_endpoints.append((y, x))  # Add valid endpoint to the list
    return valid_endpoints

# Process and visualize the image
binary_image = preprocess_image(image)  # Preprocess the image
show_image(binary_image, title='Post-processing Binary Image')  # Display the binary image

skeleton = skeletonize_image(binary_image)  # Skeletonize the binary image
show_image(skeleton, title='Skeletonized Image')  # Display the skeletonized image

filtered_skeleton = filter_hyphae(skeleton, min_size=50)  # Filter small components (spores/noise)
show_image(filtered_skeleton, title='Filtered Hyphae Skeleton')  # Display the filtered skeleton

endpoints = find_hyphal_endpoints(filtered_skeleton)  # Detect hyphal endpoints
print("Amount of hyphal tip positions is:", len(endpoints))  # Print the number of detected endpoints
print("Hyphal Tip Positions:", endpoints)  # Print the coordinates of the detected endpoints

print('Hello world')


#DISTANCE TO REGIONS OF INTEREST
# Example: Regions of interest (e.g., spore centroids)
regions_of_interest = [(100, 200), (150, 300)]  # Example coordinates
distances = []

for tip in endpoints:
    distances.append([distance.euclidean(tip, roi) for roi in regions_of_interest])

print("Distances from Hyphal Tips to Regions of Interest:", distances)

#TIP GROWTH RATE
# Assuming tip_positions_t1 and tip_positions_t2 are lists of tip positions at times t1 and t2
growth_rates = []
for tip_t1, tip_t2 in zip(tip_positions_t1, tip_positions_t2):
    growth_rate = distance.euclidean(tip_t1, tip_t2) / time_interval
    growth_rates.append(growth_rate)

print("Hyphal Tip Growth Rates:", growth_rates)


#TIP GROWTH ANGLE
# Example: Tip positions at t1 and t2
for tip_t1, tip_t2 in zip(tip_positions_t1, tip_positions_t2):
    dx = tip_t2[1] - tip_t1[1]
    dy = tip_t2[0] - tip_t1[0]
    angle = math.degrees(math.atan2(dy, dx))
    print("Growth Angle:", angle)


#TIP AREA/SHAPE
# Circular region around the tip
radius = 10  # Example: 10 microns
def count_pixels_around_tip(binary_image, tip, radius):
    y, x = tip
    pixels = binary_image[max(0, y-radius):y+radius, max(0, x-radius):x+radius]
    y_grid, x_grid = np.ogrid[-radius:radius, -radius:radius]
    mask = x_grid**2 + y_grid**2 <= radius**2
    return np.sum(pixels[mask])

for tip in endpoints:
    print("Pixels near tip:", count_pixels_around_tip(binary_image, tip, radius))



#MYCELIAL METRICS

#BRANCHING RATE/FREQUENCY
def find_branch_points(skeleton):
    from scipy.ndimage import convolve
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    convolved = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
    return np.argwhere((convolved > 12))  # More than two neighbors

branch_points = find_branch_points(skeleton)
print("Branch Points:", branch_points)
print("Branching Frequency:", len(branch_points))


#BIOMASS
biomass = np.sum(binary_image)
print("Biomass (pixel count):", biomass)


#NUMBER/SIZE/DISTRIBUTION OF SPORES (SPHEREICAL STRUCTURES)
# Detect circular structures
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
spores = [cv2.minEnclosingCircle(c) for c in contours if len(c) > 5]  # Only circular regions

for spore in spores:
    center, radius = spore
    print("Spore Center:", center, "Radius:", radius)
print("Total Spores:", len(spores))

print("Hello")