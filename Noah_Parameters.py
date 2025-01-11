import cv2
import numpy as np
from skimage import io, color
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import math



image = cv2.imread('/Users/noahweiler/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Noah SWE Project/HyphaTracker/Skeletonized_image.png', cv2.IMREAD_GRAYSCALE)  # Load in grayscale


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
binary_image = preprocess_image(image)
show_image(binary_image, title='Binary Image')


#Skeletonizing the image for further processing into essentially a binary image where black is 0 and white is 1
skeleton = skeletonize(binary_image > 0)
show_image(skeleton, title='Skeletonized Image')

#TIP POSITION
# Detect endpoints by counting connected neighbors
def find_endpoints(skeleton):
    from scipy.ndimage import convolve
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    convolved = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
    return np.argwhere((convolved == 11))  # Only one neighbor

endpoints = find_endpoints(skeleton)
print("Hyphal Tip Positions:", endpoints)


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
