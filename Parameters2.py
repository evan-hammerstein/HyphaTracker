import cv2
import numpy as np
from skimage import io, color
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

# Load the grayscale image
image = cv2.imread('/Users/lindaschermeier/Desktop/Skel_Im.jpg', cv2.IMREAD_GRAYSCALE)

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

# Display Skeleton with Tips and Labels
def display_tips(skeleton, tips):
    """
    Display the skeleton image with tips marked as red dots and labeled with their coordinates.
    
    :param skeleton: Skeletonized binary image as a NumPy array.
    :param tips: List of (row, col) coordinates of tip positions.
    """
    # Create a plot
    plt.figure(figsize=(10, 10))
    plt.imshow(skeleton, cmap='gray')  # Display the skeleton

    # Overlay red dots and labels for tips
    for idx, (y, x) in enumerate(tips):
        plt.scatter(x, y, c='red', s=50, label=f"Tip {idx+1}" if idx == 0 else None)  # Add red dot
        plt.text(x + 2, y - 2, f"({y}, {x})", color='red', fontsize=8)  # Add label next to the dot

    # Add title and hide axes
    plt.title("Skeleton with Tips and Coordinates")
    plt.axis('off')

    # Display the image
    plt.show()

display_tips(filtered_skeleton, endpoints) # Display skeleton with tips and labels

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

# sequence of pngs - compare old to new and find difference

from scipy.spatial.distance import cdist

# Match tips between frames and handle branching
def track_tips_across_frames(tip_positions, distance_threshold=15):
    """
    Track hyphal tips across frames, creating separate lists for new branches.
    
    :param tip_positions: List of tip positions for each frame (list of lists of (y, x) tuples).
    :param distance_threshold: Maximum distance to consider two tips as the same.
    :return: Dictionary with keys as tip IDs and values as lists of positions [(frame, y, x)].
    """
    tracked_tips = {}  # Dictionary to store tip tracking {tip_id: [(frame, y, x)]}
    next_tip_id = 0  # Unique ID for each tip
    
    # Iterate over frames
    for frame_idx, current_tips in enumerate(tip_positions):
        if frame_idx == 0:
            # Initialize tracking for the first frame
            for tip in current_tips:
                tracked_tips[next_tip_id] = [(frame_idx, *tip)]
                next_tip_id += 1
            continue

        # Match tips to the previous frame
        previous_tips = [positions[-1][1:] for positions in tracked_tips.values()]
        distances = cdist(previous_tips, current_tips)  # Compute distances between tips
        
        # Match previous tips to current tips
        matched_current = set()
        for i, prev_tip in enumerate(previous_tips):
            # Find the nearest current tip within the distance threshold
            nearest_idx = np.argmin(distances[i])
            if distances[i, nearest_idx] < distance_threshold:
                tracked_tips[i].append((frame_idx, *current_tips[nearest_idx]))
                matched_current.add(nearest_idx)
            else:
                # Terminate the tip if no match is found
                continue

        # Add new tips that were not matched
        for j, current_tip in enumerate(current_tips):
            if j not in matched_current:
                tracked_tips[next_tip_id] = [(frame_idx, *current_tip)]
                next_tip_id += 1

    return tracked_tips


# Process a sequence of images and track tips
def process_sequence(image_files, min_size=50, distance_threshold=15): # adjust threshold as needed after testing
    """
    Process a sequence of images and track hyphal tips over time.
    
    :param image_files: List of file paths to the PNG images.
    :param min_size: Minimum size for connected components (filtering small noise).
    :param distance_threshold: Maximum distance to consider two tips as the same.
    :return: Dictionary of tracked tips.
    """
    tip_positions = []  # List to store tip positions for each frame
    
    for file in image_files:
        # Load and preprocess the image
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        binary_image = preprocess_image(image)
        skeleton = skeletonize_image(binary_image)
        filtered_skeleton = filter_hyphae(skeleton, min_size=min_size)
        
        # Find tips in the current frame
        tips = find_hyphal_endpoints(filtered_skeleton)
        tip_positions.append(tips)
    
    # Track tips across frames
    tracked_tips = track_tips_across_frames(tip_positions, distance_threshold)
    
    return tracked_tips


# Visualize tracked tips
def visualize_tracked_tips(tracked_tips, image_files):
    """
    Visualize tracked tips over time.
    
    :param tracked_tips: Dictionary of tracked tips.
    :param image_files: List of file paths to the PNG images.
    """
    for frame_idx, file in enumerate(image_files):
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        plt.imshow(image, cmap='gray')
        
        # Overlay tracked tips
        for tip_id, positions in tracked_tips.items():
            for pos in positions:
                if pos[0] == frame_idx:  # Check if this tip exists in the current frame
                    y, x = pos[1:]
                    plt.scatter(x, y, c='red', s=50)
                    plt.text(x + 2, y - 2, str(tip_id), color='yellow', fontsize=8)
        
        plt.title(f"Frame {frame_idx}")
        plt.axis('off')
        plt.show()


def find_biomass(binary_image, fov_1x, magnification):
    """
    Calculate the biomass (physical area) of the fungal structure in the binary image.
    
    :param binary_image: Binary image as a NumPy array (1 for foreground, 0 for background).
    :param fov_1x: Field of View at 1x magnification (width, height) in micrometers.
    :param magnification: Magnification level of the image.
    :return: Biomass in micrometers squared.
    """
    # Image dimensions
    image_height, image_width = binary_image.shape

    # Calculate the FOV at the given magnification
    fov_width = fov_1x[0] / magnification  # µm
    fov_height = fov_1x[1] / magnification  # µm

    # Calculate pixel dimensions
    pixel_width = fov_width / image_width  # µm per pixel
    pixel_height = fov_height / image_height  # µm per pixel

    # Calculate pixel area in micrometers squared
    pixel_area = pixel_width * pixel_height  # µm² per pixel

    # Calculate biomass (number of foreground pixels * pixel area)
    biomass_pixels = np.sum(binary_image)  # Count the number of white pixels
    biomass_area = biomass_pixels * pixel_area  # Total biomass in µm²

    return biomass_area

def calculate_biomass_over_time(image_files, fov_1x, magnification):
    """
    Calculate biomass over time for a sequence of images.
    
    :param image_files: List of file paths to the PNG images.
    :param fov_1x: Field of View at 1x magnification (width, height) in micrometers.
    :param magnification: Magnification level of the images.
    :return: List of biomass values (one for each frame).
    """
    biomass_values = []

    for file in image_files:
        # Load and preprocess the image
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        binary_image = preprocess_image(image)
        
        # Calculate biomass
        biomass = find_biomass(binary_image, fov_1x, magnification)
        biomass_values.append(biomass)

    return biomass_values


from scipy.spatial.distance import cdist
import numpy as np

def calculate_branching_rate(tip_positions, distance_threshold=15):
    """
    Calculate the branching rate/frequency of fungal hyphae over time.

    :param tip_positions: List of lists of (y, x) tip positions for each frame.
    :param distance_threshold: Maximum distance to consider tips as originating from the same source.
    :return: List of branching events per frame and total branching events.
    """
    branching_events_per_frame = []  # List to store branching events for each frame
    total_branching_events = 0  # Total number of branching events

    # Iterate over consecutive frames
    for frame_idx in range(1, len(tip_positions)):
        current_tips = tip_positions[frame_idx]  # Tips in the current frame
        previous_tips = tip_positions[frame_idx - 1]  # Tips in the previous frame

        if not previous_tips or not current_tips:
            branching_events_per_frame.append(0)
            continue

        # Calculate distances between previous and current tips
        distances = cdist(previous_tips, current_tips)

        # For each tip in the previous frame, count the number of associated tips in the current frame
        branching_events = 0
        for i, _ in enumerate(previous_tips):
            # Find indices of current tips within the distance threshold
            matching_tips = np.where(distances[i] < distance_threshold)[0]
            
            # If there are more than one matching tip, it indicates branching
            if len(matching_tips) > 1:
                branching_events += len(matching_tips) - 1  # Count new branches

        # Update the branching events
        branching_events_per_frame.append(branching_events)
        total_branching_events += branching_events

    return branching_events_per_frame, total_branching_events



#NUMBER/SIZE/DISTRIBUTION OF SPORES (SPHEREICAL STRUCTURES)
from scipy.spatial.distance import cdist

def track_spores_over_time(image_files, min_size=10, max_size=200, circularity_threshold=0.7, distance_threshold=15):
    """
    Track spores over time across a sequence of images and output their sizes over time.
    
    :param image_files: List of file paths to the PNG images.
    :param min_size: Minimum size of objects to consider as spores.
    :param max_size: Maximum size of objects to consider as spores.
    :param circularity_threshold: Minimum circularity to consider an object as a spore.
    :param distance_threshold: Maximum distance to match spores between frames.
    :return: Dictionary of tracked spores with their sizes over time.
    """
    def identify_spores(image, min_size, max_size, circularity_threshold):
        """
        Identify spores in the image based on size and circularity.
        """
        # Preprocess the image
        threshold = threshold_otsu(image)
        binary_image = (image > threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        spores = []

        # Analyze each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if min_size <= area <= max_size and perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                if circularity >= circularity_threshold:
                    (x, y), _ = cv2.minEnclosingCircle(contour)  # Center of spore
                    spores.append({"center": (int(x), int(y)), "size": area})

        return spores

    # Dictionary to store tracked spores: {spore_id: {"history": [(frame_idx, size)], "last_position": (x, y)}}
    tracked_spores = {}
    next_spore_id = 0

    # Process each frame
    for frame_idx, file in enumerate(image_files):
        # Load the image
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"File not found: {file}")

        # Identify spores in the current frame
        current_spores = identify_spores(image, min_size, max_size, circularity_threshold)

        if frame_idx == 0:
            # Initialize tracking for the first frame
            for spore in current_spores:
                tracked_spores[next_spore_id] = {
                    "history": [(frame_idx, spore["size"])],
                    "last_position": spore["center"],
                }
                next_spore_id += 1
            continue

        # Match spores to those in the previous frame
        previous_positions = [data["last_position"] for data in tracked_spores.values()]
        current_positions = [spore["center"] for spore in current_spores]

        if previous_positions and current_positions:
            distances = cdist(previous_positions, current_positions)

            matched_current = set()
            for spore_id, prev_position in enumerate(previous_positions):
                # Find the nearest current spore
                nearest_idx = np.argmin(distances[spore_id])
                if distances[spore_id, nearest_idx] < distance_threshold:
                    # Update the spore's history and last position
                    tracked_spores[spore_id]["history"].append((frame_idx, current_spores[nearest_idx]["size"]))
                    tracked_spores[spore_id]["last_position"] = current_spores[nearest_idx]["center"]
                    matched_current.add(nearest_idx)

            # Add new spores that were not matched
            for j, spore in enumerate(current_spores):
                if j not in matched_current:
                    tracked_spores[next_spore_id] = {
                        "history": [(frame_idx, spore["size"])],
                        "last_position": spore["center"],
                    }
                    next_spore_id += 1

    # Extract the size history for each spore
    spore_size_histories = {spore_id: [entry[1] for entry in data["history"]] for spore_id, data in tracked_spores.items()}
    return spore_size_histories

