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

# ========== IMAGE PROCESSING FUNCTIONS ==========

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

# Skeletonize Image
def skeletonize_image(binary_image):
    """
    Skeletonize a binary image to reduce structures to 1-pixel-wide lines.
    :param binary_image: Binary image as input.
    :return: Skeletonized binary image.
    """
    return skeletonize(binary_image > 0)  # Convert to boolean and skeletonize

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

# ========== VISUALIZATION FUNCTIONS ==========

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

# ========== HYPHAL TIP DETECTION ==========

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

# ========== HYPHAL METRICS ==========

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

#DISTANCE TO REGIONS OF INTEREST
# Example: Regions of interest (e.g., spore centroids)
roi = [(100, 200), (150, 300)]  # Example coordinates

def calculate_distances_to_roi(tracked_tips, tip_id, roi):
    """
    Calculate the distances of a specific hyphal tip to a defined region of interest (ROI) over time.
    
    :param tracked_tips: Dictionary with tip IDs as keys and lists of positions [(frame, y, x)] as values.
    :param tip_id: The ID of the tip for which distances should be calculated.
    :param roi: Tuple (y, x) defining the region of interest.
    :return: List of distances to the ROI for the specified tip over time.
    """
    if tip_id not in tracked_tips:
        raise ValueError(f"Tip ID {tip_id} not found in tracked tips.")
    
    distances = []
    for _, y, x in tracked_tips[tip_id]:
        # Calculate the Euclidean distance to the ROI
        distance = np.sqrt((y - roi[0])**2 + (x - roi[1])**2)
        distances.append(distance)
    
    return distances

#TIP GROWTH RATE

def calculate_average_growth_rate(tracked_tips, frame_interval, time_per_frame):
    """
    Calculate the average growth rate of hyphal tips over a specified number of frames.
    
    :param tracked_tips: Dictionary with tip IDs as keys and lists of positions [(frame, y, x)] as values.
    :param frame_interval: Number of frames over which to calculate the growth rate.
    :param time_per_frame: Time difference between consecutive frames.
    :return: Dictionary with tip IDs as keys and average growth rates as values.
    """
    average_growth_rates = {}
    total_time = frame_interval * time_per_frame  # Total time for the specified frame interval

    for tip_id, positions in tracked_tips.items():
        growth_distances = []
        for i in range(len(positions) - frame_interval):
            # Get the positions separated by frame_interval
            _, y1, x1 = positions[i]
            _, y2, x2 = positions[i + frame_interval]
            
            # Calculate Euclidean distance
            distance = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            growth_distances.append(distance / total_time)
        
        # Calculate the average growth rate for the tip
        if growth_distances:
            average_growth_rate = sum(growth_distances) / len(growth_distances)
        else:
            average_growth_rate = 0  # If no valid growth distances are found

        average_growth_rates[tip_id] = average_growth_rate

        # Calculate the general average growth rate
        if total_growth_rates:
            general_average_growth_rate = sum(total_growth_rates) / len(total_growth_rates)
        else:
            general_average_growth_rate = 0

    return average_growth_rates, general_average_growth_rate


#TIP GROWTH ANGLE
import math

def calculate_growth_angles(tracked_tips, tip_id):
    """
    Calculate the growth angles of a specific hyphal tip over time.
    
    :param tracked_tips: Dictionary with tip IDs as keys and lists of positions [(frame, y, x)] as values.
    :param tip_id: The ID of the tip for which growth angles should be calculated.
    :return: List of growth angles (in degrees) for the specified tip over time.
    """
    if tip_id not in tracked_tips:
        raise ValueError(f"Tip ID {tip_id} not found in tracked tips.")
    
    positions = tracked_tips[tip_id]  # Get the positions of the specified tip
    growth_angles = []  # List to store growth angles
    
    for i in range(1, len(positions)):
        _, y1, x1 = positions[i - 1]
        _, y2, x2 = positions[i]
        
        # Compute differences
        delta_x = x2 - x1
        delta_y = y2 - y1
        
        # Calculate angle in radians and convert to degrees
        angle_radians = math.atan2(delta_y, delta_x)
        angle_degrees = math.degrees(angle_radians)
        
        growth_angles.append(angle_degrees)
    
    return growth_angles

# ========== MYCELIAL METRICS ==========

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

#NUMBER/SIZE/DISTRIBUTION OF SPORES (SPHERICAL STRUCTURES)
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

# ========== SEQUENCE PROCESSING ==========

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

# ========== MAIN EXECUTION ==========

# Example usage (customize parameters as needed)
binary_image = preprocess_image(image)
show_image(binary_image, title='Post-processing Binary Image')

skeleton = skeletonize_image(binary_image)
show_image(skeleton, title='Skeletonized Image')

filtered_skeleton = filter_hyphae(skeleton, min_size=50)
show_image(filtered_skeleton, title='Filtered Hyphae Skeleton')

endpoints = find_hyphal_endpoints(filtered_skeleton)
print("Amount of hyphal tip positions is:", len(endpoints))
print("Hyphal Tip Positions:", endpoints)

display_tips(filtered_skeleton, endpoints)