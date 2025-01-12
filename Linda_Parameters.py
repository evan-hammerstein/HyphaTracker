import cv2
import numpy as np
from skimage import io, color
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import math

# Load the grayscale image
image = cv2.imread('/Users/noahweiler/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Noah SWE Project/HyphaTracker/Skeletonized_image.png', cv2.IMREAD_GRAYSCALE)

# ========== IMAGE PROCESSING FUNCTIONS ==========

# Preprocess Image
def preprocess_image(image, crop_points=None):
    """
    Preprocess the image by cropping (optional with non-rectangular shape), 
    applying Otsu's thresholding, and binarizing the image.
    :param image: Grayscale image as a NumPy array.
    :param crop_points: Optional list of (x, y) points defining a polygon for cropping.
    :return: Binary image as a NumPy array (1 for foreground, 0 for background).
    """
    # Step 1: Create a mask for non-rectangular cropping
    if crop_points:
        # Create an empty mask
        mask = np.zeros_like(image, dtype=np.uint8)
        
        # Define the polygon and fill it on the mask
        polygon = np.array(crop_points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)  # Fill the polygon with white (255)
        
        # Apply the mask to the image
        image = cv2.bitwise_and(image, image, mask=mask)

    threshold = threshold_otsu(image)                                           # Compute optimal threshold using Otsu's method
    binary_image = image > threshold                                            # Binarize image using the threshold
    return binary_image.astype(np.uint8)                                        # Convert to uint8 for further processing

# Skeletonize Image
def skeletonize_image(binary_image):
    """
    Skeletonize a binary image to reduce structures to 1-pixel-wide lines.
    :param binary_image: Binary image as input.
    :return: Skeletonized binary image.
    """
    return skeletonize(binary_image > 0)  # Convert to boolean and skeletonize

# Remove small objects (e.g., spores or noise)
def filter_hyphae(binary_image, min_size=100):
    """
    Remove small connected components (e.g., spores or noise) to retain only large hyphae.
    :param binary_image: Binary image of the skeleton.
    :param min_size: Minimum size (in pixels) for connected components to retain.
    :return: Filtered binary image with small components removed.
    """
    labeled_image = label(binary_image)                                         # Label connected components in the image
    filtered_image = remove_small_objects(labeled_image, min_size=min_size)     # Remove small components
    return filtered_image > 0                                                   # Return as binary image (True for retained components)


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
def display_tips(binary_image, tips, frame_idx):
    """
    Display the skeleton image with tips marked as red dots and labeled with their coordinates.
    
    :param skeleton: Skeletonized binary image as a NumPy array.
    :param tips: List of (row, col) coordinates of tip positions.
    :param frame_idx: (Optional) The frame index to display in the title.
    """
    # Create a plot
    plt.figure(figsize=(10, 10))
    plt.imshow(binary_image, cmap='gray')  # Display the skeleton

    # Overlay red dots and labels for tips
    for idx, (y, x) in enumerate(tips):
        plt.scatter(x, y, c='red', s=0.5, label=f"Tip {idx+1}" if idx == 0 else None)  # Add red dot

    # Update title to include frame index if provided
    title = "Binary image with Tips"
    if frame_idx is not None:
        title += f" for Frame {frame_idx}"
    plt.title(title)  # Set the updated title

    # Hide axes
    plt.axis('off')

    # Display the image
    plt.show()


# Visualize tracked tips
def visualize_tracked_tips(tracked_tips, image_file, frame_idx):
    """
    Visualize tracked tips over time.
    
    :param tracked_tips: Dictionary of tracked tips.
    :param image_files: List of file paths to the PNG images.
    """
  
    # image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')
    
    # Overlay tracked tips
    for tip_id, positions in tracked_tips.items():
        for pos in positions:
            if pos[0] == frame_idx:  # Check if this tip exists in the current frame
                y, x = pos[1:]
                plt.scatter(x, y, c='red', s=0.5)
                plt.text(x + 2, y - 2, str(tip_id), color='yellow', fontsize=3)
    
    plt.title(f"Frame {frame_idx}")
    plt.axis('off')
    plt.show()




# ========== HYPHAL TIP DETECTION ==========

# Detect endpoints
def find_hyphal_endpoints(filtered_skeleton):
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
    convolved = convolve(filtered_skeleton.astype(int), kernel, mode='constant', cval=0)
    
    # Identify pixels with exactly one neighbor (endpoints)
    endpoints = np.argwhere((convolved == 11))
    
    # Filter endpoints to ensure they belong to large hyphae components
    labeled_skeleton = label(filtered_skeleton)  # Label connected components in the skeleton
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
    :return: Dictionary with tip IDs as keys and average growth rates as values, 
             and the general average growth rate for all tips.
    """
    average_growth_rates = {}
    total_growth_rates = []  # To store growth rates for all tips
    total_time = frame_interval * time_per_frame  # Total time for the specified frame interval

    for tip_id, positions in tracked_tips.items():
        growth_distances = []
        for i in range(len(positions) - frame_interval):
            # Get the positions separated by frame_interval
            _, y1, x1 = positions[i]
            _, y2, x2 = positions[i + frame_interval]
            
            # Calculate Euclidean distance
            distance = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            growth_rate = distance / total_time
            growth_distances.append(growth_rate)
            total_growth_rates.append(growth_rate)  # Add to the overall growth rates

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

def calculate_tip_size(binary_image, tip_position, radius_microns = 10):
    """
    Calculate the size of a single tip by counting the filled pixels within a specified radius.
    
    :param binary_image: Binary image as a NumPy array (1 for foreground, 0 for background).
    :param tip_position: Tuple (y, x) representing the position of the tip.
    :param radius_microns: Radius around the tip in microns.
    :param pixel_area: Area of a single pixel in µm².
    :return: Tip size in µm².
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

    y, x = tip_position
    radius_pixels = int(np.sqrt(radius_microns**2 / pixel_area))                # Convert radius from microns to pixels

    mask = np.zeros_like(binary_image, dtype=bool)                              # Create a circular mask for the ROI
    y_grid, x_grid = np.ogrid[:binary_image.shape[0], :binary_image.shape[1]]
    distance_from_tip = np.sqrt((y_grid - y)**2 + (x_grid - x)**2)
    mask[distance_from_tip <= radius_pixels] = True

    # Count filled pixels within the circular mask
    tip_pixels = np.sum(binary_image[mask])

    # Convert the count of pixels to area in microns squared
    tip_size = tip_pixels * pixel_area
    return tip_size


def track_tip_size_over_time(tracked_tips, binary_images, tip_id, radius_microns, pixel_area):
    """
    Track the size of a specific tip over time across multiple frames.
    
    :param tracked_tips: Dictionary with tip IDs as keys and lists of positions [(frame, y, x)] as values.
    :param binary_images: List of binary images (one per frame).
    :param tip_id: The ID of the tip to track.
    :param radius_microns: Radius around the tip in microns.
    :param pixel_area: Area of a single pixel in µm².
    :return: List of tip sizes (in µm²) over time.
    """
    if tip_id not in tracked_tips:
        raise ValueError(f"Tip ID {tip_id} not found in tracked tips.")
    
    tip_sizes = []  # To store the size of the tip in each frame
    tip_positions = tracked_tips[tip_id]  # Get the positions of the specified tip
    
    for frame_idx, (frame, y, x) in enumerate(tip_positions):
        # Get the binary image for the current frame
        binary_image = binary_images[frame]
        
        # Calculate the size of the tip in the current frame
        tip_size = calculate_tip_size(binary_image, (y, x), radius_microns, pixel_area)
        tip_sizes.append(tip_size)
    
    return tip_sizes



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




# ==========SPORES===========

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
    next_tip_id = 0  # Unique ID for each tip (Keys of dictionary)
    
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

# ========== Input Parameters ==========
import os

# Define the folder containing the images
folder_path = '/Users/noahweiler/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Noah SWE Project/Processed_images'

# Collect all `.tif` file paths from the folder
image_files = [
    os.path.join(folder_path, file) 
    for file in os.listdir(folder_path) 
    if file.endswith('.tif')
]


#SORTING FRAMES BASED ON FRAME NUMBER

# Sort the files based on the frame number extracted from the filenames
def extract_frame_number(file_path):
    file_name = os.path.basename(file_path)  # Get the file name
    # Extract the frame number using string splitting
    # Assuming filenames are in the format "processed_frame_<number>_timestamp.tif"
    parts = file_name.split('_')
    return int(parts[2])  # Frame number is the 3rd part (index 2)

# Sort using the extracted frame number
image_files.sort(key=extract_frame_number)



fov_1x = (1000, 1000)  # Field of view at 1x magnification in micrometers (width, height)
magnification = 10  # Magnification level
time_per_frame = 2  # Time difference between consecutive frames in seconds
frame_interval = 2  # Number of frames to calculate growth rates
distance_threshold = 15  # Distance threshold for tip matching
min_size_spores = 10  # Minimum size of spores
max_size_spores = 200  # Maximum size of spores
circularity_threshold = 0.7  # Circularity threshold for spores
roi = (200, 300)  # Example region of interest (y, x)

#========== Process Image Sequence ==========

#Preprocess images and extract hyphal tips
tip_positions_sequence = []
biomass_values = []
print(image_files[0])

for frame_idx, image_file in enumerate(image_files):
#     # Load the image
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # Preprocess and visualize
    binary_image = preprocess_image(image)

    skeleton = skeletonize_image(binary_image)

    filtered_skeleton = filter_hyphae(skeleton, min_size=50)

    # Find hyphal endpoints
    endpoints = find_hyphal_endpoints(filtered_skeleton)
    tip_positions_sequence.append(endpoints)

    # Calculate biomass
    biomass = find_biomass(binary_image, fov_1x, magnification)
    biomass_values.append(biomass)

    print(f'Execution for frame{frame_idx}')
    #Track hyphal tips across frames
    tracked_tips = track_tips_across_frames(tip_positions_sequence, distance_threshold)

    #display_tips(binary_image, endpoints, frame_idx)

    # Visualize tracked tips
    visualize_tracked_tips(tracked_tips, image_file, frame_idx)



# ========== Calculate Metrics ==========
# Calculate average growth rates for each tip
average_growth_rates, general_average_growth_rate = calculate_average_growth_rate(
    tracked_tips, frame_interval, time_per_frame
)
# Format the growth rates to 3 significant figures and add units (e.g., µm/s)
average_growth_rates = {tip_id: f"{rate:.3g} µm/s" for tip_id, rate in average_growth_rates.items()}
general_average_growth_rate = f"{general_average_growth_rate:.3g} µm/s"
print("Average Growth Rates for Each Tip:", average_growth_rates)
print("General Average Growth Rate:", general_average_growth_rate)

# Calculate distances to ROI for a specific tip
tip_id = 0  # Example tip ID
distances_to_roi = calculate_distances_to_roi(tracked_tips, tip_id, roi)
# Format the distances to 3 significant figures and add units (e.g., µm)
distances_to_roi = [f"{distance:.3g} µm" for distance in distances_to_roi]
print(f"Distances of Tip {tip_id} to ROI:", distances_to_roi)

# Calculate growth angles for a specific tip
growth_angles = calculate_growth_angles(tracked_tips, tip_id)
# Format the growth angles to 3 significant figures and add units (e.g., degrees)
growth_angles = [f"{angle:.3g}°" for angle in growth_angles]
print(f"Growth Angles for Tip {tip_id}:", growth_angles)

# Calculate branching rate
branching_events_per_frame, total_branching_events = calculate_branching_rate(
    tip_positions_sequence, distance_threshold
)
# Format branching events per frame (unitless as they are counts)
branching_events_per_frame = [f"{event:.3g}" for event in branching_events_per_frame]
total_branching_events = f"{total_branching_events:.3g}"
print("Branching Events Per Frame:", branching_events_per_frame)
print("Total Branching Events:", total_branching_events)

# ========== Find Spores ==========

spore_list=identify_spores(image, min_size_spores, max_size_spores, circularity_threshold)
print("Spores found: ", spore_list)

# ========== Track Spores ==========
spore_tracking = track_spores_over_time(
     image_files, min_size=min_size_spores, max_size=max_size_spores,
     circularity_threshold=circularity_threshold, distance_threshold=distance_threshold
)
# Format spore sizes to 3 significant figures and add units (e.g., µm² for area)
formatted_spore_tracking = {
    spore_id: [f"{size:.3g} µm²" for size in sizes]
    for spore_id, sizes in spore_tracking.items()
}
print("Spore Size Histories Over Time:", formatted_spore_tracking)

# ========== Biomass Analysis ==========
# Format biomass values to 3 significant figures and add units (e.g., µm² for area)
biomass_values = [f"{biomass:.3g} µm²" for biomass in biomass_values]
print("Biomass Over Time:", biomass_values)