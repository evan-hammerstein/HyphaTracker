import cv2
import numpy as np
from skimage import io, color
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import math
import os

# Load the grayscale image
image = cv2.imread('/Users/noahweiler/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Noah SWE Project/HyphaTracker/Skeletonized_image.png', cv2.IMREAD_GRAYSCALE)

# Ensure the graphs folder exists
# graphs_folder = "graphs"
# os.makedirs(graphs_folder, exist_ok=True)
# print(f"Graphs folder created: {graphs_folder}")

# # ========== Graph Functions ==========
# def save_growth_rate_graph(average_growth_rates, output_folder):
#     """Generate and save a graph for growth rates per frame."""
#     tip_ids = list(average_growth_rates.keys())
#     rates = list(average_growth_rates.values())

#     plt.figure(figsize=(10, 6))
#     plt.bar(tip_ids, rates, color='blue')
#     plt.xlabel("Tip ID")
#     plt.ylabel("Growth Rate (µm/s)")
#     plt.title("Growth Rate per Tip")
#     plt.tight_layout()

#     graph_path = os.path.join(output_folder, "growth_rate_per_tip.png")
#     plt.savefig(graph_path)
#     plt.close()
#     print(f"Growth rate graph saved to {graph_path}")

# def save_growth_angle_graph(growth_angles, tip_id, output_folder):
#     """Generate and save a graph for growth angles per frame for a specific tip."""
#     frames = range(1, len(growth_angles) + 1)

#     plt.figure(figsize=(10, 6))
#     plt.plot(frames, growth_angles, marker='o', color='orange')
#     plt.xlabel("Frame")
#     plt.ylabel("Growth Angle (°)")
#     plt.title(f"Growth Angles for Tip {tip_id}")
#     plt.tight_layout()

#     graph_path = os.path.join(output_folder, f"growth_angles_tip_{tip_id}.png")
#     plt.savefig(graph_path)
#     plt.close()
#     print(f"Growth angles graph for Tip {tip_id} saved to {graph_path}")

# def save_tip_size_graph(tip_sizes, tip_id, output_folder):
#     """Generate and save a graph for tip sizes over frames for a specific tip."""
#     frames = [frame for frame, _ in tip_sizes]
#     sizes = [size for _, size in tip_sizes]

#     plt.figure(figsize=(10, 6))
#     plt.plot(frames, sizes, marker='o', color='green')
#     plt.xlabel("Frame")
#     plt.ylabel("Tip Size (µm²)")
#     plt.title(f"Tip Size Over Frames for Tip {tip_id}")
#     plt.tight_layout()

#     graph_path = os.path.join(output_folder, f"tip_size_tip_{tip_id}.png")
#     plt.savefig(graph_path)
#     plt.close()
#     print(f"Tip size graph for Tip {tip_id} saved to {graph_path}")

# def save_biomass_graph(biomass_values, output_folder):
#     """Generate and save a graph for biomass over frames."""
#     frames = range(len(biomass_values))

#     plt.figure(figsize=(10, 6))
#     plt.plot(frames, biomass_values, marker='o', color='purple')
#     plt.xlabel("Frame")
#     plt.ylabel("Biomass (µm²)")
#     plt.title("Biomass Over Frames")
#     plt.tight_layout()

#     graph_path = os.path.join(output_folder, "biomass_over_frames.png")
#     plt.savefig(graph_path)
#     plt.close()
#     print(f"Biomass graph saved to {graph_path}")

# def save_branching_rate_graph(branching_events_per_frame, output_folder):
#     """Generate and save a graph for branching rate per frame."""
#     frames = range(len(branching_events_per_frame))

#     plt.figure(figsize=(10, 6))
#     plt.bar(frames, branching_events_per_frame, color='red')
#     plt.xlabel("Frame")
#     plt.ylabel("Branching Events")
#     plt.title("Branching Events Per Frame")
#     plt.tight_layout()

#     graph_path = os.path.join(output_folder, "branching_rate_per_frame.png")
#     plt.savefig(graph_path)
#     plt.close()
#     print(f"Branching rate graph saved to {graph_path}")

# def save_distance_to_roi_graph(distances_to_roi, tip_id, output_folder):
#     """Generate and save a graph for distances to ROI over frames for a specific tip."""
#     frames = range(len(distances_to_roi))

#     plt.figure(figsize=(10, 6))
#     plt.plot(frames, distances_to_roi, marker='o', color='cyan')
#     plt.xlabel("Frame")
#     plt.ylabel("Distance to ROI (µm)")
#     plt.title(f"Distances to ROI for Tip {tip_id}")
#     plt.tight_layout()

#     graph_path = os.path.join(output_folder, f"distance_to_roi_tip_{tip_id}.png")
#     plt.savefig(graph_path)
#     plt.close()
#     print(f"Distance to ROI graph for Tip {tip_id} saved to {graph_path}")

# def save_spore_graph(spore_data, output_folder):
#     """Generate and save a graph for spore count or size per frame."""
#     frames = list(spore_data.keys())
#     spore_counts = [len(sizes) for sizes in spore_data.values()]
#     spore_sizes = [sum(sizes) / len(sizes) if sizes else 0 for sizes in spore_data.values()]

#     plt.figure(figsize=(10, 6))
#     plt.bar(frames, spore_counts, color='brown', label="Spore Count")
#     plt.plot(frames, spore_sizes, marker='o', color='gold', label="Average Spore Size (µm²)")
#     plt.xlabel("Frame")
#     plt.ylabel("Count / Size (µm²)")
#     plt.title("Spore Count and Average Size Per Frame")
#     plt.legend()
#     plt.tight_layout()

#     graph_path = os.path.join(output_folder, "spore_count_and_size.png")
#     plt.savefig(graph_path)
#     plt.close()
#     print(f"Spore graph saved to {graph_path}")

# ========== IMAGE PROCESSING FUNCTIONS ==========

# Preprocess Image
def preprocess_image(image):
    """
    Preprocess the image by cropping (using a predefined polygon region), 
    applying Otsu's thresholding, and binarizing the image.
    
    :param image: Grayscale image as a NumPy array.
    :return: Binary image as a NumPy array (1 for foreground, 0 for background).
    """
    # Define the crop points (polygon coordinates)
    crop_points = [
        (1625, 1032), (1827, 3045), (1897, 5848), 
        (2614, 6323), (9328, 5879), (9875, 5354),
        (9652, 2133), (9592, 376), (1988, 780)
    ]

    # Step 1: Create a mask for non-rectangular cropping
    # Create an empty mask
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # Define the polygon and fill it on the mask
    polygon = np.array(crop_points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)  # Fill the polygon with white (255)
    
    # Apply the mask to the image
    image = cv2.bitwise_and(image, image, mask=mask)

    # Step 2: Apply Otsu's thresholding
    threshold = threshold_otsu(image)                                           # Compute optimal threshold using Otsu's method
    binary_image = image > threshold                                            # Binarize image using the threshold
    
    # Step 3: Return the binary image
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
# Save Skeleton with Tips and Labels
def display_tips(binary_image, tips, frame_idx, output_folder):
    """
    Save the skeleton image with tips marked as red dots and labeled with their coordinates.

    :param binary_image: Skeletonized binary image as a NumPy array.
    :param tips: List of (row, col) coordinates of tip positions.
    :param frame_idx: Frame index to include in the output file name.
    :param output_folder: Folder path to save the tip visualization images.
    """
    import matplotlib.pyplot as plt
    import os

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Create a plot
    plt.figure(figsize=(10, 10))
    plt.imshow(binary_image, cmap='gray')  # Display the skeleton

    # Overlay red dots and labels for tips
    for idx, (y, x) in enumerate(tips):
        plt.scatter(x, y, c='red', s=1)  # Red dots for tips

    # Update title with frame index
    plt.title(f"Tips Visualization for Frame {frame_idx}")

    # Hide axes
    plt.axis('off')

    # Save the image to the output folder
    output_path = os.path.join(output_folder, f"tips_frame_{frame_idx}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()  # Close the plot to avoid memory issues
    print(f"Tip visualization for frame {frame_idx} saved to {output_path}")


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













# =========== HYPHAL METRICS ===========================
#=======================================================

# ========== HYPHAL TIP DETECTION ==========

# Detect endpoints

# Global CSV output folder
global_csv_folder = "function_outputs"
if not os.path.exists(global_csv_folder):
    os.makedirs(global_csv_folder)
    print(f"Global folder created: {global_csv_folder}")


def save_to_csv(data, filepath):
    """
    Save data to a CSV file.
    
    :param data: List of rows to write to the CSV file.
    :param filepath: Full path to the output CSV file.
    """
    # Check if folder exists
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    
    # Save the CSV
    with open(filepath, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)
    print(f"Data saved to {filepath}")




import os
import numpy as np
from scipy.ndimage import convolve
from skimage.measure import label
import csv

def find_hyphal_endpoints(filtered_skeleton, frame_idx, output_folder="hyphal_endpoints"):
    """
    Detect endpoints of hyphae by identifying pixels with exactly one connected neighbor.
    Save the results for each frame in a separate CSV file.
    
    :param filtered_skeleton: Skeletonized binary image.
    :param frame_idx: Index of the current frame.
    :param output_folder: Folder to store the CSV files.
    :return: List of (y, x) coordinates of detected endpoints.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder created: {output_folder}")

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
    
    # Save to a frame-specific CSV file
    csv_filename = os.path.join(output_folder, f"hyphal_endpoints_frame_{frame_idx}.csv")
    with open(csv_filename, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["y", "x"])  # Header row
        csv_writer.writerows(valid_endpoints)  # Write the endpoints
    
    print(f"Hyphal endpoints for frame {frame_idx} saved to {csv_filename}")
    
    return valid_endpoints





#DISTANCE TO REGIONS OF INTEREST
# Example: Regions of interest (e.g., spore centroids)
roi = [(100, 200), (150, 300)]  # Example coordinates
import os
import cv2
import numpy as np
import csv
def calculate_distances_to_roi_and_visualize(tracked_tips, tip_id, roi_vertices, images, output_folder):
    """
    Calculate the distances of a specific hyphal tip to a rectangular region of interest (ROI) and create visualizations.

    :param tracked_tips: Dictionary with tip IDs as keys and lists of positions [(frame, y, x)] as values.
    :param tip_id: The ID of the tip for which distances should be calculated.
    :param roi_vertices: List of exactly 4 vertices [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] defining the rectangle ROI.
    :param images: List of images corresponding to each frame.
    :param output_folder: Folder path to store visualizations and CSV data.
    :return: List of distances to the ROI for the specified tip over all frames.
    """
    if tip_id not in tracked_tips:
        raise ValueError(f"Tip ID {tip_id} not found in tracked tips.")

    if len(roi_vertices) != 4:
        raise ValueError("ROI must be defined by exactly 4 vertices.")

    # Ensure the base output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Create folders for visualizations and CSV
    visualization_folder = os.path.join(output_folder, "tip_distance_visualization")
    csv_folder = output_folder
    os.makedirs(visualization_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)

    distances = []
    visualization_data = [["Frame", "Shortest Distance to ROI (µm)"]]

    # Convert ROI vertices into a proper NumPy array
    roi_polygon = np.array(roi_vertices, dtype=np.int32)

    for frame_idx, (frame, y_tip, x_tip) in enumerate(tracked_tips[tip_id]):
        if frame_idx >= len(images):
            break  # Ensure we do not exceed the number of frames

        # Get the corresponding grayscale image for the frame
        image = images[frame_idx]
        
        # Convert the image to RGB for visualization
        visualized_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Highlight the ROI in yellow
        cv2.polylines(visualized_image, [roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)
        
        # Highlight the tip in red
        cv2.circle(visualized_image, (int(x_tip), int(y_tip)), radius=5, color=(0, 0, 255), thickness=-1)

        # Check if the tip is inside the ROI
        point_in_roi = cv2.pointPolygonTest(roi_polygon, (int(x_tip), int(y_tip)), False)
        if point_in_roi >= 0:
            distances.append(0)
            visualization_data.append([frame_idx, "0"])
            print(f"Tip at frame {frame_idx} is in the region of interest.")
        else:
            # Calculate the shortest distance from the tip to the ROI
            shortest_distance = float('inf')
            closest_point = None
            for i in range(len(roi_vertices)):
                # Get consecutive points in the polygon
                x1, y1 = roi_vertices[i]
                x2, y2 = roi_vertices[(i + 1) % len(roi_vertices)]  # Wrap around to the first point
                
                # Compute the closest point on the line segment to the tip
                px, py = closest_point_on_line_segment(x1, y1, x2, y2, x_tip, y_tip)
                distance = np.sqrt((px - x_tip) ** 2 + (py - y_tip) ** 2)
                if distance < shortest_distance:
                    shortest_distance = distance
                    closest_point = (px, py)
            
            distances.append(shortest_distance)
            visualization_data.append([frame_idx, f"{shortest_distance:.3f}"])

            # Draw the dotted line between the tip and the closest point on the ROI
            if closest_point:
                px, py = closest_point
                draw_dotted_line(visualized_image, (int(x_tip), int(y_tip)), (int(px), int(py)), color=(255, 255, 0))
        
        # Save the visualization
        output_path = os.path.join(visualization_folder, f"tip_{tip_id}_frame_{frame_idx}.png")
        cv2.imwrite(output_path, visualized_image)

    # Save distances to a CSV file
    csv_file_path = os.path.join(csv_folder, f"distances_to_roi_tip_{tip_id}.csv")
    with open(csv_file_path, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(visualization_data)
    
    print(f"Distances to ROI for Tip {tip_id} saved to {csv_file_path} and visualizations saved in {visualization_folder}.")

    return distances

# Define helper functions
def closest_point_on_line_segment(x1, y1, x2, y2, x, y):
    """
    Calculate the closest point on a line segment to a given point.
    """
    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:
        return x1, y1  # The segment is a single point
    
    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))  # Clamp t to the range [0, 1]
    return x1 + t * dx, y1 + t * dy

def draw_dotted_line(image, start, end, color, thickness=1, gap=5):
    """
    Draw a dotted line on an image between two points.
    """
    x1, y1 = start
    x2, y2 = end
    length = int(np.hypot(x2 - x1, y2 - y1))
    for i in range(0, length, gap * 2):
        start_x = int(x1 + i / length * (x2 - x1))
        start_y = int(y1 + i / length * (x2 - y1))
        end_x = int(x1 + min(i + gap, length) / length * (x2 - x1))
        end_y = int(y1 + min(i + gap, length) / length * (y2 - y1))
        cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)

# ========== HYPHAL METRICS ==========








#TIP GROWTH RATE
import matplotlib.pyplot as plt
import os

def calculate_average_growth_rate(tracked_tips, frame_interval, time_per_frame, output_folder):
    """
    Calculate the average growth rate of hyphal tips over a specified number of frames and save a graph.

    :param tracked_tips: Dictionary with tip IDs as keys and lists of positions [(frame, y, x)] as values.
    :param frame_interval: Number of frames over which to calculate the growth rate.
    :param time_per_frame: Time difference between consecutive frames.
    :param output_folder: Folder to store the output CSV file and graph.
    :return: Dictionary with tip IDs as keys and average growth rates as values, 
             and the general average growth rate for all tips.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")

    graphs_folder = os.path.join(output_folder, "graphs")
    os.makedirs(graphs_folder, exist_ok=True)
    print(f"Graphs folder created: {graphs_folder}")

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
            distance = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
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

    # Save average growth rates to CSV
    growth_rate_data = [["Tip ID", "Average Growth Rate (µm/s)"]]
    growth_rate_data += [[tip_id, f"{rate:.3f}"] for tip_id, rate in average_growth_rates.items()]
    growth_rate_data.append([])
    growth_rate_data.append(["General Average Growth Rate", f"{general_average_growth_rate:.3f}"])

    save_to_csv(growth_rate_data, os.path.join(output_folder, "average_growth_rates.csv"))
    print("Average growth rates saved to CSV.")

    # Plot and save growth rate graph
    plt.figure(figsize=(10, 6))
    plt.bar(average_growth_rates.keys(), average_growth_rates.values(), color='blue')
    plt.xlabel("Tip ID")
    plt.ylabel("Average Growth Rate (µm/s)")
    plt.title("Average Growth Rate of Hyphal Tips")
    plt.xticks(rotation=45)
    plt.tight_layout()

    graph_path = os.path.join(graphs_folder, "average_growth_rates.png")
    plt.savefig(graph_path)
    plt.close()
    print(f"Growth rate graph saved to {graph_path}")

    return average_growth_rates, general_average_growth_rate



#TIP GROWTH ANGLE

import matplotlib.pyplot as plt
import os

def calculate_growth_angles(tracked_tips, tip_id, output_folder):
    """
    Calculate the growth angles of a specific hyphal tip over time with respect to the horizontal 
    and save the results as a graph and CSV.

    :param tracked_tips: Dictionary with tip IDs as keys and lists of positions [(frame, y, x)] as values.
    :param tip_id: The ID of the tip for which growth angles should be calculated.
    :param output_folder: Folder to store the output CSV file and graph.
    :return: List of growth angles (in degrees) for the specified tip over time.
    """
    if tip_id not in tracked_tips:
        raise ValueError(f"Tip ID {tip_id} not found in tracked tips.")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")

    graphs_folder = os.path.join(output_folder, "graphs")
    os.makedirs(graphs_folder, exist_ok=True)
    print(f"Graphs folder created: {graphs_folder}")

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
    
    # Save growth angles to CSV
    growth_angle_data = [["Frame", "Growth Angle (°)"]]
    growth_angle_data += [[i + 1, f"{angle:.3f}"] for i, angle in enumerate(growth_angles)]

    csv_path = os.path.join(output_folder, f"growth_angles_tip_{tip_id}.csv")
    save_to_csv(growth_angle_data, csv_path)
    print(f"Growth angles for Tip {tip_id} saved to CSV: {csv_path}")

    # Plot and save growth angle graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(growth_angles) + 1), growth_angles, marker='o', color='blue', label=f"Tip {tip_id}")
    plt.xlabel("Frame")
    plt.ylabel("Growth Angle (°)")
    plt.title(f"Growth Angles of Tip {tip_id} Over Frames")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    graph_path = os.path.join(graphs_folder, f"growth_angles_tip_{tip_id}.png")
    plt.savefig(graph_path)
    plt.close()
    print(f"Growth angle graph for Tip {tip_id} saved to {graph_path}")

    return growth_angles








def calculate_tip_size(binary_image, tip_position, radius_microns=10, fov_1x=(1000, 1000), magnification=10):
    """
    Calculate the size of a single tip by counting the filled pixels within a specified radius.

    :param binary_image: Binary image as a NumPy array (1 for foreground, 0 for background).
    :param tip_position: Tuple (y, x) representing the position of the tip.
    :param radius_microns: Radius around the tip in microns.
    :param fov_1x: Field of View at 1x magnification (width, height) in micrometers.
    :param magnification: Magnification level of the image.
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
    radius_pixels = int(np.sqrt(radius_microns**2 / pixel_area))  # Convert radius from microns to pixels

    mask = np.zeros_like(binary_image, dtype=bool)  # Create a circular mask for the ROI
    y_grid, x_grid = np.ogrid[:binary_image.shape[0], :binary_image.shape[1]]
    distance_from_tip = np.sqrt((y_grid - y)**2 + (x_grid - x)**2)
    mask[distance_from_tip <= radius_pixels] = True

    # Count filled pixels within the circular mask
    tip_pixels = np.sum(binary_image[mask])

    # Convert the count of pixels to area in microns squared
    tip_size = tip_pixels * pixel_area
    return tip_size



import matplotlib.pyplot as plt
import os

def track_tip_size_over_time(tracked_tips, binary_images, tip_id, radius_microns=10, fov_1x=(1000, 1000), magnification=10, output_folder="output"):
    """
    Track the size of a specific tip over time across multiple frames, save the results to a CSV file, and generate a graph.

    :param tracked_tips: Dictionary with tip IDs as keys and lists of positions [(frame, y, x)] as values.
    :param binary_images: List of binary images (one per frame).
    :param tip_id: The ID of the tip to track.
    :param radius_microns: Radius around the tip in microns.
    :param fov_1x: Field of View at 1x magnification (width, height) in micrometers.
    :param magnification: Magnification level of the image.
    :param output_folder: Folder path to store the CSV file and graph.
    :return: List of tip sizes (in µm²) over time.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")

    graphs_folder = os.path.join(output_folder, "graphs")
    os.makedirs(graphs_folder, exist_ok=True)
    print(f"Graphs folder created: {graphs_folder}")

    if tip_id not in tracked_tips:
        raise ValueError(f"Tip ID {tip_id} not found in tracked tips.")

    tip_sizes = []  # To store the size of the tip in each frame
    tip_positions = tracked_tips[tip_id]  # Get the positions of the specified tip

    for frame_idx, (frame, y, x) in enumerate(tip_positions):
        # Get the binary image for the current frame
        binary_image = binary_images[frame]

        # Calculate the size of the tip in the current frame
        tip_size = calculate_tip_size(binary_image, (y, x), radius_microns, fov_1x, magnification)
        tip_sizes.append((frame, tip_size))

    # Save the results to a CSV file
    csv_file = os.path.join(output_folder, f"tip_{tip_id}_sizes.csv")
    with open(csv_file, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Frame", "Tip Size (µm²)"])  # Header row
        csv_writer.writerows(tip_sizes)  # Write data rows

    print(f"Tip sizes for Tip {tip_id} saved to {csv_file}")

    # Plot and save the tip size graph
    frames = [entry[0] for entry in tip_sizes]
    sizes = [entry[1] for entry in tip_sizes]

    plt.figure(figsize=(10, 6))
    plt.plot(frames, sizes, marker='o', color='green', label=f"Tip {tip_id}")
    plt.xlabel("Frame")
    plt.ylabel("Tip Size (µm²)")
    plt.title(f"Tip Size of Tip {tip_id} Over Frames")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    graph_path = os.path.join(graphs_folder, f"tip_size_tip_{tip_id}.png")
    plt.savefig(graph_path)
    plt.close()
    print(f"Tip size graph for Tip {tip_id} saved to {graph_path}")

    return tip_sizes


def calculate_overall_average_tip_size(tracked_tips, binary_images, radius_microns=10, output_folder="csv_outputs"):
    """
    Calculate the overall average size of all tips across all frames and save the result to a CSV file.

    :param tracked_tips: Dictionary with tip IDs as keys and lists of positions [(frame, y, x)] as values.
    :param binary_images: List of binary images (one per frame).
    :param radius_microns: Radius around the tip in microns for size calculation.
    :param output_folder: Folder path to store the CSV file.
    :return: The overall average tip size (µm²).
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")

    total_size = 0
    total_count = 0

    for tip_id, positions in tracked_tips.items():
        for frame, y, x in positions:
            # Get the binary image for the current frame
            binary_image = binary_images[frame]

            # Calculate the size of the tip in the current frame
            tip_size = calculate_tip_size(binary_image, (y, x), radius_microns)
            total_size += tip_size
            total_count += 1

    # Calculate overall average size
    overall_average_size = total_size / total_count if total_count > 0 else 0

    # Save the result to a CSV file
    csv_file = os.path.join(output_folder, "overall_average_tip_size.csv")
    with open(csv_file, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Metric", "Value"])  # Header row
        csv_writer.writerow(["Overall Average Tip Size (µm²)", overall_average_size])

    print(f"Overall average tip size saved to {csv_file}")
    return overall_average_size




#============ BRANCHING FREQUENCY ===============
import os
import csv
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

def calculate_branching_rate(tip_positions, distance_threshold=15, output_folder="csv_outputs"):
    """
    Calculate the branching rate/frequency of fungal hyphae over time, save to a CSV file, and generate a graph.

    :param tip_positions: List of lists of (y, x) tip positions for each frame.
    :param distance_threshold: Maximum distance to consider tips as originating from the same source.
    :param output_folder: Folder path to store the CSV file and graphs.
    :return: List of branching events per frame and total branching events.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")

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

    # Save the results to a CSV file
    csv_file = os.path.join(output_folder, "branching_rate.csv")
    with open(csv_file, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Frame", "Branching Events"])  # Header row
        csv_writer.writerows(enumerate(branching_events_per_frame))  # Frame-wise data
        csv_writer.writerow(["Total Branching Events", total_branching_events])

    # Create a graph of branching events per frame
    graph_folder = os.path.join(output_folder, "graphs")
    os.makedirs(graph_folder, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(branching_events_per_frame) + 1), branching_events_per_frame, marker='o', label="Branching Events")
    plt.title("Branching Events Per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Branching Events")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(graph_folder, "branching_rate_graph.png"))
    plt.close()

    print(f"Branching rate saved to {csv_file} and graph saved to {graph_folder}")
    return branching_events_per_frame, total_branching_events



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


def calculate_biomass_over_time(image_files, fov_1x, magnification, output_folder="csv_outputs"):
    """
    Calculate biomass over time for a sequence of images, save to a CSV file, and generate a graph.

    :param image_files: List of file paths to the PNG images.
    :param fov_1x: Field of View at 1x magnification (width, height) in micrometers.
    :param magnification: Magnification level of the images.
    :param output_folder: Folder path to store the CSV file and graphs.
    :return: List of biomass values (one for each frame).
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")

    biomass_values = []

    for frame_idx, file in enumerate(image_files):
        # Load and preprocess the image
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        binary_image = preprocess_image(image)
        
        # Calculate biomass
        biomass = find_biomass(binary_image, fov_1x, magnification)
        biomass_values.append((frame_idx, biomass))

    # Save the results to a CSV file
    csv_file = os.path.join(output_folder, "biomass_over_time.csv")
    with open(csv_file, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Frame", "Biomass (µm²)"])  # Header row
        csv_writer.writerows(biomass_values)  # Write data rows

    # Create a graph of biomass over time
    graph_folder = os.path.join(output_folder, "graphs")
    os.makedirs(graph_folder, exist_ok=True)
    plt.figure(figsize=(10, 6))
    frames, biomass = zip(*biomass_values)
    plt.plot(frames, biomass, marker='o', label="Biomass")
    plt.title("Biomass Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Biomass (µm²)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(graph_folder, "biomass_graph.png"))
    plt.close()

    print(f"Biomass over time saved to {csv_file} and graph saved to {graph_folder}")
    return [value[1] for value in biomass_values]




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
def track_spores_over_time(image_files, min_size=10, max_size=200, circularity_threshold=0.7, distance_threshold=15, output_folder="csv_outputs"):
    """
    Track spores over time across a sequence of images and output their sizes over time, spore count per frame, and generate graphs.

    :param image_files: List of file paths to the PNG images.
    :param min_size: Minimum size of objects to consider as spores.
    :param max_size: Maximum size of objects to consider as spores.
    :param circularity_threshold: Minimum circularity to consider an object as a spore.
    :param distance_threshold: Maximum distance to match spores between frames.
    :param output_folder: Folder path to store the CSV file and graphs.
    :return: Dictionary of tracked spores with their sizes over time.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    graph_folder = os.path.join(output_folder, "graphs")
    os.makedirs(graph_folder, exist_ok=True)
    print(f"Output folder created: {output_folder} and graph folder: {graph_folder}")

    # Dictionary to store tracked spores: {spore_id: {"history": [(frame_idx, size)], "last_position": (x, y)}}
    tracked_spores = {}
    next_spore_id = 0

    spore_count_per_frame = []  # Store the number of spores detected per frame
    average_spore_size_per_frame = []  # Store the average spore size per frame

    # Process each frame
    for frame_idx, file in enumerate(image_files):
        # Load the image
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"File not found: {file}")

        # Identify spores in the current frame
        current_spores = identify_spores(image, min_size, max_size, circularity_threshold)

        # Update spore count and average size
        spore_count = len(current_spores)
        spore_count_per_frame.append((frame_idx, spore_count))

        if spore_count > 0:
            average_size = sum(spore["size"] for spore in current_spores) / spore_count
        else:
            average_size = 0
        average_spore_size_per_frame.append((frame_idx, average_size))

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

    # Save spore count and average size data to CSV
    spore_count_csv = os.path.join(output_folder, "spore_count_per_frame.csv")
    with open(spore_count_csv, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Frame", "Spore Count"])  # Header row
        csv_writer.writerows(spore_count_per_frame)  # Write data rows

    average_spore_size_csv = os.path.join(output_folder, "average_spore_size_per_frame.csv")
    with open(average_spore_size_csv, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Frame", "Average Spore Size (µm²)"])  # Header row
        csv_writer.writerows(average_spore_size_per_frame)  # Write data rows

    # Create graphs
    frames, spore_counts = zip(*spore_count_per_frame)
    _, average_sizes = zip(*average_spore_size_per_frame)

    # Spore Count Graph
    plt.figure(figsize=(10, 6))
    plt.bar(frames, spore_counts, color='blue', alpha=0.7, label="Spore Count")
    plt.title("Spore Count Per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Spore Count")
    plt.grid(axis='y')
    plt.legend()
    plt.savefig(os.path.join(graph_folder, "spore_count_graph.png"))
    plt.close()

    # Average Spore Size Graph
    plt.figure(figsize=(10, 6))
    plt.plot(frames, average_sizes, marker='o', color='green', label="Average Spore Size")
    plt.title("Average Spore Size Per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Average Spore Size (µm²)")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(graph_folder, "average_spore_size_graph.png"))
    plt.close()

    print(f"Spore count and size saved to CSV, and graphs saved to {graph_folder}.")
    return tracked_spores














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


import os
import cv2
import numpy as np

# Define constants
fov_1x = (1000, 1000)  # Field of view at 1x magnification in micrometers (width, height)
magnification = 10  # Magnification level
time_per_frame = 2  # Time difference between consecutive frames in seconds
frame_interval = 2  # Number of frames to calculate growth rates
distance_threshold = 15  # Distance threshold for tip matching
min_size_spores = 10  # Minimum size of spores
max_size_spores = 200  # Maximum size of spores
circularity_threshold = 0.7  # Circularity threshold for spores
roi_polygon = np.array([  # Convert ROI coordinates to NumPy array
    (1000, 1000), (4000, 1000), (4000, 4000), (1000, 4000)
], dtype=np.int32)

# Ensure the input images are sorted by frame
image_files = sorted(
    [os.path.join("/Users/noahweiler/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Noah SWE Project/Processed_images", f) 
     for f in os.listdir("/Users/noahweiler/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Noah SWE Project/Processed_images") 
     if f.endswith(".tif")]
)

# Output base folders
base_visualization_folder = "roi_visualizations"
hyphal_endpoints_folder = "hyphal_endpoints"
tip_visualization_folder = "tip_visualization_images"
os.makedirs(base_visualization_folder, exist_ok=True)
os.makedirs(hyphal_endpoints_folder, exist_ok=True)
os.makedirs(tip_visualization_folder, exist_ok=True)
print(f"Output folders created or already exist: {base_visualization_folder}, {hyphal_endpoints_folder}, {tip_visualization_folder}")

# Process Image Sequence
tip_positions_sequence = []
biomass_values = []
images = []  # Collect grayscale images for visualization

for frame_idx, image_file in enumerate(image_files):
    # Load the image
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {image_file}")
    images.append(image)

    # Preprocess and skeletonize
    binary_image = preprocess_image(image)
    skeleton = skeletonize_image(binary_image)
    filtered_skeleton = filter_hyphae(skeleton, min_size=500)

    # Find and save hyphal endpoints per frame
    endpoints = find_hyphal_endpoints(filtered_skeleton, frame_idx, output_folder=hyphal_endpoints_folder)
    tip_positions_sequence.append(endpoints)

    # Save the tip visualizations
    #display_tips(binary_image, endpoints, frame_idx, output_folder=tip_visualization_folder)

    # Calculate biomass
    biomass = find_biomass(binary_image, fov_1x, magnification)
    biomass_values.append(biomass)

    print(f"Processed frame {frame_idx}")

# Track Tips Across Frames
tracked_tips = track_tips_across_frames(tip_positions_sequence, distance_threshold)

# Visualize and Calculate Distances to ROI
iteration_folder = os.path.join(base_visualization_folder, f"iteration_{len(os.listdir(base_visualization_folder)) + 1}")
os.makedirs(iteration_folder, exist_ok=True)
print(f"Iteration folder created: {iteration_folder}")

# Ensure Tip ID exists in tracked_tips
if 1000 in tracked_tips:
    tip_id = 1000
    calculate_distances_to_roi_and_visualize(tracked_tips, tip_id, roi_polygon, images, iteration_folder)
else:
    print("Tip ID 1000 not found in tracked tips. Skipping distance to ROI calculation.")

# Calculate Metrics
# Average Growth Rates
average_growth_rate_folder = os.path.join(base_visualization_folder, "average_growth_rates")
os.makedirs(average_growth_rate_folder, exist_ok=True)
average_growth_rates, general_average_growth_rate = calculate_average_growth_rate(
    tracked_tips, frame_interval, time_per_frame, output_folder=average_growth_rate_folder
)
print("Average Growth Rates for Each Tip:", average_growth_rates)
print("General Average Growth Rate:", general_average_growth_rate)

# Growth Angles
growth_angles_folder = os.path.join(base_visualization_folder, "growth_angles")
os.makedirs(growth_angles_folder, exist_ok=True)
growth_angles = calculate_growth_angles(tracked_tips, tip_id, output_folder=growth_angles_folder)
print(f"Growth Angles for Tip {tip_id}:", growth_angles)

# Branching Rate
branching_rate_folder = os.path.join(base_visualization_folder, "branching_rate")
os.makedirs(branching_rate_folder, exist_ok=True)
branching_events_per_frame, total_branching_events = calculate_branching_rate(
    tip_positions_sequence, distance_threshold, output_folder=branching_rate_folder
)
print("Branching Events Per Frame:", branching_events_per_frame)
print("Total Branching Events:", total_branching_events)

# Spore Tracking
spore_tracking_folder = os.path.join(base_visualization_folder, "spore_tracking")
os.makedirs(spore_tracking_folder, exist_ok=True)
spore_tracking = track_spores_over_time(
    image_files, min_size=min_size_spores, max_size=max_size_spores,
    circularity_threshold=circularity_threshold, distance_threshold=distance_threshold,
    output_folder=spore_tracking_folder
)
print("Spore Size Histories Over Time:", spore_tracking)

# Biomass Analysis
biomass_folder = os.path.join(base_visualization_folder, "biomass")
os.makedirs(biomass_folder, exist_ok=True)
calculate_biomass_over_time(image_files, fov_1x, magnification, output_folder=biomass_folder)
print("Biomass Over Time:", biomass_values)

# Processing Complete
print("Processing complete. All results are saved as CSV files and visualizations in their respective folders.")
