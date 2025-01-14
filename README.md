# HyphaTracker

This project is designed to analyze fungal growth using image processing techniques. It provides tools to process a sequence of tiff files representing a time lapse of fungal growth, detect and track hyphal tips, calculate growth metrics, and visualize results. The analysis includes measurements of tip growth rates, branching frequency, spore identification, biomass quantification, and distances to regions of interest (ROIs).

# Features

**1. Image Preprocessing:** Cropping, binarization, and skeletonization of grayscale images.

**2. Hyphal Analysis:**

* Detect endpoints and calculate growth rates and angles.

* Track hyphal tips and calculate distances to regions of interest (ROI).

* Analyze tip size and branching frequencies.

**3. Biomass Analysis:** Measure fungal biomass over time.

**4. Spore Tracking:**

* Identify spores based on size, shape, and proximity to biomass.

* Track spores across multiple frames.

**5. Visualization:** Generate visual outputs, including skeletonized images, tracked tips, and ROI distance visualizations.

**6. Metrics Output:** Save results in CSV files and generate graphs for key metrics.

# Requirements

# Software Dependencies

The code requires the following Python libraries:

* `sys`
* `os`
* `cv2` (OpenCV)
* `numpy`
* `skimage`
* `scipy`
* `matplotlib`
* `csv`

Ensure all dependencies are installed before running the script.

# Input Data

* A folder containing grayscale image files (`.tif` format).

* Images should be named with frame numbers to allow sequential processing.

# Usage Instructions

## Step 1: Prepare Input Data

Place all `.tif` image files in a single folder.

Ensure filenames follow a pattern allowing frame numbers to be extracted (e.g., processed_frame_001.tif).

## Step 2: Run the Script

Use the following command:

`python script.py <image_folder_path> <magnification> <output_folder>`

`<image_folder_path>`: Path to the folder containing .tif files.

`<magnification>`: Objective lens magnification (10, 20, 40, or 100).

`<output_folder>`: Path to save the outputs (CSV files, visualizations, and graphs).

## Step 3: Review Results

Results will be saved in the specified output folder:

**1. CSV Files:** Contain metrics for tips, spores, biomass, etc.

**2. Visualizations:** Skeletonized images and tracked tip visualizations.

**3. Graphs:** Growth rates, branching frequencies, and other metrics as `.png` files.

# Key Functions

## Image Preprocessing

**`preprocess_image`:** Crops the image and applies Otsu's thresholding.

**`skeletonize_image`:** Reduces structures to 1-pixel-wide lines.

## Hyphal Analysis

**`find_hyphal_endpoints`:** Detects hyphal endpoints.

**`track_tips_across_frames`:** Matches tips between frames.

**`calculate_average_growth_rate`:** Computes average growth rate of tips.

**`calculate_growth_angles`:** Determines growth angles relative to the horizontal.

**`calculate_branching_rate`:** Identifies and counts branching events.

## Biomass Analysis

**`find_biomass`:** Calculates the area covered by fungal biomass.

**`calculate_biomass_over_time`:** Tracks biomass changes across frames.

## Spore Analysis

**`identify_spores`:** Detects spores based on size, shape, and proximity to biomass.

**`track_spores_over_time`:** Tracks spores across frames and calculates size changes.

## Visualization

**`show_image`:** Displays an image with optional saving.

**`display_tips`:** Visualizes skeletonized images with tips marked.

**`visualize_tracked_tips`:** Shows tracked tips across frames.

**`calculate_distances_to_roi_and_visualize`:** Computes and visualizes distances from tips to a specified ROI.

# Outputs

## CSV Files:

* Hyphal tip metrics (growth rate, angles, sizes).

* Branching frequency data.

* Biomass values over time.

* Spore counts and sizes.

## Graphs:

Growth rates, branching frequencies, biomass trends, and spore metrics.

## Visualizations:

* Skeletonized images with tips.

* Tip distance visualizations.

# Customization

Modify `distance_threshold`, `min_size`, or `circularity_threshold` to adjust sensitivity.

Update `roi_polygon` for custom regions of interest.

Use different magnification levels to adjust pixel-to-area conversion factors.

# Notes

Ensure input images are preprocessed for optimal results.

Review logs for any warnings or errors during processing.

# Contact

For questions or support, contact the developer at [evan_hammerstein22@imperial.ac.uk].
