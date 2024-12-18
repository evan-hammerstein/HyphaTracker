import cv2
import tifffile as tiff
import numpy as np
from multiprocessing import Pool
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt


def process_frame(frame):
    # Normalize
    if frame.max() > 255:
        frame = (255 * (frame / frame.max())).astype(np.uint8)
    
    # Convert to grayscale
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Apply bilateral filter
    filtered = cv2.bilateralFilter(frame, d=15, sigmaColor=50, sigmaSpace=25) #better edge preservation 
    #filtered = cv2.GaussianBlur(frame, (5,5), sigmaX=15, sigmaY=15)
    
    #CLAHE filter?

    #tesing with bilateral filter and a threshold to binarise 
    thres_gauss = cv2.adaptiveThreshold(filtered,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    thres_mean = cv2.adaptiveThreshold(filtered,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)



    # Divide
    divide = cv2.divide(filtered, frame, scale=255) #change filtered component to test with the thresholds/change the filter
    divide = 255 - divide

    # Stretch
    maxval = np.amax(divide) / 4
    stretch = rescale_intensity(divide, in_range=(0, maxval), out_range=(0, 255)).astype(np.uint8)

    return stretch
    
def plot_histogram(image, frame_idx):
    """Plots a histogram of pixel intensities for the given image."""
    plt.figure(figsize=(8, 5))
    plt.hist(image.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.7)
    plt.title(f"Pixel Intensity Histogram - Frame {frame_idx + 1}")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

# Load the TIFF
tiff_file = r"C:\Users\Harry\OneDrive - Imperial College London\Bioeng\Year 3\Software Engineering\HyphaTracker\timelapse1.tif"
frames = tiff.imread(tiff_file)  # Load all frames as a NumPy array

# Display each frame after processing
for frame_idx, frame in enumerate(frames):
    print(f"Processing and displaying frame {frame_idx + 1}")

    # Process the current frame
    processed_frame = process_frame(frame)

    # Display the processed frame
    window_name = "Processed Frame"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Flexible window size
    cv2.imshow(window_name, processed_frame)

       # Plot the histogram
    # plot_histogram(processed_frame, frame_idx)

    # Wait for user input
    key = cv2.waitKey(0)
    if key == 27:  # Esc key to exit
        print("Exiting...")
        break
    elif key == ord('s'):  # Save the current frame
        cv2.imwrite(f"processed_frame_{frame_idx + 1}.png", processed_frame)
        print(f"Saved frame {frame_idx + 1}.")

# Clean up
cv2.destroyAllWindows()
