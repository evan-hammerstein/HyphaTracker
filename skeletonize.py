from skimage import io, color, morphology
from skimage.filters import threshold_local
from skimage.color import rgba2rgb
from skimage.exposure import equalize_adapthist
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image_path = '/Users/lindaschermeier/Desktop/HyIm.png'
image = io.imread(image_path)

# Convert RGBA to RGB if necessary
if image.shape[-1] == 4:  # Check if the image has an alpha channel
    image = rgba2rgb(image)

# Convert to grayscale
gray_image = color.rgb2gray(image)

# Enhance contrast using adaptive histogram equalization
enhanced_image = equalize_adapthist(gray_image)

# Apply local thresholding with fine-tuned offset
block_size = 35  # Size of the neighborhood for adaptive thresholding
offset = 0.05  # Fine-tuned offset value
local_thresh = threshold_local(enhanced_image, block_size, offset=offset)
binary_image = enhanced_image > local_thresh

# Morphological cleaning: remove small objects and smooth edges
cleaned_binary = morphology.remove_small_objects(binary_image, min_size=500)  # Adjust min_size as needed
cleaned_binary = morphology.binary_opening(cleaned_binary, morphology.disk(2))  # Smooth small artifacts

# Skeletonize the binary image
skeleton = morphology.skeletonize(cleaned_binary)

# Post-skeletonization cleanup: remove spurious branches
skeleton_cleaned = morphology.remove_small_objects(skeleton, min_size=50)  # Remove small branches

# Display the images at different stages
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original Grayscale Image')
axes[0].axis('off')

axes[1].imshow(cleaned_binary, cmap='gray')
axes[1].set_title('Binary Image (Cleaned)')
axes[1].axis('off')

axes[2].imshow(skeleton, cmap='gray')
axes[2].set_title('Skeletonized Image (Raw)')
axes[2].axis('off')

axes[3].imshow(skeleton_cleaned, cmap='gray')
axes[3].set_title('Skeletonized Image (Cleaned)')
axes[3].axis('off')

plt.show()
