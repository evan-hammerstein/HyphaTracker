import cv2
import tifffile as tiff
import numpy as np
from multiprocessing import Pool
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

class PanZoomWindow(object):
    """ Controls an OpenCV window. Registers a mouse listener so that:
        1. right-dragging up/down zooms in/out
        2. right-clicking re-centers
        3. trackbars scroll vertically and horizontally 
    You can open multiple windows at once if you specify different window names.
    You can pass in an onLeftClickFunction, and when the user left-clicks, this 
    will call onLeftClickFunction(y,x), with y,x in original image coordinates."""
    
    def __init__(self, img, windowName = 'PanZoomWindow', onLeftClickFunction = None):
        self.WINDOW_NAME = windowName
        self.H_TRACKBAR_NAME = 'x'
        self.V_TRACKBAR_NAME = 'y'
        self.img = img
        self.onLeftClickFunction = onLeftClickFunction
        self.TRACKBAR_TICKS = 1000
        self.panAndZoomState = PanAndZoomState(img.shape, self)
        self.lButtonDownLoc = None
        self.mButtonDownLoc = None
        self.rButtonDownLoc = None
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        self.redrawImage()
        cv2.setMouseCallback(self.WINDOW_NAME, self.onMouse)
        cv2.createTrackbar(self.H_TRACKBAR_NAME, self.WINDOW_NAME, 0, self.TRACKBAR_TICKS, self.onHTrackbarMove)
        cv2.createTrackbar(self.V_TRACKBAR_NAME, self.WINDOW_NAME, 0, self.TRACKBAR_TICKS, self.onVTrackbarMove)
    def onMouse(self, event, x, y, _ignore1, _ignore2):
        """ Responds to mouse events within the window. 
        The x and y are pixel coordinates in the image currently being displayed.
        If the user has zoomed in, the image being displayed is a sub-region, so you'll need to
        add self.panAndZoomState.ul to get the coordinates in the full image."""
        if event == cv2.EVENT_MOUSEMOVE:
            return
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Record where the user started to right-drag
            self.mButtonDownLoc = np.array([y, x])
        elif event == cv2.EVENT_RBUTTONUP and self.mButtonDownLoc is not None:
            # The user just finished right-dragging
            dy = y - self.mButtonDownLoc[0]
            pixelsPerDoubling = 0.2 * self.panAndZoomState.shape[0]  # lower = zoom more
            changeFactor = (1.0 + abs(dy) / pixelsPerDoubling)
            changeFactor = min(max(1.0, changeFactor), 5.0)
            if changeFactor < 1.05:
                dy = 0  # this was a click, not a draw. So don't zoom, just re-center.
            if dy > 0:  # moved down, so zoom out.
                zoomInFactor = 1.0 / changeFactor
            else:
                zoomInFactor = changeFactor
            self.panAndZoomState.zoom(self.mButtonDownLoc[0], self.mButtonDownLoc[1], zoomInFactor)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # The user pressed the left button.
            coordsInDisplayedImage = np.array([y, x])
            if np.any(coordsInDisplayedImage < 0) or np.any(coordsInDisplayedImage > self.panAndZoomState.shape[:2]):
                print("You clicked outside the image area.")
            else:
                print("You clicked on %s within the zoomed rectangle." % coordsInDisplayedImage)
                coordsInFullImage = self.panAndZoomState.ul + coordsInDisplayedImage
                print("This is %s in the actual image." % coordsInFullImage)
                pixel_value = self.img[coordsInFullImage[0], coordsInFullImage[1]]
                if self.img.ndim == 2:  # Grayscale image
                    print("This pixel holds intensity %s." % pixel_value)
                else:  # Color image
                    print("This pixel holds color values: %s." % (pixel_value,))
                if self.onLeftClickFunction is not None:
                    self.onLeftClickFunction(coordsInFullImage[0], coordsInFullImage[1])


        #you can handle other mouse click events here
    def onVTrackbarMove(self,tickPosition):
        self.panAndZoomState.setYFractionOffset(float(tickPosition)/self.TRACKBAR_TICKS)
    def onHTrackbarMove(self,tickPosition):
        self.panAndZoomState.setXFractionOffset(float(tickPosition)/self.TRACKBAR_TICKS)
    def redrawImage(self):
        pzs = self.panAndZoomState
        cv2.imshow(self.WINDOW_NAME, self.img[pzs.ul[0]:pzs.ul[0]+pzs.shape[0], pzs.ul[1]:pzs.ul[1]+pzs.shape[1]])


class PanAndZoomState(object):
    """ Tracks the currently-shown rectangle of the image.
    Does the math to adjust this rectangle to pan and zoom."""
    MIN_SHAPE = np.array([50,50])
    def __init__(self, imShape, parentWindow):
        self.ul = np.array([0,0]) #upper left of the zoomed rectangle (expressed as y,x)
        self.imShape = np.array(imShape[0:2])
        self.shape = self.imShape #current dimensions of rectangle
        self.parentWindow = parentWindow
    def zoom(self,relativeCy,relativeCx,zoomInFactor):
        self.shape = (self.shape.astype(float)/zoomInFactor).astype(int)
        #expands the view to a square shape if possible. (I don't know how to get the actual window aspect ratio)
        self.shape[:] = np.max(self.shape) 
        self.shape = np.maximum(PanAndZoomState.MIN_SHAPE,self.shape) #prevent zooming in too far
        c = self.ul+np.array([relativeCy,relativeCx])
        self.ul = (c-self.shape/2).astype(int)
        self._fixBoundsAndDraw()
    def _fixBoundsAndDraw(self):
        """ Ensures we didn't scroll/zoom outside the image. 
        Then draws the currently-shown rectangle of the image."""
#        print("in self.ul: %s shape: %s"%(self.ul,self.shape))
        self.ul = np.maximum(0,np.minimum(self.ul, self.imShape-self.shape))
        self.shape = np.minimum(np.maximum(PanAndZoomState.MIN_SHAPE,self.shape), self.imShape-self.ul)
#        print("out self.ul: %s shape: %s"%(self.ul,self.shape))
        yFraction = float(self.ul[0])/max(1,self.imShape[0]-self.shape[0])
        xFraction = float(self.ul[1])/max(1,self.imShape[1]-self.shape[1])
        cv2.setTrackbarPos(self.parentWindow.H_TRACKBAR_NAME, self.parentWindow.WINDOW_NAME,int(xFraction*self.parentWindow.TRACKBAR_TICKS))
        cv2.setTrackbarPos(self.parentWindow.V_TRACKBAR_NAME, self.parentWindow.WINDOW_NAME,int(yFraction*self.parentWindow.TRACKBAR_TICKS))
        self.parentWindow.redrawImage()
    def setYAbsoluteOffset(self,yPixel):
        self.ul[0] = min(max(0,yPixel), self.imShape[0]-self.shape[0])
        self._fixBoundsAndDraw()
    def setXAbsoluteOffset(self,xPixel):
        self.ul[1] = min(max(0,xPixel), self.imShape[1]-self.shape[1])
        self._fixBoundsAndDraw()
    def setYFractionOffset(self,fraction):
        """ pans so the upper-left zoomed rectange is "fraction" of the way down the image."""
        self.ul[0] = int(round((self.imShape[0]-self.shape[0])*fraction))
        self._fixBoundsAndDraw()
    def setXFractionOffset(self,fraction):
        """ pans so the upper-left zoomed rectange is "fraction" of the way right on the image."""
        self.ul[1] = int(round((self.imShape[1]-self.shape[1])*fraction))
        self._fixBoundsAndDraw()



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
    

# Load the TIFF
tiff_file = r"C:\Users\Harry\OneDrive - Imperial College London\Bioeng\Year 3\Software Engineering\HyphaTracker\timelapse1.tif"
frames = tiff.imread(tiff_file)  # Load all frames as a NumPy array

# Display each frame after processing
for frame_idx, frame in enumerate(frames):
    print(f"Processing and displaying frame {frame_idx + 1}")

    # Process the current frame
    processed_frame = process_frame(frame)

    # Display the processed frame
    #window_name = "Processed Frame"
    #cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Flexible window size
   # cv2.imshow(window_name, processed_frame)

    window_name = f"Processed Frame {frame_idx + 1}"
    window = PanZoomWindow(processed_frame, window_name)
    key = -1    


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
