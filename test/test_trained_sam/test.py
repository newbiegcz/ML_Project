import cv2
import segment_anything

# Load an image
img = cv2.imread("image.jpg")

# Define a font for the text
font = cv2.FONT_HERSHEY_SIMPLEX

# Define a callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    # If the left button is clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Segment the image at the clicked point
        seg, label, score = segment_anything.segment(img, x, y)
        # Create a copy of the image for masking
        mask = img.copy()
        # Update the mask with the segmentation result
        mask[seg != 0] = seg[seg != 0]
        # Put the label and score on the mask
        text = f"Label: {label}, Score: {score}"
        cv2.putText(mask, text, (10, 30), font, 1, (255, 255, 255), 2)
        # Make the mask semi-transparent by blending it with the original image
        alpha = 0.5 # Adjust this value to change the transparency level
        blend = cv2.addWeighted(img, alpha, mask, 1 - alpha, 0)
        # Show the blended image
        cv2.imshow("Blend", blend)

# Create a window and set the mouse callback
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

# Show the original image
cv2.imshow("Image", img)

# Wait for a key press to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
