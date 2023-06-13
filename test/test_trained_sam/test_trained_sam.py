import cv2
import numpy as np
from utils.load_extracted_checkpoint import load_extracted_checkpoint
from modeling.predictor import SamWithLabelPredictor
import albumentations

model = load_extracted_checkpoint("extracted.pth").cuda()
predictor = SamWithLabelPredictor(model)

# Load an image
img = cv2.imread("test/test_trained_sam/local_files/image70.jpg")

# Define a font for the text
font = cv2.FONT_HERSHEY_SIMPLEX

# img = albumentations.CLAHE(p=1.0)(image=img)['image']
predictor.set_image(img)

# Define a callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    # If the left button is clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        masks, iou_predictions, label_predictions, low_res_masks = predictor.predict(np.array([[x, y]]), np.array([1]))
        
        label = label_predictions.argmax()
        seg = masks[label]
        score = iou_predictions[label]

        mask = img.copy()
        # Update the mask with the segmentation result
        mask[seg != 0] = np.array([255, 0, 0])
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
