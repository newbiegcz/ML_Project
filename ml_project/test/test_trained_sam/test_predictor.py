import cv2
import numpy as np
from utils.automatic_label_generator import SamAutomaticLabelGenerator
from utils.load_extracted_checkpoint import load_extracted_checkpoint
from modeling.predictor import SamWithLabelPredictor
import albumentations
import torch

model = load_extracted_checkpoint("checkpoint/extracted.pth").cuda()
mask_generator = SamAutomaticLabelGenerator(model)
predictor = mask_generator.predictor

# Load an image
image_id = 70
img = cv2.imread("test/test_trained_sam/local_files/image%d.jpg" % image_id)
with open("test/test_trained_sam/local_files/height%d.txt" % image_id, "r") as f:
    normalized_z = float(f.read())

# Define a font for the text
font = cv2.FONT_HERSHEY_SIMPLEX

# img = albumentations.CLAHE(p=1.0)(image=img)['image']
label = mask_generator.generate_labels(img, normalized_z)


def imshow(name, img):
    img = cv2.resize(img, (256, 256)) # reduce the size of the image
    cv2.imshow(name, img)

# Define a callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    # If the left button is clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        normalized_x = y / img.shape[1] # 注意反过来了这个非常严重的问题
        normalized_y = x / img.shape[0]
        assert normalized_x >= 0 and normalized_x <= 1
        assert normalized_y >= 0 and normalized_y <= 1
        masks, iou_predictions, label_predictions, low_res_masks = predictor.predict(
            np.array([[x, y]]), np.array([1]), 
            prompt_3d = np.array([normalized_x, normalized_y, normalized_z])
        )
        
        label = label_predictions.argmax()
        prob = torch.nn.functional.softmax(torch.from_numpy(label_predictions).unsqueeze(0), dim=1)[0, label]
        
        # seg = masks[label]
        seg = masks[0]
        score = iou_predictions[0]


        mask = img.copy()
        # Update the mask with the segmentation result
        mask[seg != 0] = np.array([255, 0, 0])
        # Put the label and score on the mask
        text = f"Label: {label}, Score: {score}\n, Prob: {prob}"
        cv2.putText(mask, text, (10, 30), font, 1, (255, 255, 255), 2)
        # Make the mask semi-transparent by blending it with the original image
        alpha = 0.5 # Adjust this value to change the transparency level
        blend = cv2.addWeighted(img, alpha, mask, 1 - alpha, 0)
        # Show the blended image
        imshow("Blend", blend)

# Create a window and set the mouse callback

predictor.set_image(img)
# Show the original image

pd_label_image = (label.reshape((1024, 1024, 1)).repeat(3, axis=2) / 14 * 255.0).astype(np.uint8)
gt_label_image = cv2.imread("test/test_trained_sam/local_files/label%d.jpg" % image_id)


imshow("pd_Label", pd_label_image)
imshow("gt_Label", gt_label_image)

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)
cv2.imshow("Image", img)
# Wait for a key press to exit
while True:
    key = cv2.waitKey(10)
    if key == ord("q"):
        break
cv2.destroyAllWindows()
