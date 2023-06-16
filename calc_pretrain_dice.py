import data.dataset
import model.prompter
import utils.eval
import gc
import numpy as np
import cv2
import matplotlib.pyplot as plt
from data.dataset import get_dataset_3d
from segment_anything import SamPredictor, sam_model_registry

dataset_3d = get_dataset_3d('validation', crop_roi=True)

model_type = "vit_h"
model_checkpoint = "checkpoint/sam_vit_h_4b8939.pth"

sam = sam_model_registry[model_type](checkpoint=model_checkpoint)
sam_predictor = SamPredictor(sam)

Promptor = model.prompter.Prompter()

import matplotlib.pyplot as plt
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)  

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def predict(image, label):

    print(image.shape)
    print(label.shape)

    image = image.repeat(3, 1, 1)

    image = image.permute(1, 2, 0).numpy()
    image *= 255
    image = np.array(image, dtype = np.uint8)

    #global image2
    #image2 = image

    print(image.shape)

    sam_predictor.set_image(image)
        
    result = np.zeros(label.shape)
    scores = np.zeros(label.shape, dtype = float)

    for i in range(1, 14):
        #print(i)
        mask = np.zeros(label.shape, dtype = np.uint8)
        mask[label == i] = 1
        if (np.max(mask) == 0):
            continue
        prompt = Promptor(mask, sam_predictor, 1, True, False)
        #for j in range(mask.shape[0]):
        #    for k in range(mask.shape[1]):
        #        print(mask[j][k], end = ' ')
        #    print()
        #print(prompt)
        msk, score, logit = sam_predictor.predict(**prompt)  

        x, y, z = msk.shape
        msk = msk.reshape(y, z)

        #plt.ion()

        #plt.figure(figsize=(10,10))
        #plt.imshow(image)
        #show_points(prompt['point_coords'], prompt['point_labels'], plt.gca())
        #plt.axis('on')
        #plt.show()  

        #print(scores.shape)
        #print(msk.shape)
        cond = (scores < score[0]) & msk
        scores[cond] = score[0]
        result[cond] = i

    #global res2
    #res2 = result

    #global ulabel
    #ulabel = label.numpy()

    return result

print(utils.eval.evaluate(predict, dataset_3d))