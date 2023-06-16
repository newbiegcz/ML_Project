import data.dataset
import model.prompter
import utils.eval
from data.dataset import data_files
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations.pytorch.transforms
from data.dataset import get_dataset_3d
from data.dataset import data_files

import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry

data_files_validation = data_files["validation"]
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


def predict(image, label):
    image *= 255.00

    image = torch.tensor(image, dtype = torch.uint8)

    x, y = image.shape

    image = image.unsqueeze(2).repeat(1, 1, 3).numpy()

    #cv2.imshow("image", image)
    #cv2.waitKey(0)

    global image2
    image2 = image
    
    x, y = label.shape

    sam_predictor.set_image(image)
    result = np.zeros(label.shape)
    scores = np.zeros(label.shape, dtype = float)
    

    for i in range(1, 14):
        mask = np.zeros(label.shape, dtype = np.uint8)
        mask[label == i] = 1
        if (np.max(mask) == 0):
            continue
        prompt = Promptor(mask, 1)

        # print(prompt)
        masks, score, logits = sam_predictor.predict(**prompt)  

        x, y, z = masks.shape
        print(x, y, z)
        assert(x == 3)
        for j in range(3):
            assert(((mask[j] == 0) | (mask[j] == 1)).all())

        max_id = 2

        plt.ion()

        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_points(prompt['point_coords'], prompt['point_labels'], plt.gca())
        plt.axis('on')
        plt.show()  


        cond = (scores < score[max_id]) & masks[max_id]
        scores[cond] = score[max_id]
        result[cond] = i

    global res2
    res2 = result

    global ulabel
    ulabel = label

    return result


print(utils.eval.evaluate(predict, dataset_3d))