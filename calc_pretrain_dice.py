import data.dataset
import model.prompter
import utils.eval
from data.dataset import data_files
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations.pytorch.transforms
from data.dataset import Dataset2D
from data.dataset import data_files

from monai.data import (
    load_decathlon_datalist,
    set_track_meta,
    ThreadDataLoader,
    CacheDataset
)
import torch.utils.data as data

from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry

class Dataset2D(data.Dataset):
    def __init__(self, files, *, device, transform, dtype=np.float64, first_only=False, compress=False):
        if first_only:
            files = files.copy()[:1]

        self.files = files
        self.device = device
        self.transform = transform
        set_track_meta(True)
        if compress:
            _default_transform = Compose(
                [
                    LoadImaged(keys=["image", "label"], ensure_channel_first=True, dtype=dtype),
                    ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True, dtype=dtype),
                    CropForegroundd(keys=["image", "label"], source_key="image", dtype=dtype),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    EnsureTyped(keys=["image", "label"], device=self.device, track_meta=False, dtype=dtype),
                ]
            )
        else:
            _default_transform = Compose(
                [
                    LoadImaged(keys=["image", "label"], ensure_channel_first=True, dtype=dtype),
                    ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True, dtype=dtype),
                    CropForegroundd(keys=["image", "label"], source_key="image", dtype=dtype),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    EnsureTyped(keys=["image", "label"], device=self.device, track_meta=False, dtype=dtype),
                ]
            )
        
        self.cache = CacheDataset(
            data=files, 
            transform=_default_transform, 
            cache_rate=1.0, 
            num_workers=4
        )
        set_track_meta(False)

        self.data_list = []
        for d in self.cache:
            # print(d['image_meta_dict']['filename_or_obj']) ## raw_data\imagesTr\img0035.nii.gz
            cur_list = []
            img, label = d['image'][0], d['label'][0]
            h = img.shape[2]
            # print(compress, img.shape)
            for i in range(h):
                cur_list.append({
                    "image": img[:, :, i],
                    "label": label[:, :, i],
                    "h": i / h
                })
            self.data_list.append(cur_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, id, idx):
        ret = self.data_list[id][idx].copy()
        #print('image shape:', ret['image'].shape)
        #print('label shape:', ret['label'].shape)
        if self.transform:
            t = self.transform(ret)
            #print(t['image'].shape, t['label'].shape)
            return t
        else:
            return ret

data_files_validation = data_files["validation"]
raw_dataset_validation = Dataset2D(data_files_validation, device=torch.device('cpu'), transform=None, dtype=np.float32, compress = True)
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

    #for i in range(x):
    #    for j in range(y):
    #        if i == 46 and j == 75:
    #            print("star", end = ' ')
    #        else:
    #            print(label[i][j], end = ' ')
    #    print()


    sam_predictor.set_image(image)
    result = np.zeros(label.shape)
    scores = np.zeros(label.shape, dtype = float)
    

    for i in range(1, 14):
        mask = np.zeros(label.shape, dtype = np.uint8)
        mask[label == i] = 1
        if (np.max(mask) == 0):
            continue
        prompt = Promptor(mask, 1)
        #for j in range(mask.shape[0]):
        #    for k in range(mask.shape[1]):
        #        print(mask[j][k], end = ' ')
        #    print()
        print(prompt)
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


print(utils.eval.evaluate(predict, raw_dataset_validation))