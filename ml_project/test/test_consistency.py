from data.cache_dataset import DiskCacheDataset
import random
import numpy as np
from data.dataset import get_dataset_3d
import cv2

embedding_file_path = "/root/autodl-tmp/"
datapoint_file_path = "/root/autodl-tmp/"
model_type = "vit_h"

def check_dataset(dataset, dataset_3d, n_aug,):
    dataset_len = len(dataset)
    i = random.randint(0, dataset_len - 1)
    datapoint = dataset[i]
    image_id = dataset.datapoint_cache[(dataset.key, i)].image_id
    p1 = np.array(datapoint["3d"])

    slice_id = image_id // n_aug
    _ = 0
    for ct_id in range(len(dataset_3d)):
        if slice_id < _ + dataset_3d[ct_id]['image'].shape[-1]:
            break
        _ += dataset_3d[ct_id]['image'].shape[-1]
    
    cv2.imshow("p1", image_id)
    cv2.waitKey(0)

train_dataset = DiskCacheDataset(
    embedding_file_path=embedding_file_path,
    datapoint_file_path=datapoint_file_path,
    model_type=model_type,
    key="training"
)
val_dataset = DiskCacheDataset(
    embedding_file_path=embedding_file_path,
    datapoint_file_path=datapoint_file_path,
    model_type=model_type,
    key="validation"
)

train_dataset_3d = get_dataset_3d("training", crop_roi=True, spacing=True)
val_dataset_3d = get_dataset_3d("validation", crop_roi=True, spacing=True)

check_dataset(train_dataset, train_dataset_3d, n_aug=15)
check_dataset(val_dataset, train_dataset_3d, n_aug=1)