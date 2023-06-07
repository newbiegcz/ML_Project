from modeling.sam import SamWithLabel
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from utils.automatic_label_generator import SamAutomaticLabelGenerator
from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    EnsureTyped,
)

from monai.data import (
    load_decathlon_datalist,
    set_track_meta,
    CacheDataset
)

from data.dataset import DictTransform, PreprocessForModel
import torchvision
import os

class LabelPredicter():
    """
    predict the labels of CT data, using grid points as the prompt.
    """

    def __init__(self, sam_model : SamWithLabel):
        """
        Arguments:
        sam_model is a SamWithLabel model;
        """
        self.model = sam_model
        self.automatic_label_generator = SamAutomaticLabelGenerator(self.model)
        #print(self.model.device)

    def predict_one(self, images : List[np.ndarray], ground_truths : List[np.ndarray]):
        """
        predict single CT data.

        Arguments:
        images: a list of np.ndarray, representing a single CT data, each element is a 2D image slice
        each element in the list has shape (H, W, C), where H, W, C is the height, width and channel of the images
        images are in HWC uint8 format, with pixel values in [0, 255]

        ground_truths: a list of np.ndarray, with length same as `images`
        each element in the list has shape (H, W) and type np.uint8

        Returns:
        res: a list of np.ndarray, each element is a 2D mask, with shape (H, W).
        dice: a np.ndarray of length 14, indicating dice
        """
        res = []
        intersection = np.zeros(13,dtype=np.uint64)
        union = np.zeros(13,dtype=np.uint64)
        for (image, ground_truth) in tqdm(zip(images, ground_truths), desc="slice"):
            labels = self.automatic_label_generator.generate_labels(image)
            res.append(labels)
            for i in range(13):
                prediction_mask = labels == (i+1)
                ground_truth_mask = ground_truth == (i+1)
                intersection[i] += 2 * np.sum(prediction_mask & ground_truth_mask)
                union[i] += np.sum(prediction_mask) + np.sum(ground_truth_mask)
        dice = np.zeros(13, dtype=np.float64)
        for i in range(13):
            if union[i] != 0:
                dice[i] = intersection[i] / union[i]
        return res, dice
    
    def predict(self, file_key = 'validation', data_list_file_path = 'raw_data/dataset_0.json'):
        """
        predict all CT data in training set or validation set.

        Arguments:
        file_key: the key of the file to load, should be one of "training" and "validation"
        data_list_file_path: the path of the data list file

        Results:
        save the predicted labels and dice to file result/img00xx.npy
        two numpy arrays will be saved:
        1. the predicted labels, with shape (num_slices, H, W)
        2. dice, with shape (13, )
        see code for more details

        print the mean dice
        """
        # get file names
        files = load_decathlon_datalist(data_list_file_path, True, file_key)
        
        # get transform
        transform = torchvision.transforms.Compose(
            [DictTransform(["image", "label"], torchvision.transforms.Lambda(lambda x: x.unsqueeze(0).repeat(3, 1, 1))),
            PreprocessForModel(normalize=False)]
        )

        # read files
        set_track_meta(True)
        _default_transform = Compose(
            [
                LoadImaged(keys=["image", "label"], ensure_channel_first=True, dtype=np.float64),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True, dtype=np.float64),
                CropForegroundd(keys=["image", "label"], source_key="image", dtype=np.float64),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                EnsureTyped(keys=["image", "label"], track_meta=False, dtype=np.float64),
            ]
        )
        cache = CacheDataset(
            data=files, 
            transform=_default_transform, 
            cache_rate=1.0, 
            num_workers=4
        )
        set_track_meta(False)

        # predict
        dices = []
        for d in tqdm(cache, desc="CT"):
            file_path = d['image_meta_dict']['filename_or_obj']
            file_name = os.path.basename(file_path)
            index_of_dot = file_name.index('.')
            file_name_without_extension = file_name[:index_of_dot] # img0035

            images, labels = d['image'][0], d['label'][0]
            h = images.shape[2]
            images_list = []
            ground_truths_list = []

            print('reading data...')
            for i in range(h):
                # print('{}/{}'.format(i,h))
                data = {
                    "image": images[:, :, i],
                    "label": labels[:, :, i],
                    "h": i / h
                }
                data = transform(data)
                image = data['image'].numpy().transpose(1, 2, 0)
                image = (image*255).astype(np.uint8)
                label = data['label'][0].numpy()
                images_list.append(image)
                ground_truths_list.append(label)

            # predict for single CT data
            print('predicting one...')
            labels, dice = self.predict_one(images_list, ground_truths_list)
            labels = np.array(labels)
            dices.append(np.mean(dice))
            #print(labels.shape)
            #print(dice.shape)
            #print(dice)

            # save labels and dice to file
            if not os.path.exists('result'):
                os.mkdir('result')
            np.save(f'result/{file_name_without_extension}.npy', labels)
            np.save(f'result/{file_name_without_extension}_dice.npy', dice)
        
        # output mdice
        print('mdice: {}'.format(np.mean(dices)))
            


