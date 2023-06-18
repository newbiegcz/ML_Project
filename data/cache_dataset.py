from collections import namedtuple
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np

import cv2
import torch
import numpy as np

import cv2
import torch
import numpy as np

class DiskCacheDataset(Dataset):

    def __init__(self, *,
                    datapoint_cache,
                    embedding_cache,
                    key: str,
                    model_type='vit_h',
                    calculate_connected_mask=False,
                 ):   
        
        super().__init__()

        assert(model_type == 'vit_h'), "Only vit_h is supported for now."

        self.model_type = model_type
        self.key = key

        self.embedding_cache = embedding_cache

        self.datapoint_cache = datapoint_cache

        self.num_image = self.embedding_cache["num_image_for_" + self.key]
        self.num_datapoints = self.datapoint_cache["num_datapoints_for_" + self.key]
        
        self.calculate_connected_mask = True

        self.calculate_connected_mask = True

    def __len__(self):
        return self.num_datapoints

    def __getitem__(self, idx):
        res = dict()
        im_id = self.datapoint_cache[(self.key, idx)].image_id
        res["embedding"] = self.embedding_cache[(self.key, im_id)]["embedding"]
        res["label"] = self.embedding_cache[(self.key, im_id)]["label"]
        res["mask_cls"] = self.datapoint_cache[(self.key, idx)].mask_cls
        res["prompt"] = [self.datapoint_cache[(self.key, idx)].prompt_point[0], self.datapoint_cache[(self.key, idx)].prompt_point[1]]
        res["3d"] = [self.datapoint_cache[(self.key, idx)].prompt_point[2], self.datapoint_cache[(self.key, idx)].prompt_point[3], self.embedding_cache[(self.key, im_id)]["h"]]
        
        if self.calculate_connected_mask:
            mask = res["label"] == res["mask_cls"]
            assert mask.dim() == 3
            mask = mask[0].numpy().astype(np.uint8)
            col = cv2.connectedComponents(mask)[1]
            mask = torch.from_numpy(col == col[int(res["prompt"][1])][int(res["prompt"][0])])
            assert(res["label"][0][int(res["prompt"][1])][int(res["prompt"][0])] == res["mask_cls"])
            res["connected_mask"] = mask.unsqueeze(0)

        return res
    
