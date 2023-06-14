from collections import namedtuple
from torch.utils.data import Dataset

class DiskCacheDataset(Dataset):

    def __init__(self, *,
                    datapoint_cache,
                    embedding_cache,
                    key: str,
                    model_type='vit_h'
                 ):   
        
        super().__init__()

        assert(model_type == 'vit_h'), "Only vit_h is supported for now."

        self.model_type = model_type
        self.key = key

        self.embedding_cache = embedding_cache

        self.datapoint_cache = datapoint_cache

        self.num_image = self.embedding_cache["num_image_for_" + self.key]
        self.num_datapoints = self.datapoint_cache["num_datapoints_for_" + self.key]

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
        return res