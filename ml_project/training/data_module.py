import lightning.pytorch as pl

from torch.utils.data import DataLoader, Dataset
from ..data.cache_dataset import DiskCacheDataset
from tqdm import tqdm
import diskcache

# 分布式训练时，每个进程应当只加载自己负责的部分
class MemoryDataModule(pl.LightningDataModule):
    def __init__(self, 
                 embedding_file_path: str,
                 datapoint_file_path: str,
                 model_type: str = "vit_h",
                 batch_size: int = 128,
                 aug_per_img: int = 10,
                 total_aug_per_img: int = 10,
                 debug: bool = False
                ):
        super().__init__()

        self.model_type = model_type
        self.batch_size = batch_size
        self.aug_per_img = aug_per_img
        self.total_aug_per_img = total_aug_per_img

        _train_dataset = DiskCacheDataset(
                embedding_file_path=embedding_file_path,
                datapoint_file_path=datapoint_file_path,
                model_type=self.model_type,
                key="training",
            )
    
        _validation_dataset = DiskCacheDataset(
            embedding_file_path=embedding_file_path,
            datapoint_file_path=datapoint_file_path,
            model_type=self.model_type,
            key="validation",
        )

        self.train_image_cache = {}
        self.val_image_cache = {}

        # TODO: 这部分不多进程也太慢了..得类似 torch data iter 搞搞
        for i in tqdm(range(_train_dataset.num_image)):
            if i % self.total_aug_per_img < self.aug_per_img:
                self.train_image_cache[i] = _train_dataset.embedding_cache[("training", i)]
                #import sys
                for k in self.train_image_cache[i].keys():
                    self.train_image_cache[i][k] = self.train_image_cache[i][k].clone()
                    #print(k, sys.getsizeof(v.clone().storage()), v.shape, v.dtype, v.grad)

        for i in tqdm(range(_validation_dataset.num_image)):
            self.val_image_cache[i] = _validation_dataset.embedding_cache[("validation", i)]
            for k in self.val_image_cache[i].keys():
                self.val_image_cache[i][k] = self.val_image_cache[i][k].clone()

        self.train_datapoints = []
        self.val_datapoints = []
        for i in tqdm(range(_train_dataset.num_datapoints)):
            datapoint = _train_dataset.datapoint_cache[("training", i)]
            image_id = datapoint["image_id"]
            if image_id in self.train_image_cache:
                self.train_datapoints.append(datapoint)
        
        for i in tqdm(range(_validation_dataset.num_datapoints)):
            datapoint = _validation_dataset.datapoint_cache[("validation", i)]
            image_id = datapoint["image_id"]
            if image_id in self.val_image_cache:
                self.val_datapoints.append(datapoint)

        class _Dataset(Dataset):
            def __init__(self, image_cache, datapoints):
                super().__init__()
                self.image_cache = image_cache
                self.datapoints = datapoints
            def __len__(self):
                return len(self.datapoints)
            def __getitem__(self, idx):
                res = dict()
                img_id = self.datapoints[idx]["image_id"]
                res["embedding"] = self.image_cache[img_id]["embedding"]
                res["label"] = self.image_cache[img_id]["label"]
                res["mask_cls"] = self.datapoints[idx]["mask_cls"]
                res["prompt"] = self.datapoints[idx]["prompt_point"]
                return res
            
        self.training_dataset = _Dataset(self.train_image_cache, self.train_datapoints)
        self.validation_dataset = _Dataset(self.val_image_cache, self.val_datapoints)

    def setup(self, stage: str):
        # this is only called once on each process
        pass

    def train_dataloader(self):
        # num_workers must be 0
        # Important: shuffle must be True, otherwise the training will be wrong.
        return DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    def val_dataloader(self):
        # num_workers must be 0
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=0, pin_memory=True)

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass

class DiskDataModule(pl.LightningDataModule):
    def __init__(self, 
                 embedding_file_path: str,
                 datapoint_file_path: str,
                 model_type: str = "vit_h",
                 batch_size: int = 128,
                 num_workers: int = 4,
                 calculate_connected_mask: bool = True,
                 debug: bool = False
                ):
        super().__init__()

        self.model_type = model_type
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.embedding_cache = diskcache.Cache(embedding_file_path, eviction_policy = "none")
        self.datapoint_cache = diskcache.Cache(datapoint_file_path, eviction_policy = "none")

        self.training_dataset = DiskCacheDataset(
                embedding_cache=self.embedding_cache,
                datapoint_cache=self.datapoint_cache,
                model_type=self.model_type,
                calculate_connected_mask=calculate_connected_mask,
                key="training",
            )
    
        self.validation_dataset = DiskCacheDataset(
                embedding_cache=self.embedding_cache,
                datapoint_cache=self.datapoint_cache,
                model_type=self.model_type,
                calculate_connected_mask=calculate_connected_mask,
                key="validation",
            )
        

    def setup(self, stage: str):
        # this is only called once on each process
        pass

    def train_dataloader(self):
        # Important: shuffle must be True, otherwise the training will be wrong.
        return DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass