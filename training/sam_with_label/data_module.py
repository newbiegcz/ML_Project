import lightning.pytorch as pl
from data.exp_dataset import ExpDataset
from data.dataset import data_files
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import diskcache

class TrainDataset(Dataset):

    def __init__(self, *,
                    datapoint_file_path,
                    embedding_file_path,
                    model_type='vit_h'
                 ):   
        
        super().__init__()

        self.model_type = model_type

        self.embedding_cache = diskcache.Cache(embedding_file_path, eviction_policy = "none")
        self.datapoint_cache = diskcache.Cache(datapoint_file_path, eviction_policy = "none")

        self.num_image = self.embedding_cache["num_image_for_training"]
        self.num_datapoints = self.datapoint_cache["num_datapoints_for_training"]

    def __len__(self):
        return self.num_datapoints

    def __getitem__(self, idx):
        res = dict()
        res["embedding"] = self.embedding_cache[("training", self.datapoint_cache[("training", idx)]["image_id"])]["embedding"]
        res["label"] = self.embedding_cache[("training", self.datapoint_cache[("training", idx)]["image_id"])]["label"]
        res["mask_cls"] = self.datapoint_cache[("training", idx)]["mask_cls"]
        res["prompt"] = self.datapoint_cache[("training", idx)]["prompt_point"]
        return res 

    def __del__(self):
        self.embedding_cache.close()
        self.datapoint_cache.close()

class ValidationDataset(Dataset):

    def __init__(self, *,
                    datapoint_file_path,
                    embedding_file_path,
                    model_type='vit_h'
                 ):   
        
        super().__init__()

        self.model_type = model_type

        self.embedding_cache = diskcache.Cache(embedding_file_path, eviction_policy = "none")
        self.datapoint_cache = diskcache.Cache(datapoint_file_path, eviction_policy = "none")

        self.num_image = self.embedding_cache["num_image_for_validation"]
        self.num_datapoints = self.datapoint_cache["num_datapoints_for_validation"]

    def __len__(self):
        return self.num_datapoints

    def __getitem__(self, idx):
        res = dict()
        res["embedding"] = self.embedding_cache[("validation", self.datapoint_cache[("validation", idx)]["image_id"])]["embedding"]
        res["label"] = self.embedding_cache[("validation", self.datapoint_cache[("validation", idx)]["image_id"])]["label"]
        res["mask_cls"] = self.datapoint_cache[("validation", idx)]["mask_cls"]
        res["prompt"] = self.datapoint_cache[("validation", idx)]["prompt_point"]
        return res 

    def __del__(self):
        self.embedding_cache.close()
        self.datapoint_cache.close()


# TODO: 在实现多进程训练时，setup 会被每个进程调用一次。应该考虑让所有进程共享一个 queue，而 workers 负责分配不同的数据生成任务.
class DataModule(pl.LightningDataModule):
    def __init__(self, 
                 embedding_file_path,
                 datapoint_file_path,
                 model_type: str = "vit_h",
                 batch_size: int = 128,
                 debug: bool = False
                ):
        super().__init__()

        self.model_type = model_type
        self.batch_size = batch_size

        # TODO: This will keep two copies of the encoder in memory. We should find a way to avoid this.
        self.training_dataset = TrainDataset(
                embedding_file_path=embedding_file_path,
                datapoint_file_path=datapoint_file_path,
                model_type=self.model_type,
            )
    
        self.validation_dataset = ValidationDataset(
            embedding_file_path=embedding_file_path,
            datapoint_file_path=datapoint_file_path,
            model_type=self.model_type,
        )
        

    def setup(self, stage: str):
        # this is only called once on each process
        pass

    def train_dataloader(self):
        # Important: shuffle must be True, otherwise the training will be wrong.
        return DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=5)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=5)

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass