import lightning.pytorch as pl

from torch.utils.data import DataLoader, Dataset
from data.cache_dataset import TrainDataset, ValidationDataset

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

        _train_dataset = TrainDataset(
                embedding_file_path=embedding_file_path,
                datapoint_file_path=datapoint_file_path,
                model_type=self.model_type,
            )
    
        _validation_dataset = ValidationDataset(
            embedding_file_path=embedding_file_path,
            datapoint_file_path=datapoint_file_path,
            model_type=self.model_type,
        )

        self.train_image_cache = {}
        self.val_image_cache = {}

        # TODO: 这部分不多进程也太慢了..得类似 torch data iter 搞搞
        for i in range(_train_dataset.num_image):
            if i % self.total_aug_per_img < self.aug_per_img:
               self.train_image_cache[i] = _train_dataset.embedding_cache[("training", i)]

        for i in range(_validation_dataset.num_image):
            self.val_image_cache[i] = _validation_dataset.embedding_cache[("validation", i)]

        self.train_datapoints = []
        self.val_datapoints = []
        for i in range(_train_dataset.num_datapoints):
            datapoint = _train_dataset.datapoint_cache[("training", i)]
            image_id = datapoint["image_id"]
            if image_id in self.train_image_cache:
                self.train_datapoints.append(datapoint)
        
        for i in range(_validation_dataset.num_datapoints):
            datapoint = _validation_dataset.datapoint_cache[("validation", i)]
            image_id = datapoint["image_id"]
            if image_id in self.val_image_cache:
                self.val_datapoints.append(datapoint)

        class _Dataset(Dataset):
            def __init__(self, image_cache, datapoints, type_name):
                super().__init__()
                self.image_cache = image_cache
                self.datapoints = datapoints
                self.type_name = type_name
            def __len__(self):
                return len(self.datapoints)
            def __getitem__(self, idx):
                res = dict()
                img_id = self.datapoints[(self.type_name, idx)]["image_id"]
                res["embedding"] = self.image_cache[(self.type_name, img_id)]["embedding"]
                res["label"] = self.image_cache[(self.type_name, img_id)]["label"]
                res["mask_cls"] = self.datapoints[(self.type_name, idx)]["mask_cls"]
                res["prompt"] = self.datapoints[(self.type_name, idx)]["prompt_point"]
                return res
            
        self.training_dataset = _Dataset(self.train_image_cache, self.train_datapoints, "training")
        self.validation_dataset = _Dataset(self.val_image_cache, self.val_datapoints, "validation")

    def setup(self, stage: str):
        # this is only called once on each process
        pass

    def train_dataloader(self):
        # num_workers must be 0
        # Important: shuffle must be True, otherwise the training will be wrong.
        return DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        # num_workers must be 0
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=0)

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
                 debug: bool = False
                ):
        super().__init__()

        self.model_type = model_type
        self.batch_size = batch_size
        self.num_workers = num_workers

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