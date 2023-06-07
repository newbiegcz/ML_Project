import lightning.pytorch as pl

from torch.utils.data import DataLoader
from data.cache_dataset import TrainDataset, ValidationDataset

class DataModule(pl.LightningDataModule):
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