import lightning.pytorch as pl
from data.exp_dataset import ExpDataset
from data.dataset import data_files
import torch
from torch.utils.data import DataLoader

size_10gb = 10 * 1024 * 1024 * 1024

# TODO: 在实现多进程训练时，setup 会被每个进程调用一次。应该考虑让所有进程共享一个 queue，而 workers 负责分配不同的数据生成任务.
class DataModule(pl.LightningDataModule):
    def __init__(self, 
                 model_type: str = "vit_h",
                 training_epoch_len: int = 10000,
                 validation_epoch_len: int = 1000,
                 chunk_size: int = 512,
                 batch_size: int = 128,
                 encoder_batch_size: int = 2,
                 encoder_device: str = "cuda:0",
                 cache_path: str = "embedding_cache",
                 cache_size_limit: int = size_10gb,
                 seed: int = 1166117,
                 delay: int = 128,
                 augment_training_data: bool = True,
                 augment_validation_data: bool = True,
                 debug: bool = False
                ):
        super().__init__()

        self.model_type = model_type
        self.training_epoch_len = training_epoch_len
        self.validation_epoch_len = validation_epoch_len
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.encoder_batch_size = encoder_batch_size
        self.encoder_device = torch.device(encoder_device)
        self.cache_path = cache_path
        self.cache_size_limit = cache_size_limit
        self.seed = seed
        self.delay = delay
        self.augment_training_data = augment_training_data
        self.augment_validation_data = augment_validation_data
        self.debug = debug

        data_files_training = data_files["training"]
        data_files_validation = data_files["validation"]
        if self.debug:
            data_files_training = data_files_training[:1]
            data_files_validation = data_files_validation[:1]

        # TODO: This will keep two copies of the encoder in memory. We should find a way to avoid this.
        self.training_dataset = ExpDataset(
                data_files=data_files_training,
                epoch_len=self.training_epoch_len,
                chunk_size=self.chunk_size,
                encoder_device=self.encoder_device,
                encoder_batch_size=self.encoder_batch_size,
                model_type=self.model_type,
                size_limit=self.cache_size_limit,
                path=self.cache_path,
                seed=self.seed,
                delay=self.delay,
                augment_data=self.augment_training_data,
                debug=self.debug
            )
    
        self.validation_dataset = ExpDataset(
            data_files=data_files_validation,
            epoch_len=self.validation_epoch_len,
            chunk_size=self.chunk_size,
            encoder_device=self.encoder_device,
            encoder_batch_size=self.encoder_batch_size,
            model_type=self.model_type,
            size_limit=self.cache_size_limit,
            path=self.cache_path,
            seed=self.seed,
            delay=self.delay,
            augment_data=self.augment_validation_data,
            debug=self.debug
        )
        

    def setup(self, stage: str):
        # this is only called once on each process
        pass

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass