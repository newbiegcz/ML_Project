from collections import namedtuple
from typing import Any, Dict
from lightning.pytorch import Trainer
from lightning.pytorch.cli import LightningCLI
from ml_project.training.task2.model_module import SAMWithInteractiveTraining
from ml_project.training.data_module import DiskDataModule
from lightning.pytorch.loggers.wandb import WandbLogger
import torch.multiprocessing as multiprocessing

POINT = namedtuple("POINT", ["image_id", "mask_cls", "prompt_point"])

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.model_type", "data.model_type")
        parser.link_arguments("model.debug", "data.debug")

class MyTrainer(Trainer):
    def __init__(self, wandb_config: Any, **kwargs):
        logger = WandbLogger(**wandb_config)
        super().__init__(logger=logger, **kwargs)

def cli_main():
    cli = MyLightningCLI(SAMWithInteractiveTraining, DiskDataModule,
        trainer_class=MyTrainer,
        parser_kwargs={
            "default_config_files": ["configs/task2_config.yaml"],   
        },
        save_config_callback=None)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    cli_main()
    