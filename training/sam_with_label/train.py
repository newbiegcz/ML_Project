from typing import Any, Dict
from lightning.pytorch import Trainer
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from training.sam_with_label.model_module import SAMWithLabelModule
from training.sam_with_label.data_module import DataModule
from lightning.pytorch.loggers.wandb import WandbLogger
import torch.multiprocessing as multiprocessing

# TODO: 优化 optimizer 的内存 (parameters & 更好的优化器)
# TODO: 降低精度 && warning 中的建议
# TODO: 考虑 freeze 一部分，或者直接不要丢进 optimizer
# TODO: 允许 Encoder 的微调 (现在的问题是 vram 不太够)
# TODO: Finetune 时的 Prompt 改为 多点、矩形、或者一个 mask && 注意考虑 negative 的店
# TODO: 正确考虑选在 background 时的处理。分为整个 background 不合适，现在没有计算 Loss。
#           可以考虑对数据集预先分割，又或者限制选在 background 时和原本模型不能相差太多
# TODO: 仔细阅读原论文训练方法 (drop path?)  learning rate decay? .... 
# TODO: 按 mask 抽取的 prompt
# TODO: 尝试加入位置信息

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.model_type", "data.model_type")
        parser.link_arguments("model.debug", "data.debug")

class MyTrainer(Trainer):
    def __init__(self, wandb_config: Any, **kwargs):
        logger = WandbLogger(**wandb_config)
        super().__init__(logger=logger, **kwargs)

def cli_main():
    cli = MyLightningCLI(SAMWithLabelModule, DataModule,
        trainer_class=MyTrainer,
        parser_kwargs={
            "default_config_files": ["training/sam_with_label/config.yaml"],   
        },
        save_config_callback=None)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    cli_main()
    