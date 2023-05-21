import lightning.pytorch as pl

import numpy as np
import torch

from segment_anything.build_sam import build_sam_vit_h
from model.sam import build_sam_with_label_vit_h
from model.sam import SamWithLabel
import torch.optim as optim
import torch.nn as nn

from data.dataset import get_data_loader

# TODO: 考虑 freeze 一部分，或者直接不要丢进 optimizer

# TODO: fix hyperparameters
max_epochs = 10
batch_size = 10
optimizer_type = "AdamW"
optimizer_kwargs = {
    "lr": 1e-5,
    "weight_decay": 0.1,
}

class SAMWithLabelModule(pl.LightningModule):
    def __init__(self, model : SamWithLabel):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        assert optimizer_type in ["AdamW"]
        if optimizer_type == "AdamW":
            optimizer = optim.AdamW(self.parameters(), **optimizer_kwargs)
        return optimizer
    
def get_sam_with_label(pretrained_checkpoint):
    org_sam = build_sam_vit_h(pretrained_checkpoint)
    org_state_dict = org_sam.state_dict()
    sam_with_label = build_sam_with_label_vit_h(None)
    new_state_dict = sam_with_label.state_dict()
    for k, v in org_state_dict:
        assert k in new_state_dict.keys()
        new_state_dict[k] = v
    sam_with_label.load_state_dict(new_state_dict)
    return sam_with_label

checkpoint_path = "checkpoint/sam_vit_h_4b8939.pth"

sam_module = SAMWithLabelModule(get_sam_with_label(checkpoint_path))

train_dataloader = get_data_loader("training", "naive_to_rgb_and_resize", batch_size, True)
val_dataloader = get_data_loader("validation", "naive_to_rgb_and_resize", batch_size, False)

trainer = pl.Trainer(max_epochs=max_epochs)
trainer.fit(model=sam_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)