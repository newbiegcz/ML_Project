import lightning.pytorch as pl
import torch
checkpoint_path = "/root/ML_Project/experiment_logs/SAM with Labels/rsu7ongn/checkpoints/epoch=1-step=6250.ckpt"
from training.sam_with_label.model_module2 import SAMWithInteractiveTraining
sam_module = SAMWithInteractiveTraining.load_from_checkpoint(checkpoint_path)
import torch
sam_model = sam_module.model
sam_model.to('cpu')
torch.save(sam_model.state_dict(), "extracted.pth")
