import argparse

parser = argparse.ArgumentParser(description='Extract model from checkpoint')
parser.add_argument('input', metavar='input', type=str, help='path to input file')
parser.add_argument('output', metavar='output', type=str, help='path to output file')
args = parser.parse_args()

import lightning.pytorch as pl
import torch

checkpoint_path = "/root/ML_Project/experiment_logs/SAM with Labels/uwg9vnbz/checkpoints/epoch=15-step=100000.ckpt"

from training.sam_with_label.model_module import SAMWithLabelModule
sam_module = SAMWithLabelModule.load_from_checkpoint(checkpoint_path)
import torch
sam_model = sam_module.model
sam_model.to('cpu')
torch.save(sam_model.state_dict(), "extracted.pth")
