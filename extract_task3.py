import argparse
import os
import lightning.pytorch as pl
import torch
from ml_project.training.task2.model_module import SAMWithLabelModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract model from checkpoint')
    parser.add_argument('input', metavar='input', type=str, help='path to input file')
    parser.add_argument('output', metavar='output', type=str, help='path to output file')
    args = parser.parse_args()

    checkpoint_path = args.input

    if not os.path.exists(checkpoint_path):
        print("Checkpoint path does not exist")
        exit(1)

    if os.path.exists(args.output):
        print("Output path already exists")
        exit(1)
    
    if not os.path.exists(os.path.dirname(args.output)):
        print("Output directory does not exist")
        exit(1)

    sam_module = SAMWithLabelModule.load_from_checkpoint(checkpoint_path)
    sam_model = sam_module.model
    sam_model.to('cpu')

    torch.save(sam_model.state_dict(), args.output)
