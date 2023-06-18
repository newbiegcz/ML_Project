import diskcache
import torch
from torch.utils.data import Dataset
from torch.multiprocessing import Queue
import numpy as np
import torchvision
from collections import deque
import matplotlib.pyplot as plt
import random
import albumentations 
import albumentations.pytorch.transforms
from data.dataset import PreprocessForModel
from data.dataset import DictTransform
import cv2
import rich
from data.dataset import Dataset2D
from modeling.build_sam import build_pretrained_encoder
from data.dataset import data_files
from tqdm import tqdm

data_files_training = data_files["training"][:1]
raw_dataset_training = Dataset2D(data_files_training, device=torch.device('cpu'), transform=None, dtype=np.float32, compress = True)

print(len(raw_dataset_training))