#do not import this!!!!
assert(__name__ == "__main__")

'''
This script is used to preprocess the data and save it to disk.

The data is saved in the following format:
    datapoints: A list of datapoints, each datapoint is a dictionary with the following keys:
        image_id: The image id(the id in the dictionary of embeddings)
        prompt_point: The prompt point
        prompt_box: The prompt box
        mask_cls: The mask class

    embeddings: A dictionary of embeddings, each embedding is a dictionary with the following keys:
        embedding: The embedding
        low_res_image: The low resolution image
        label: The label

'''
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
import utils.visualize as viz
from collections import namedtuple

checkpoint_path = "checkpoint/sam_vit_h_4b8939.pth"
datapoints_disk_path = "/root/autodl-tmp/datapoints"
embedding_disk_path = "/root/autodl-tmp/embeddings"

debug = False
times = 1000
datapoints_for_training = 100000000
datapoints_for_validation = 100000

seed_rng = torch.Generator(device='cpu')
seed_rng.manual_seed(19260817)
datapoints_cache = diskcache.Cache(datapoints_disk_path, eviction_policy = "none")
image_cache = diskcache.Cache(embedding_disk_path, eviction_policy = "none")

class TorchSeed:
    def __init__(self, seed):
        self.seed = seed
    def __enter__(self):
        # backup random states
        self.old_random_state = random.getstate()
        self.old_np_rng_state = np.random.get_state()
        self.old_rng_state = torch.random.get_rng_state()

        # set random states
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.random.manual_seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        # restore random states
        random.setstate(self.old_random_state)
        np.random.set_state(self.old_np_rng_state)
        torch.random.set_rng_state(self.old_rng_state)

def wrap_with_torchseed(func, seed=None):
    if seed is None:
        def wrapper(*args, **kwargs):
            with TorchSeed(kwargs['seed']):
                kwargs.pop('seed')
                return func(*args, **kwargs)
    else:
        def wrapper(*args, **kwargs):
            with TorchSeed(seed):
                return func(*args, **kwargs)
    return wrapper

def bounding_box(label):
    nonzero_indexes = torch.nonzero(label)
    x1 = nonzero_indexes[:, 1].min().item()
    x2 = nonzero_indexes[:, 1].max().item()
    y1 = nonzero_indexes[:, 0].min().item()
    y2 = nonzero_indexes[:, 0].max().item()
    return x1, x2, y1, y2

def is_box_valid(x1, x2, y1, y2):
    if x1 > x2 or y1 > y2:
        return False
    
    if x1 >= 0 and x1 < 1024 and y1 >= 0 and y1 < 1024 and x2 >= 0 and x2 < 1024 and y2 >= 0 and y2 < 1024:
        return True
    
    return False

def get_prompt(label, cur_label):
    result = dict()
    result["mask_cls"] = cur_label
    nonzero_indexes = torch.nonzero(label == cur_label)
    inds = torch.randint(len(nonzero_indexes), (1,), generator=seed_rng).item()
    result["prompt_point"] = torch.tensor([nonzero_indexes[inds][1], nonzero_indexes[inds][0]])

    return result

num_image_training = image_cache["num_image_for_training"]
num_image_validation = image_cache["num_image_for_validation"]
print("generating training datapoints...")

POINT = namedtuple("POINT", ["image_id", "mask_cls", "prompt_point"])

lst_points = np.zeros((num_image_training, 14, times, 2), dtype=np.int32)
label_list = [[] for _ in range(14)]
for i in tqdm(range(num_image_training)):
    label = image_cache[("training", i)]["label"][0]
    for j in range(0, 14):
        if torch.count_nonzero(label == j) > 0:
            label_list[j].append(i)
            nonzero_indexes = torch.nonzero(label == j)
            inds = torch.randint(len(nonzero_indexes), (times,), generator=seed_rng)
            for k in range(times):
                lst_points[i, j, k, 0] = nonzero_indexes[inds[k]][1]
                lst_points[i, j, k, 1] = nonzero_indexes[inds[k]][0]

for i in tqdm(range(datapoints_for_training)):
    cur_label = torch.randint(13, (1,), generator=seed_rng).item() + 1 
    
    image_index = torch.randint(len(label_list[cur_label]), (1,), generator=seed_rng).item()
    image_index = label_list[cur_label][image_index]
    id = torch.randint(times, (1,), generator=seed_rng).item()

    datum = POINT(image_id = int(image_index), mask_cls = int(cur_label), prompt_point = list(lst_points[image_index, cur_label, id]))

    datapoints_cache[("training", i)] = datum

print("training datapoints completed!The number of total datapoints is %d." % (datapoints_for_training))
datapoints_cache["num_datapoints_for_training"] = datapoints_for_training

print("generating validation datapoints...")
lst_points = np.zeros((num_image_validation, 14, times, 2), dtype=np.int32)
label_list = [[] for _ in range(14)]
for i in tqdm(range(num_image_validation)):
    label = image_cache[("validation", i)]["label"][0]
    for j in range(0, 14):
        if torch.count_nonzero(label == j) > 0:
            label_list[j].append(i)
            nonzero_indexes = torch.nonzero(label == j)
            inds = torch.randint(len(nonzero_indexes), (times,), generator=seed_rng)
            for k in range(times):
                lst_points[i, j, k, 0] = nonzero_indexes[inds[k]][1]
                lst_points[i, j, k, 1] = nonzero_indexes[inds[k]][0]

for i in tqdm(range(datapoints_for_validation)):
    cur_label = torch.randint(13, (1,), generator=seed_rng).item() + 1 
    
    image_index = torch.randint(len(label_list[cur_label]), (1,), generator=seed_rng).item()
    image_index = label_list[cur_label][image_index]
    id = torch.randint(times, (1,), generator=seed_rng).item()

    datum = POINT(image_id = int(image_index), mask_cls = int(cur_label), prompt_point = list(lst_points[image_index, cur_label, id]))

    datapoints_cache[("validation", i)] = datum


print("validation datapoints completed!The number of total datapoints is %d." % (datapoints_for_validation))
datapoints_cache["num_datapoints_for_validation"] = datapoints_for_validation

if debug:
    viz.initialize_window()

    for i in range(datapoints_cache["num_datapoints_for_training"]):
        datum = datapoints_cache[("training", i)]
        prompt_point = datum.prompt_point
        cls = datum.mask_cls
        viz.add_object_2d("image" + str(i),
                          image=image_cache[("training", datum.image_id)]["low_res_image"].squeeze(0).numpy(),
                          pd_label=None,
                          gt_label=torch.nn.functional.interpolate(image_cache[("training", datum.image_id)]["label"].unsqueeze(0), size=(256, 256), mode='nearest').numpy()[0],
                          prompt_points=[([prompt_point[0] // 4, prompt_point[1] // 4], 0)],
                          label_name=viz.default_label_names,
                             extras={
                              "prompt": prompt_point,
                              "prompt_label": viz.default_label_names[cls]
                          }
            )

datapoints_cache.close()
image_cache.close()