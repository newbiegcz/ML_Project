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
import utils.visualize as viz
from tqdm import tqdm

checkpoint_path = "checkpoint/sam_vit_h_4b8939.pth"
datapoints_disk_path = "processed_data/datapoints"
embedding_disk_path = "processed_data/embeddings"

debug = True
times = 20 # The number of times to augment an image
datapoints = 100000 # The number of datapoints
min_pixels = 5


datapoints_cache = diskcache.Cache(datapoints_disk_path)
image_cache = diskcache.Cache(embedding_disk_path)

encoder = build_pretrained_encoder("vit_h", eval=True)

all_cmaps = plt.colormaps()
exclude = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                      'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
                      'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
                      'turbo', 'nipy_spectral', 'gist_ncar'] + ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                      'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                      'tab20c']
exclude_ = []
for e in exclude:
    exclude_.append(e + "_r")
exclude += exclude_

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

class RandCmap:
    cmaps = [plt.get_cmap(c) for c in all_cmaps if c not in exclude]
    def __init__(self, use_torch=True):
        self.use_torch = use_torch
    def __call__(self, x, **kwargs):
        if len(x.shape) == 3:
            if self.use_torch:
                assert x.shape[0] == 1
                x = x[0]
            else :
                assert x.shape[2] == 1
                x = x[:, :, 0]
        y = self.cmaps[torch.randint(0, len(self.cmaps), (1,))](x)
        if self.use_torch:
            y = y[:, :, :3].transpose(2, 0, 1) # remove alpha channel and move channel to the front
            return torch.from_numpy(y)
        else:
            return y[:, :, :3]

def wrap_albumentations_transform(transform):
    def wrapper(x):
        res = transform(image=x['image'], mask=x['label'])
        return {
            'image': res['image'],
            'label': res['mask']
        }
    return wrapper

def unsqueeze(x, **kwargs):
    return x.reshape(x.shape + (1,))
            
def gen_clache(**kwargs):
    return albumentations.Compose([
        albumentations.FromFloat(dtype="uint8"),
        albumentations.CLAHE(**kwargs),
        albumentations.ToFloat()
    ])

transform_2d = (
    wrap_with_torchseed(
        torchvision.transforms.Compose([
            DictTransform(["image", "label"], lambda x : x.numpy()),
            wrap_albumentations_transform(
                albumentations.Compose([
                    albumentations.Lambda(image=unsqueeze, mask=unsqueeze),
                    albumentations.RandomBrightnessContrast(p=0.2),
                    albumentations.Lambda(image=RandCmap(use_torch=False)),
                    gen_clache(p=0.8),
                    albumentations.RandomBrightnessContrast(p=0.8),
                    albumentations.RandomGamma(p=0.8),
                    albumentations.HorizontalFlip(p=0.5),
                    albumentations.VerticalFlip(p=0.5),
                    albumentations.RandomRotate90(p=0.5),
                    albumentations.OneOf([
                        albumentations.CropNonEmptyMaskIfExists(256, 256, p=1.),
                    ], p=.5),
                    albumentations.pytorch.transforms.ToTensorV2(transpose_mask=True)
                ])
            ),
            PreprocessForModel(normalize=True),
        ])
    )
)

seed_rng = torch.Generator(device='cpu')
seed_rng.manual_seed(19260817)
data_files = data_files["training"][:1]  #only the first person
raw_dataset = Dataset2D(data_files, device=torch.device('cpu'), transform=None, dtype=np.float32)

def gen(idx):
    image_seed = torch.randint(1000000, (1,), generator=seed_rng).item()
    with TorchSeed(image_seed):
        d = transform_2d(raw_dataset[idx], seed=image_seed+1)
    return d

raw_dataset = raw_dataset[40 : 41]

print("doing data augmentation...")

image_list = []
for i in range(len(raw_dataset)):
    for j in range(times):
        image_list.append(gen(i))

print("data augmentation completed!")

print("doing image encoding...")

num_images = len(image_list)

import torch.nn.functional

batch_size = 1

for i in tqdm(range(0, num_images, batch_size)):
    if i + batch_size > num_images:
        break
    img_lst = []
    for j in range(i, i + batch_size):
        img_lst.append(image_list[j]["image"])

    img = torch.stack(img_lst)

    with torch.inference_mode():
        embeddings = encoder(img)

    for j in range(batch_size):
        cur = dict()
        cur["embedding"] = embeddings[j]
        cur["low_res_image"] = torch.nn.functional.interpolate(img[j].unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False) * PreprocessForModel.pixel_std + PreprocessForModel.pixel_mean
        cur["low_res_label"] = torch.nn.functional.interpolate(image_list[i + j]["label"].unsqueeze(0), size=(256, 256), mode='nearest')
        cur["low_res_label"] = torch.tensor(cur["low_res_label"], dtype=torch.uint8)
        image_cache[i + j] = cur

print("image encoding completed!")

num_datapoints = 0

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
    result["prompt_point"] = [nonzero_indexes[inds][1], nonzero_indexes[inds][0]]
    x1, x2, y1, y2 = bounding_box(label == cur_label)
    lenx = x2 - x1
    leny = y2 - y1
    dx1, dx2 = np.random.normal(0, lenx * 0.1, 2)
    dy1, dy2 = np.random.normal(0, leny * 0.1, 2)
    x1 += min(20, max(-20, int(dx1)))
    x2 += min(20, max(-20, int(dx2)))
    y1 += min(20, max(-20, int(dy1)))
    y2 += min(20, max(-20, int(dy2)))
    if is_box_valid(x1, x2, y1, y2):  
        result["prompt_box"] = [[x1, y1], [x2, y2]]
    else:
        result["prompt_box"] = None

    return result

print("generating datapoints...")
label_list = [[] for _ in range(14)]
for i in range(num_images):
    for j in range(1, 14):
        if torch.count_nonzero(image_list[i]["label"][0] == j) >= min_pixels:
            label_list[j].append(i)

for i in tqdm(range(datapoints)):
    while True:
        cur_label = torch.randint(13, (1,), generator=seed_rng).item() + 1
        if len(label_list[cur_label]) > 0:
            break

    image_index = torch.randint(len(label_list[cur_label]), (1,), generator=seed_rng).item()
    image_index = label_list[cur_label][image_index]

    datum = get_prompt(image_list[image_index]["label"][0], cur_label)
    datum["image_id"] = image_index
    datapoints_cache[i] = datum

print("datapoints completed!")

if debug:
    viz.initialize_window()

    for i in range(datapoints):
        datum = datapoints_cache[i]
        prompt_point = datum["prompt_point"]
        prompt_box = datum["prompt_box"]
        cls = datum["mask_cls"]
        if prompt_box is None:
            viz.add_object_2d("image" + str(i),
                          image=image_cache[datum['image_id']]["low_res_image"].squeeze(0).numpy(),
                          pd_label=None,
                          gt_label=image_cache[datum['image_id']]["low_res_label"][0].numpy(),
                          prompt_points=[([prompt_point[0] // 4, prompt_point[1] // 4], 0)],
                          label_name=viz.default_label_names,
                             extras={
                              "prompt": prompt_point,
                              "prompt_label": viz.default_label_names[cls]
                          }
            )
        else:
            viz.add_object_2d("image" + str(i),
                          image=image_cache[datum['image_id']]["low_res_image"].squeeze(0).numpy(),
                          pd_label=None,
                          gt_label=image_cache[datum['image_id']]["low_res_label"][0].numpy(),
                          prompt_points=[([prompt_point[0] // 4, prompt_point[1] // 4], 0), ([prompt_box[0][0] // 4, prompt_box[0][1] // 4], 0), ([prompt_box[1][0] // 4, prompt_box[1][1] // 4], 0)],
                          label_name=viz.default_label_names,
                             extras={
                              "prompt": prompt_point,
                              "prompt_label": viz.default_label_names[cls]
                          }
            )

datapoints_cache.close()
image_cache.close()