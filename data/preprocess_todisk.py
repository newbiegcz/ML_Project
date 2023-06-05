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
from sys import getsizeof
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

size_threshold_in_bytes= 200 * 1024 * 1024 * 1024 # 200 GB
debug = True
times = 3 # The number of times to augment an image
datapoints_for_training = 200 # The number of datapoints to use for training
datapoints_for_validation = 100 # The number of datapoints to use for validation
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

transform_2d_for_training = (
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

transform_2d_for_validation = (
    wrap_with_torchseed(
        torchvision.transforms.Compose([
            DictTransform(["image", "label"], lambda x : x.numpy()),
            wrap_albumentations_transform(
                albumentations.Compose([
                    albumentations.Lambda(image=unsqueeze, mask=unsqueeze),
                    gen_clache(p=1.0),
                    albumentations.pytorch.transforms.ToTensorV2(transpose_mask=True)
                ])
            ),
            PreprocessForModel(normalize=True),
        ])
    )
)

seed_rng = torch.Generator(device='cpu')
seed_rng.manual_seed(19260817)
data_files_training = data_files["training"][:1]
data_files_validation = data_files["validation"][:1]
raw_dataset_training = Dataset2D(data_files_training, device=torch.device('cpu'), transform=None, dtype=np.float32, compress = True)
raw_dataset_validation = Dataset2D(data_files_validation, device=torch.device('cpu'), transform=None, dtype=np.float32, compress = True)
raw_dataset_training = raw_dataset_training[40 : 50]
raw_dataset_validation = raw_dataset_validation[40 : 50]

def gen_training(idx):
    image_seed = torch.randint(1000000, (1,), generator=seed_rng).item()
    with TorchSeed(image_seed):
        d = transform_2d_for_training(raw_dataset_training[idx], seed=image_seed+1)
    return d

def gen_validation(idx):
    image_seed = torch.randint(1000000, (1,), generator=seed_rng).item()
    with TorchSeed(image_seed):
        d = transform_2d_for_validation(raw_dataset_validation[idx], seed=image_seed+1)
    return d

print("doing image encoding for training...")
import torch.nn.functional

batch_size = 2

img_list = []
label_list = []
num_image_training = 0

for i in tqdm(range(len(raw_dataset_training))):
    for j in range(times):
        image = gen_training(i)
        img_list.append(image["image"])
        label_list.append(image["label"])
        if (len(img_list) >= batch_size):
            img = torch.stack(img_list)

            with torch.inference_mode():
                embeddings = encoder(img)

            for k in range(batch_size):
                cur = dict()
                cur["embedding"] = embeddings[k]
                cur["low_res_image"] = torch.nn.functional.interpolate(img[k].unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False) * PreprocessForModel.pixel_std + PreprocessForModel.pixel_mean
                cur["label"] = torch.tensor(label_list[k], dtype=torch.uint8)
                image_cache[("training", num_image_training + k)] = cur
    
            num_image_training += batch_size
            img_list = []
            label_list = []

        if image_cache.volume() > size_threshold_in_bytes:
            break

    if image_cache.volume() > size_threshold_in_bytes:
        break

print("encoding for training data done by encode %d embeddings, the total estimate number of image is %d." % (num_image_training, len(raw_dataset_training) * times))
image_cache["num_image_for_training"] = num_image_training

print("doing image encoding for validation...")

img_list = []
label_list = []
num_image_validation = 0

for i in tqdm(range(len(raw_dataset_validation))):
    image = gen_validation(i)
    img_list.append(image["image"])
    label_list.append(image["label"])
    if (len(img_list) >= batch_size):
        img = torch.stack(img_list)

        with torch.inference_mode():
            embeddings = encoder(img)

        for k in range(batch_size):
            cur = dict()
            cur["embedding"] = embeddings[k]
            cur["low_res_image"] = torch.nn.functional.interpolate(img[k].unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False) * PreprocessForModel.pixel_std + PreprocessForModel.pixel_mean
            cur["label"] = torch.tensor(label_list[k], dtype=torch.uint8)
            image_cache[("validation", num_image_validation + k)] = cur
    
        num_image_validation += batch_size
        img_list = []
        label_list = []

print("validation encoding done! The total number of image is %d." % (num_image_validation))
image_cache["num_image_for_validation"] = num_image_validation

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
    result["prompt_point"] = torch.tensor([nonzero_indexes[inds][1], nonzero_indexes[inds][0]])

    return result

print("generating training datapoints...")
label_list = [[] for _ in range(14)]
for i in range(num_image_training):
    for j in range(1, 14):
        if torch.count_nonzero(image_cache[("training", i)]["label"][0] == j) >= min_pixels:
            label_list[j].append(i)

for i in tqdm(range(datapoints_for_training)):
    while True:
        cur_label = torch.randint(13, (1,), generator=seed_rng).item() + 1
        if len(label_list[cur_label]) > 0:
            break

    image_index = torch.randint(len(label_list[cur_label]), (1,), generator=seed_rng).item()
    image_index = label_list[cur_label][image_index]

    datum = get_prompt(image_cache[("training", image_index)]["label"][0], cur_label)
    datum["image_id"] = image_index
    
    datapoints_cache[("training", i)] = datum

print("training datapoints completed!The number of total datapoints is %d." % (datapoints_for_training))
datapoints_cache["num_datapoints_for_training"] = datapoints_for_training

print("generating validation datapoints...")
label_list = [[] for _ in range(14)]
for i in range(num_image_validation):
    for j in range(1, 14):
        if torch.count_nonzero(image_cache[("validation", i)]["label"][0] == j) >= min_pixels:
            label_list[j].append(i)

for i in tqdm(range(datapoints_for_validation)):
    while True:
        cur_label = torch.randint(13, (1,), generator=seed_rng).item() + 1
        if len(label_list[cur_label]) > 0:
            break

    image_index = torch.randint(len(label_list[cur_label]), (1,), generator=seed_rng).item()
    image_index = label_list[cur_label][image_index]

    datum = get_prompt(image_cache[("validation", image_index)]["label"][0], cur_label)
    datum["image_id"] = image_index
    
    datapoints_cache[("validation", i)] = datum

print("validation datapoints completed!The number of total datapoints is %d." % (datapoints_for_validation))
datapoints_cache["num_datapoints_for_validation"] = datapoints_for_validation

if debug:
    viz.initialize_window()

    for i in range(datapoints_cache["num_datapoints_for_validation"]):
        datum = datapoints_cache[("validation", i)]
        prompt_point = datum["prompt_point"]
        cls = datum["mask_cls"]
        viz.add_object_2d("image" + str(i),
                          image=image_cache[("validation", datum['image_id'])]["low_res_image"].squeeze(0).numpy(),
                          pd_label=None,
                          gt_label=torch.nn.functional.interpolate(image_cache[("validation", datum['image_id'])]["label"].unsqueeze(0), size=(256, 256), mode='nearest').numpy()[0],
                          prompt_points=[([prompt_point[0] // 4, prompt_point[1] // 4], 0)],
                          label_name=viz.default_label_names,
                             extras={
                              "prompt": prompt_point,
                              "prompt_label": viz.default_label_names[cls]
                          }
            )

datapoints_cache.close()
image_cache.close()