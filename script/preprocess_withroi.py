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

from monai.data import (
    load_decathlon_datalist,
    set_track_meta,
    ThreadDataLoader,
    CacheDataset
)
import torch.utils.data as data

from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
)

checkpoint_path = "checkpoint/sam_vit_h_4b8939.pth"
datapoints_disk_path = "processed_data/datapoints"
embedding_disk_path = "processed_data/embeddings"

size_threshold_in_bytes= 200 * 1024 * 1024 * 1024 # 200 GB
debug = False
times = 1 # The number of times to augment an image
datapoints_for_training = 100 # The number of datapoints to use for training
datapoints_for_validation = 100 # The number of datapoints to use for validation

datapoints_cache = diskcache.Cache(datapoints_disk_path, eviction_policy = "none")
image_cache = diskcache.Cache(embedding_disk_path, eviction_policy = "none")

encoder = build_pretrained_encoder("vit_h", eval=True)
encoder.to("cuda")

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

transform_2d_withnoaugmentation = (
    wrap_with_torchseed(
        torchvision.transforms.Compose([
            DictTransform(["image", "label"], lambda x : x.numpy()),
            wrap_albumentations_transform(
                albumentations.Compose([
                    albumentations.Lambda(image=unsqueeze, mask=unsqueeze),
                    gen_clache(p=1),
                    albumentations.pytorch.transforms.ToTensorV2(transpose_mask=True)
                ])
            ),
            PreprocessForModel(normalize=True),
        ])
    )
)

class Dataset2D(data.Dataset):
    def __init__(self, files, *, device, transform, dtype=np.float64, first_only=False, compress=False):
        if first_only:
            files = files.copy()[:1]

        self.files = files
        self.device = device
        self.transform = transform
        set_track_meta(True)
        if compress:
            _default_transform = Compose(
                [
                    LoadImaged(keys=["image", "label"], ensure_channel_first=True, dtype=dtype),
                    ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True, dtype=dtype),
                    CropForegroundd(keys=["image", "label"], source_key="image", dtype=dtype),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    EnsureTyped(keys=["image", "label"], device=self.device, track_meta=False, dtype=dtype),
                    Spacingd(keys=["image", "label"],pixdim=(1.0, 1.0, 2.0),mode=("bilinear", "nearest")),
                ]
            )
        else:
            _default_transform = Compose(
                [
                    LoadImaged(keys=["image", "label"], ensure_channel_first=True, dtype=dtype),
                    ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True, dtype=dtype),
                    CropForegroundd(keys=["image", "label"], source_key="image", dtype=dtype),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    EnsureTyped(keys=["image", "label"], device=self.device, track_meta=False, dtype=dtype),
                ]
            )
        
        self.cache = CacheDataset(
            data=files, 
            transform=_default_transform, 
            cache_rate=1.0, 
            num_workers=4
        )
        set_track_meta(False)

        self.data_list = []
        for d in self.cache:
            # print(d['image_meta_dict']['filename_or_obj']) ## raw_data\imagesTr\img0035.nii.gz
            cur_list = []
            img, label = d['image'][0], d['label'][0]
            h = img.shape[2]
            # print(compress, img.shape)
            for i in range(h):
                cur_list.append({
                    "image": img[:, :, i],
                    "label": label[:, :, i],
                    "h": i / h
                })
            self.data_list.append(cur_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, id, idx):
        ret = self.data_list[id][idx].copy()
        #print('image shape:', ret['image'].shape)
        #print('label shape:', ret['label'].shape)
        if self.transform:
            t = self.transform(ret)
            #print(t['image'].shape, t['label'].shape)
            return t
        else:
            return ret

seed_rng = torch.Generator(device='cpu')
seed_rng.manual_seed(19260817)
data_files_training = data_files["training"][20:21]
data_files_validation = data_files["validation"][:1]
rraw_dataset_training = Dataset2D(data_files_training, device=torch.device('cpu'), transform=None, dtype=np.float32, compress = True)
rraw_dataset_validation = Dataset2D(data_files_validation, device=torch.device('cpu'), transform=None, dtype=np.float32, compress = True)

def gen_training(img):
    image_seed = torch.randint(1000000, (1,), generator=seed_rng).item()
    with TorchSeed(image_seed):
        d = transform_2d_withnoaugmentation(img, seed=image_seed+1)
    return d

def gen_validation(img):
    image_seed = torch.randint(1000000, (1,), generator=seed_rng).item()
    with TorchSeed(image_seed):
        d = transform_2d_withnoaugmentation(img, seed=image_seed+1)
    return d

train_c = len(rraw_dataset_training)
val_c = len(rraw_dataset_validation)
print(train_c)
print(val_c)

# lst = [0 for _ in range(14)]

#for i in range(train_c):
#    for j in range(len(raw_dataset_training.data_list[i])):
#        label = raw_dataset_training.data_list[i][j]["label"]
#        x, y = label.shape
#        for k in range(x):
#            for h in range(y):
#                lst[int(label[k][h].clone())] += 1

# print(lst)

print("doing preprocess...")

def preprocess(rraw_dataset, c, gen):
    res = []
    for i in range(c):
        lz = 10000000
        rz = 0
        lx = 10000000
        rx = 0
        ly = 10000000
        ry = 0
        dz = len(rraw_dataset.data_list[i])
        dx, dy = rraw_dataset.data_list[i][0]["label"].shape
        for j in range(dz):
            nonzero_indexes = torch.nonzero(rraw_dataset.data_list[i][j]["label"])
            if nonzero_indexes.shape[0] == 0:
                continue
            else:
                lz = min(lz, j)
                rz = max(rz, j)
                x1 = nonzero_indexes[:, 0].min().item()
                x2 = nonzero_indexes[:, 0].max().item()
                y1 = nonzero_indexes[:, 1].min().item()
                y2 = nonzero_indexes[:, 1].max().item()

                lx = min(lx, x1)
                rx = max(rx, x2)
                ly = min(ly, y1)
                ry = max(ry, y2)
    
        deltaz = int(float(rz - lz) * 0.05)
        deltax = int(float(rx - lx) * 0.05)
        deltay = int(float(ry - ly) * 0.05)

        lz = max(0, lz - deltaz)
        rz = min(dz - 1, rz + deltaz)
        lx = max(0, lx - deltax)
        rx = min(dx - 1, rx + deltax)
        ly = max(0, ly - deltay)
        ry = min(dy - 1, ry + deltay)
        print(lx, rx, ly, ry, lz, rz)
    
        for j in range(lz, rz + 1):
            img = rraw_dataset.data_list[i][j]["image"][lx : rx + 1, ly : ry + 1].clone()
            label = rraw_dataset.data_list[i][j]["label"][lx : rx + 1, ly : ry + 1].clone()
            cur = dict()
            cur["image"] = img
            cur["label"] = label
            apt = gen(cur)
            apt["h"] = float(j - lz) / float(rz - lz)
            res.append(apt)

    return res

raw_dataset_training = preprocess(rraw_dataset_training, train_c, gen_training)
raw_dataset_validation = preprocess(rraw_dataset_validation, val_c, gen_validation)

print("doing image encoding for training...")
import torch.nn.functional

batch_size = 1

img_list = []
label_list = []
num_image_training = 0

for i in tqdm(range(len(raw_dataset_training))):
    for j in range(times):
        image = raw_dataset_training[i]
        img_list.append(image["image"])
        label_list.append(image["label"])
        if (len(img_list) >= batch_size):
            img = torch.stack(img_list)

            with torch.inference_mode():
                embeddings = encoder(img.cuda()).cpu()

            for k in range(batch_size):
                cur = dict()
                cur["embedding"] = embeddings[k].clone()
                cur["low_res_image"] = (torch.nn.functional.interpolate(img[k].unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False) * PreprocessForModel.pixel_std + PreprocessForModel.pixel_mean).clone()
                cur["label"] = torch.tensor(label_list[k], dtype=torch.uint8).clone()
                cur["h"] = image["h"]
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
    image = raw_dataset_validation[i]
    img_list.append(image["image"])
    label_list.append(image["label"])
    if (len(img_list) >= batch_size):
        img = torch.stack(img_list)

        with torch.inference_mode():
            embeddings = encoder(img.cuda()).cpu()

        for k in range(batch_size):
            cur = dict()
            cur["embedding"] = embeddings[k].clone()
            cur["low_res_image"] = (torch.nn.functional.interpolate(img[k].unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False) * PreprocessForModel.pixel_std + PreprocessForModel.pixel_mean).clone()
            cur["label"] = torch.tensor(label_list[k], dtype=torch.uint8).clone()
            cur["h"] = image["h"]
            image_cache[("validation", num_image_validation + k)] = cur
    
        num_image_validation += batch_size
        img_list = []
        label_list = []

print("validation encoding done! The total number of image is %d." % (num_image_validation))
image_cache["num_image_for_validation"] = num_image_validation

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
    while True:
        cur_label = torch.randint(14, (1,), generator=seed_rng).item()
        if (len(label_list[cur_label]) > 0):
            break
    
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
    while True:
        cur_label = torch.randint(14, (1,), generator=seed_rng).item()
        if (len(label_list[cur_label]) > 0):
            break
    
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