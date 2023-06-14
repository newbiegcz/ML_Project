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
datapoints_disk_path = "/root/autodl-tmp/data_with_roi/datapoints"
embedding_disk_path = "/root/autodl-tmp/data_with_roi/embeddings"

size_threshold_in_bytes= 500 * 1024 * 1024 * 1024 # 500 GB
debug = False
times = 15 # The number of times to augment an image
time_points = 1000
datapoints_for_training = 10000000 # The number of datapoints to use for training
datapoints_for_validation = 100000 # The number of datapoints to use for validation

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

def unsqueeze(x, **kwargs):
    return x.reshape(x.shape + (1,))

def extend(x, **kwargs):
    w, h, z = x.shape
    assert(z == 1)
    acc = np.tile(np.arange(w), (h, 1)).T.reshape((w, h, 1))
    bcc = np.tile(np.arange(h), (w, 1)).reshape((w, h, 1))
    res = np.concatenate((x, acc, bcc), axis = 2)
    return res

def extend2(x, **kwargs):
    return np.repeat(x, 3, axis=2)

pixel_mean=(np.array([123.675, 116.28, 103.53]) / 255).tolist()
pixel_std=(np.array([58.395, 57.12, 57.375]) / 255).tolist()

new_train_transform = (
    wrap_with_torchseed(
        albumentations.Compose([
            albumentations.Lambda(image=lambda x, **kwargs : x.reshape(x.shape + (1,)).repeat(3, axis=2), 
                                    mask=lambda x, **kwargs : x.reshape(x.shape + (1,))),
            albumentations.Resize(height=512, width=512, p=1),
            albumentations.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            albumentations.CropNonEmptyMaskIfExists(256, 256, p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.RandomRotate90(p=0.5),
            albumentations.Resize(height=1024, width=1024, p=1),
            albumentations.Normalize(pixel_mean, pixel_std, max_pixel_value=1.0),
        ], keypoint_params=albumentations.KeypointParams(format='yx', remove_invisible = False))
    )
)

new_val_transform = (
    albumentations.Compose([
        albumentations.Lambda(image=lambda x, **kwargs : x.reshape(x.shape + (1,)).repeat(3, axis=2), 
                                mask=lambda x, **kwargs : x.reshape(x.shape + (1,))),
        albumentations.Resize(height=1024, width=1024, p=1),
        albumentations.Normalize(pixel_mean, pixel_std, max_pixel_value=1.0),
    ], keypoint_params=albumentations.KeypointParams(format='yx', remove_invisible = False))
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
                    Spacingd(keys=["image", "label"],pixdim=(0.7, 0.7, 2.0),mode=("bilinear", "nearest")),
                    EnsureTyped(keys=["image", "label"], device=self.device, track_meta=False, dtype=dtype),
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
data_files_training = data_files["training"]
data_files_validation = data_files["validation"]
rraw_dataset_validation = Dataset2D(data_files_validation, device=torch.device('cpu'), transform=None, dtype=np.float32, compress = True)
rraw_dataset_training = Dataset2D(data_files_training, device=torch.device('cpu'), transform=None, dtype=np.float32, compress = True)

def gen_training(img, image_seed):
    with TorchSeed(image_seed):
        d = new_train_transform(image = img["image"], mask = img["label"], keypoints = img["keypoints"], seed=image_seed)
    return d

def gen_validation(img, image_seed):
    with TorchSeed(image_seed):
        d = new_val_transform(image = img["image"], mask = img["label"], keypoints = img["keypoints"], seed=image_seed)
    return d

def In(x, y):
    return x >= 0 and y >= 0 and x < 1024 and y < 1024

train_c = len(rraw_dataset_training)
val_c = len(rraw_dataset_validation)
# lst = [0 for _ in range(14)]

#for i in range(train_c):
#    for j in range(len(raw_dataset_training.data_list[i])):
#        label = raw_dataset_training.data_list[i][j]["label"]
#        x, y = label.shape
#        for k in range(x):
#            for h in range(y):
#                lst[int(label[k][h].clone())] += 1

# print(lst)
import torch.nn.functional
print("doing preprocess...")
batch_size = 4

def preprocess(rraw_dataset, c, gen, times, is_training):
    res = []

    lx = [10000000 for _ in range(c)]
    ly = [10000000 for _ in range(c)]
    lz = [10000000 for _ in range(c)]
    rx = [0 for _ in range(c)]
    ry = [0 for _ in range(c)]
    rz = [0 for _ in range(c)]

    for i in range(c):
        dz = len(rraw_dataset.data_list[i])
        dx, dy = rraw_dataset.data_list[i][0]["label"].shape
        for j in range(dz):
            nonzero_indexes = torch.nonzero(rraw_dataset.data_list[i][j]["label"])
            if nonzero_indexes.shape[0] == 0:
                continue
            else:
                lz[i] = min(lz[i], j)
                rz[i] = max(rz[i], j)
                x1 = nonzero_indexes[:, 0].min().item()
                x2 = nonzero_indexes[:, 0].max().item()
                y1 = nonzero_indexes[:, 1].min().item()
                y2 = nonzero_indexes[:, 1].max().item()

                lx[i] = min(lx[i], x1)
                rx[i] = max(rx[i], x2)
                ly[i] = min(ly[i], y1)
                ry[i] = max(ry[i], y2)
    
        deltaz = int(float(rz[i] - lz[i]) * 0.1)
        deltax = int(float(rx[i] - lx[i]) * 0.1)
        deltay = int(float(ry[i] - ly[i]) * 0.1)

        lz[i] = max(0, lz[i] - deltaz)
        rz[i] = min(dz - 1, rz[i] + deltaz)
        lx[i] = max(0, lx[i] - deltax)
        rx[i] = min(dx - 1, rx[i] + deltax)
        ly[i] = max(0, ly[i] - deltay)
        ry[i] = min(dy - 1, ry[i] + deltay)
        print(lx[i], rx[i], ly[i], ry[i], lz[i], rz[i])

    num_image = 0
    for i in range(c):
        num_image += ((rz[i] - lz[i] + 1) * times)
        
    print(num_image)

    lst_points = np.zeros((num_image, 14, time_points, 4), dtype=np.float32)
    lengths = np.zeros((num_image, 14), dtype=np.int32)
    idi = 0

    img_list = []
    label_list = []
    num_image = 0

    for i in range(c):
        for j in tqdm(range(lz[i], rz[i] + 1)):
            img = rraw_dataset.data_list[i][j]["image"][lx[i] : rx[i] + 1, ly[i] : ry[i] + 1].clone()
            label = rraw_dataset.data_list[i][j]["label"][lx[i] : rx[i] + 1, ly[i] : ry[i] + 1].clone()
            x, y = img.shape 
            ids = np.zeros((x, y), dtype = np.int32)
            coords = np.zeros((x * y, 2), dtype = np.int32)
            nums = np.zeros((14), dtype = np.int32)
            pnt_lst = np.zeros((14, time_points * 10), dtype = np.int32)
            cur = dict()
            cur["image"] = img.numpy()
            cur["label"] = label.numpy()
            cur["keypoints"] = []
            for k in range(x):
                for h in range(y):
                    coords[k * y + h][0] = k
                    coords[k * y + h][1] = h

            coords = coords[np.random.permutation(coords.shape[0])]
            for k in range(x * y):
                xx = coords[k][0]
                yy = coords[k][1]
                lbl = int(label[xx][yy])
                if nums[lbl] < time_points * 10:
                    ids[xx][yy] = len(cur["keypoints"])
                    pnt_lst[lbl][nums[lbl]] = ids[xx][yy]
                    cur["keypoints"].append([xx, yy])
                    nums[lbl] += 1

            for k in range(times):
                image_seed = torch.randint(1000000, (1,), generator=seed_rng).item() + 1
                if is_training:
                    image_cache[("seed", idi)] = image_seed
                    
                apt = gen(cur, image_seed)
                apt["image"] = torch.tensor(apt["image"]).permute(2, 0, 1)
                apt["mask"] = torch.tensor(apt["mask"]).permute(2, 0, 1)
                apt["h"] = float(j - lz[i]) / float(rz[i] - lz[i])
                for h in range(14):
                    if nums[h] == 0:
                        continue
                    else:
                        for p in range(nums[h]):
                            if lengths[idi][h] >= time_points:
                                break
                            id = pnt_lst[h][p]
                            xx = cur["keypoints"][id][0]
                            yy = cur["keypoints"][id][1]
                            assert(label[xx][yy] == h)
                            if In(int(apt["keypoints"][id][0]), int(apt["keypoints"][id][1])):
                                if (int(apt["mask"][0][int(apt["keypoints"][id][0])][int(apt["keypoints"][id][1])]) == h):
                                    lst_points[idi][h][lengths[idi][h]][0] = int(apt["keypoints"][id][1])
                                    lst_points[idi][h][lengths[idi][h]][1] = int(apt["keypoints"][id][0])
                                    lst_points[idi][h][lengths[idi][h]][2] = xx / x
                                    lst_points[idi][h][lengths[idi][h]][3] = yy / y
                                    lengths[idi][h] += 1

                apt["keypoints"] = []

                idi += 1

                img_list.append(apt["image"])
                label_list.append(apt["mask"])
                if (len(img_list) >= batch_size):
                    img = torch.stack(img_list)

                    with torch.inference_mode():
                        embeddings = encoder(img.cuda()).cpu()

                        for p in range(batch_size):
                            img_emb = dict()
                            img_emb["embedding"] = embeddings[p].clone()
                            img_emb["low_res_image"] = (torch.nn.functional.interpolate(img[p].unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False) * PreprocessForModel.pixel_std + PreprocessForModel.pixel_mean).clone()
                            img_emb["label"] = torch.tensor(label_list[p], dtype=torch.uint8).clone()
                            img_emb["h"] = apt["h"]
                            if is_training:
                                image_cache[("training", num_image + p)] = img_emb
                            else:
                                image_cache[("validation", num_image + p)] = img_emb
    
       
                        num_image += batch_size
                        img_list = []
                        label_list = []

    if is_training:
        image_cache["num_image_for_training"] = num_image
        print("training encoding done! The total number of image is %d." % (num_image))
    else:
        print("validation encoding done! The total number of image is %d." % (num_image))
        image_cache["num_image_for_validation"] = num_image

    return lst_points, lengths

lst_points_training, length_training = preprocess(rraw_dataset_training, train_c, gen_training, times, is_training = True)
lst_points_validation, length_validation = preprocess(rraw_dataset_validation, val_c, gen_validation, 1, is_training = False)

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

label_list = [[] for _ in range(14)]
for i in tqdm(range(num_image_training)):
    for j in range(0, 14):
        if length_training[i][j] > 0:
            label_list[j].append(i)

for i in tqdm(range(datapoints_for_training)):
    while True:
        cur_label = torch.randint(14, (1,), generator=seed_rng).item()
        if (len(label_list[cur_label]) > 0):
            break
    
    image_index = torch.randint(len(label_list[cur_label]), (1,), generator=seed_rng).item()
    image_index = label_list[cur_label][image_index]
    id = torch.randint(length_training[image_index][cur_label], (1,), generator=seed_rng).item()

    datum = POINT(image_id = int(image_index), mask_cls = int(cur_label), prompt_point = list(lst_points_training[image_index, cur_label, id]))

    datapoints_cache[("training", i)] = datum

print("training datapoints completed!The number of total datapoints is %d." % (datapoints_for_training))
datapoints_cache["num_datapoints_for_training"] = datapoints_for_training

print("generating validation datapoints...")
label_list = [[] for _ in range(14)]
for i in tqdm(range(num_image_validation)):
    for j in range(0, 14):
        if length_validation[i][j] > 0:
            label_list[j].append(i)

for i in tqdm(range(datapoints_for_validation)):
    while True:
        cur_label = torch.randint(14, (1,), generator=seed_rng).item()
        if (len(label_list[cur_label]) > 0):
            break
        
    image_index = torch.randint(len(label_list[cur_label]), (1,), generator=seed_rng).item()
    image_index = label_list[cur_label][image_index]
    id = torch.randint(length_validation[image_index][cur_label], (1,), generator=seed_rng).item()

    datum = POINT(image_id = int(image_index), mask_cls = int(cur_label), prompt_point = list(lst_points_validation[image_index, cur_label, id]))

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
                              "prompt_label": viz.default_label_names[cls],
                              "coordinate": (prompt_point[2], prompt_point[3])
                          }
            )

datapoints_cache.close()
image_cache.close()