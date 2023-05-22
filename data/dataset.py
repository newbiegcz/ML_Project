import torch
from rich import print as print
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    set_track_meta,
)

from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
)

device = torch.device("cpu")
data_dir = "raw_data/"
split_json = "dataset_0.json"

num_samples = 4

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ]
)


datasets = data_dir + split_json
datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")

_origin_train_dataset = None
_augmented_train_dataset = None
_val_dataset = None
set_track_meta(False)

def get_augmented_train_dataset():
    global _augmented_train_dataset
    if _augmented_train_dataset is None:
        set_track_meta(True)
        _augmented_train_dataset = CacheDataset(
            data=datalist,
            transform=train_transforms,
            cache_num=24,
            cache_rate=1.0,
            num_workers=8,
        )
        set_track_meta(False)
    return _augmented_train_dataset

def get_origin_train_dataset():
    '''注意: 使用了和验证集一样的 transform，因而有元数据'''
    global _origin_train_dataset
    if _origin_train_dataset is None:
        set_track_meta(True)
        _origin_train_dataset = CacheDataset(
            data=datalist,
            transform=val_transforms,
            cache_num=24,
            cache_rate=1.0,
            num_workers=8,
        )
        set_track_meta(False)
    return _origin_train_dataset

def get_val_dataset():
    global _val_dataset
    if _val_dataset is None:
        set_track_meta(True)
        val_dataset = CacheDataset(
            data=val_files, 
            transform=val_transforms, 
            cache_num=6, 
            cache_rate=1.0, 
            num_workers=4
        )
        set_track_meta(False)
    return _val_dataset

def get_augmented_train_loader(batch_size=1):
    return ThreadDataLoader(get_augmented_train_dataset(), num_workers=0, batch_size=batch_size, shuffle=False)

def get_origin_train_loader(batch_size=1):
    return ThreadDataLoader(get_origin_train_dataset(), num_workers=0, batch_size=batch_size, shuffle=False)

def get_val_loader(batch_size=1):
    return ThreadDataLoader(get_val_dataset(), num_workers=0, batch_size=1)