
import torch
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
eval_transforms = Compose(
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

_data = {
    "training": datalist,
    "validation": val_files
}

def get_dataset(data_source, transform):
    set_track_meta(True)
    dataset = CacheDataset(
            data=_data[data_source], 
            transform=transform, 
            cache_rate=1.0, 
            num_workers=4
        )
    set_track_meta(False)
    return dataset

def get_loader(data_source, transform, shuffle, batch_size=1, num_workers=0):
    assert data_source in _data.keys(), "Invalid data source!"
    loader = ThreadDataLoader(get_dataset(data_source, transform=transform), batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return loader

if __name__ == "__main__":
    import rich
    def test_loader(loader):
        d = next(iter(loader))
        rich.print(d)
    train_loader = get_loader("training", train_transforms, True)
    train_eval_loader = get_loader("training", eval_transforms, False)
    val_loader = get_loader("validation", eval_transforms, False)
    test_loader(train_loader)
    test_loader(train_eval_loader)
    test_loader(val_loader)