import torch
import torchvision
import numpy as np
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

# TODO: 数据增广，特别是 RGB & 随机漂移

# TODO: 把 3d array cache 到内存里

# TODO: 考虑 cache encoder 的结果

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "/root/autodl-tmp/raw_data/"
split_json = "dataset_0.json"

datasets = data_dir + split_json

data_files = {
    "training": load_decathlon_datalist(datasets, True, "training"),
    "validation": load_decathlon_datalist(datasets, True, "validation")
}

class DictTransform:
    def __init__(self, keys, transform):
        self.keys = keys
        self.transform = transform

    def __call__(self, x):
        x = x.copy()
        for key in self.keys:
            x[key] = self.transform(x[key])
        return x
    
class PreprocessForModel:
    pixel_mean=(torch.Tensor([123.675, 116.28, 103.53]) / 255).view(-1, 1, 1)
    pixel_std=(torch.Tensor([58.395, 57.12, 57.375]) / 255).view(-1, 1, 1)
    img_size=1024

    def __init__(self, normalize=False):
        self.normalize = normalize

    def get_preprocess_shape(self, oldh: int, oldw: int, long_side_length: int):
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __call__(self, x):
        x = x.copy()
        target_size = self.get_preprocess_shape(x['image'].shape[1], x['image'].shape[2], self.img_size)
        tr_img = torchvision.transforms.Resize(target_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        tr_label = torchvision.transforms.Resize(target_size, interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT, antialias=False)
        x['image'] = tr_img(x['image'])
        x['label'] = tr_label(x['label'])

        if self.normalize:
            x['image'] = (x['image'] - self.pixel_mean.to(x['image'].device)) / self.pixel_std.to(x['image'].device)
        h, w = target_size
        padh = self.img_size - h
        padw = self.img_size - w
        x['image'] = torch.nn.functional.pad(x['image'], (0, padw, 0, padh))
        x['label'] = torch.nn.functional.pad(x['label'], (0, padw, 0, padh))
        return x
    


transforms = {
    "naive_to_rgb_and_normalize": torchvision.transforms.Compose(
        [DictTransform(["image", "label"], torchvision.transforms.Lambda(lambda x: x.unsqueeze(0).repeat(3, 1, 1))),
        PreprocessForModel(normalize=True)]
    ),
    "naive_to_rgb": torchvision.transforms.Compose(
        [DictTransform(["image", "label"], torchvision.transforms.Lambda(lambda x: x.unsqueeze(0).repeat(3, 1, 1))),
        PreprocessForModel(normalize=False)]
    ),
}

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
            img, label = d['image'][0], d['label'][0]
            h = img.shape[2]
            # print(compress, img.shape)
            for i in range(h):
                self.data_list.append({
                    "image": img[:, :, i],
                    "label": label[:, :, i],
                    "h": i / h
                })
            print(img.shape)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        ret = self.data_list[idx].copy()
        #print('image shape:', ret['image'].shape)
        #print('label shape:', ret['label'].shape)
        if self.transform:
            t = self.transform(ret)
            #print(t['image'].shape, t['label'].shape)
            return t
        else:
            return ret
    
def get_dataloader_2d(file_key, transform_key, batch_size, shuffle, device=device, first_only=False):
    '''
    A helper function to get a DataLoader

    Args:
        file_key: The key of the file to load, should be one of "training" and "validation"
        transform_key: The key of the transform to apply, should be one of "naive_to_rgb_and_normalize" and "naive_to_rgb"
        batch_size: The batch size
        shuffle: Whether to shuffle the data
        device: The device to load the data to
        first_only: Whether to load only the first file
    '''
    assert file_key in data_files.keys(), "Invalid file key!"
    assert transform_key in transforms.keys(), "Invalid transform key!"
    loader = ThreadDataLoader(Dataset2D(data_files[file_key], 
                                        transform=transforms[transform_key],
                                        device=device, first_only=first_only), 
                                    batch_size=batch_size, 
                                    num_workers=0, 
                                    shuffle=shuffle)
    return loader 

if __name__ == "__main__":
    import rich, cv2
    it = get_dataloader_2d("validation", "naive_to_rgb", batch_size=1, shuffle=False)
    res_w = 0
    res_h = 0
    for d in it:
        res_w = max(res_w, d['image'].shape[2])
        res_h = max(res_h, d['image'].shape[3])
    rich.print(res_w, res_h)
    input("")
    d = next(iter(it))
    rich.print(d)
    rich.print(d["image"].shape)
    input("")
    while True:
        for d in it:
            v = d['image'][0].clone()
            v[0] = d['label'][0][0] / 14 + d['image'][0][0]
            cv2.imshow("qwq", v.cpu().numpy().transpose(1, 2, 0))
            cv2.waitKey(10)