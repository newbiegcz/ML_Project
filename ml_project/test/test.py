import albumentations
from data.exp_dataset import wrap_with_torchseed, wrap_albumentations_transform, DictTransform
from data.dataset import PreprocessForModel, get_dataloader_2d
import torchvision
import torch
import numpy as np
transform_2d_withaugmentation = (
    wrap_with_torchseed(
        torchvision.transforms.Compose([
            DictTransform(["image", "label"], lambda x : x.numpy()),
            wrap_albumentations_transform(
                albumentations.Compose([
                    albumentations.Lambda(image=lambda x, **kwargs : x.reshape(x.shape + (1,)).repeat(3, axis=2), 
                                          mask=lambda x, **kwargs : x.reshape(x.shape + (1,)).repeat(3, axis=2)),
                    albumentations.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5),
                    albumentations.Compose([
                        albumentations.Resize(height=512, width=512, p=1),
                        albumentations.CropNonEmptyMaskIfExists(256, 256, p=1),
                    ], p = 0.5), 
                    albumentations.HorizontalFlip(p=0.5),
                    albumentations.VerticalFlip(p=0.5),
                    albumentations.RandomRotate90(p=0.5),
                    albumentations.Resize(height=1024, width=1024, p=1), 
                    albumentations.pytorch.transforms.ToTensorV2(transpose_mask=True),
                ])
            ), 
            PreprocessForModel(normalize=True)
        ])
    )
)

pixel_mean=(np.array([123.675, 116.28, 103.53]) / 255).tolist()
pixel_std=(np.array([58.395, 57.12, 57.375]) / 255).tolist()

new_transform = (
    wrap_with_torchseed(
        albumentations.Compose([
            albumentations.Lambda(image=lambda x, **kwargs : x.reshape(x.shape + (1,)).repeat(3, axis=2), 
                                    mask=lambda x, **kwargs : x.reshape(x.shape + (1,)).repeat(3, axis=2)),

            albumentations.Compose([
                albumentations.Resize(height=512, width=512, p=1),
                albumentations.CropNonEmptyMaskIfExists(256, 256, p=1),
            ], p = 0.5), 
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.RandomRotate90(p=0.5),
            albumentations.Resize(height=1024, width=1024, p=1),
            albumentations.Normalize(pixel_mean, pixel_std, max_pixel_value=1.0),
        ])
    )
)

new_val_transform = (
    albumentations.Compose([
        albumentations.Lambda(image=lambda x, **kwargs : x.reshape(x.shape + (1,)).repeat(3, axis=2), 
                                mask=lambda x, **kwargs : x.reshape(x.shape + (1,)).repeat(3, axis=2)),
        albumentations.Resize(height=1024, width=1024, p=1),
        albumentations.Normalize(pixel_mean, pixel_std, max_pixel_value=1.0),
    ])
)

input = torch.rand(234, 452)
transformed_img = transform_2d_withaugmentation({"image":input, "label":input}, seed=10)["image"]
transformed_img2 = new_transform(image=input.numpy(), mask=input.numpy(), seed=10)["image"]
dif = torch.from_numpy(transformed_img2).permute(2, 0, 1) - transformed_img
print(dif.min(), dif.max())
print(transformed_img.max(), transformed_img.min())
print(transformed_img.mean(), transformed_img.std())

# print(new_transform(image=input.numpy(), mask=input.numpy(), seed=10).keys())