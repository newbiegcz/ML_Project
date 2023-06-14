from data.dataset import get_dataset_3d

import cv2
dataset_3d = get_dataset_3d('validation', crop_roi=True)
# show the first image
#print(dataset_3d[0]['image'].shape)
image = dataset_3d[0]['image'][0][:, :, 20]
print(dataset_3d[0]['image'][0].shape)
func = dataset_3d[0]['prompt_3d']
print(func(1,1,2))
print(func(*dataset_3d[0]['image'][0].shape))
func = dataset_3d[1]['prompt_3d']
print(func(1,1,2))
print(func(*dataset_3d[1]['image'][0].shape))
cv2.imshow('image', image.numpy())
cv2.waitKey(0)
