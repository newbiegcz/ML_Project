from data.dataset import get_dataset_3d

import cv2
dataset_3d = get_dataset_3d('validation', crop_roi=True)
# show the first image
image = dataset_3d[0]['image'][0][:, :, 40]
print(image.shape)
cv2.imshow('image', image.numpy())
cv2.waitKey(0)