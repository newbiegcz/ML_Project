from data.dataset import get_dataset_3d

import cv2
dataset_3d = get_dataset_3d('validation', crop_roi=True)
# show the first image
#print(dataset_3d[0]['image'].shape)
image = dataset_3d[0]['image'][0][:, :, 20]
label = dataset_3d[0]['label'][0][: ,:, 20]
print(label.shape)
cv2.imshow('qwq', label.numpy())
cv2.waitKey(0)
cv2.destroyAllWindows()