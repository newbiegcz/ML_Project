import data.dataset as dataset
import cv2
import numpy

loader = dataset.get_dataloader_2d("validation", "naive_to_rgb", batch_size=1, shuffle=False, first_only=True)
crop_roi = True

imgs = {}
if crop_roi:
    min_x, min_y, max_x, max_y = 100000, 100000, 0, 0
for i, it in enumerate(loader):
    im = it["image"][0]
    imgs[i] = im
    if crop_roi:
        # found bounding box of nonzero label
        label = it["label"][0]
        label = label.cpu().numpy()[0]
        nonzeros = numpy.argwhere(label != 0)
        if len(nonzeros) == 0:
            continue
        min_x = min(min_x, nonzeros[:, 0].min())
        min_y = min(min_y, nonzeros[:, 1].min())
        max_x = max(max_x, nonzeros[:, 0].max())
        max_y = max(max_y, nonzeros[:, 1].max())

if crop_roi:
    print(min_x, min_y, max_x, max_y)

for i in imgs.keys():
    imgs[i] = imgs[i][:, min_x:max_x, min_y:max_y]


    
for i in imgs.keys():
    cv2.imwrite("test/test_trained_sam/local_files/image%d.jpg" % i, imgs[i].cpu().numpy().transpose(1, 2, 0) * 255)