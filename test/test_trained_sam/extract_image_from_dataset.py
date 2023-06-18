import data.dataset as dataset
import cv2
import numpy
import torch

loader = dataset.get_dataloader_2d("validation", "naive_to_rgb", batch_size=1, shuffle=False, first_only=True)

imgs = {}
labels = {}
height = {}
min_x, min_y, max_x, max_y = 100000, 100000, 0, 0
min_z, max_z = 100000, 0

for i, it in enumerate(loader):
    im = it["image"][0]
    lb = it["label"][0]
    imgs[i] = im
    labels[i] = lb[0]
    height[i] = i

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

    min_z = min(min_z, i)
    max_z = max(max_z, i)

    bound_z = i
    bound_x, bound_y = im.shape[1], im.shape[2]

# extend roi by 10%
min_x = max(0, min_x - int(0.1 * (max_x - min_x)))
min_y = max(0, min_y - int(0.1 * (max_y - min_y)))
max_x = min(bound_x, max_x + int(0.1 * (max_x - min_x)))
max_y = min(bound_y, max_y + int(0.1 * (max_y - min_y)))
min_z = max(0, min_z - int(0.1 * (max_z - min_z)))
max_z = min(bound_z, max_z + int(0.1 * (max_z - min_z)))

label_imgs = {}
print(min_x, min_y, max_x, max_y)
keys = list(imgs.keys()).copy()
for i in keys:
    imgs[i] = imgs[i][:, min_x:max_x, min_y:max_y]
    imgs[i] = torch.nn.functional.interpolate(imgs[i].unsqueeze(0), size=(1024, 1024), mode="bilinear", align_corners=False).squeeze(0)
    labels[i] = labels[i][min_x:max_x, min_y:max_y]
    labels[i] = torch.nn.functional.interpolate((((labels[i]).to(torch.float32)).unsqueeze(0)).repeat(3, 1, 1).unsqueeze(0), size=(1024, 1024), mode="nearest").squeeze(0)
    # randomly assign a color to each label
    label_imgs[i] = labels[i].clone()
    for j in range(0, 14):
        mask = labels[i][0] == j
        label_imgs[i][0][mask] = numpy.random.rand()
        label_imgs[i][1][mask] = numpy.random.rand()
        label_imgs[i][2][mask] = numpy.random.rand()

    height[i] = (i - min_z) / (max_z - min_z)
    if i < min_z or i > max_z:
        imgs.pop(i)

for i in imgs.keys():
    cv2.imwrite("test/test_trained_sam/local_files/image%d.jpg" % i, imgs[i].cpu().numpy().transpose(1, 2, 0) * 255)
    cv2.imwrite("test/test_trained_sam/local_files/label%d.jpg" % i, label_imgs[i].cpu().numpy().transpose(1, 2, 0) * 255)
    # save height
    with open("test/test_trained_sam/local_files/height%d.txt" % i, "w") as f:
        f.write(str(height[i]))