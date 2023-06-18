import timeit

import torch

from third_party.segment_anything.build_sam import build_sam_vit_h
from data.dataset import get_data_loader

pretrained_checkpoint = "checkpoint\sam_vit_h_4b8939.pth"

sam = build_sam_vit_h(pretrained_checkpoint).to("cuda")

batch_size = 3

train_dataloader = get_data_loader("training", "naive_to_rgb_and_normalize", batch_size, True, device="cpu", first_only=False)
sam.eval()

i = 0
start = timeit.default_timer()
for d in train_dataloader:
    i += 1
    print("running... %d" % i)
    d['image'] = d['image'].to("cuda")
    with torch.no_grad():
        res = sam.image_encoder.forward(d['image'])
    stop = timeit.default_timer()
    print("avg: %.2f / img" % (stop - start) / i / batch_size)

print(stop - start)