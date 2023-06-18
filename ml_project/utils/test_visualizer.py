import utils.visualize as visualize
import torch
visualize.initialize_window()
image = torch.rand(3, 256, 256)
pd_label = torch.randint(0, 14, (1, 256, 256))
gt_label = torch.randint(0, 14, (1, 256, 256))
prompt_points = [((100, 100), 1), ((200, 200), 0)]
visualize.add_object("test2D", "2D", 
                        image=image,
                        pd_label=pd_label,
                        gt_label=gt_label,
                        prompt_points=prompt_points,
                        label_name=visualize.default_label_names)

image = torch.rand(3, 256, 256, 256)
pd_label = torch.randint(0, 14, (1, 256, 256, 256))
gt_label = torch.randint(0, 14, (1, 256, 256, 256))
prompt_points = [[((100, 100), 1), ((200, 200), 0)], [((100, 100), 1), ((200, 200), 0)]] * 128
visualize.add_object("test3D", "3D",
                    image=image,
                    pd_label=pd_label,
                    gt_label=gt_label,
                    prompt_points=prompt_points,
                    label_name=visualize.default_label_names)

from data.dataset_old import *

val_dataloader = get_loader("validation", eval_transforms, shuffle=False)

i = 0
for d in val_dataloader:
    i += 1
    visualize.add_object("dataset%d" % i, "3D",
                    image=d['image'][0],
                    pd_label=d['label'][0],
                    gt_label=d['label'][0],
                    prompt_points=None,
                    label_name=visualize.default_label_names)