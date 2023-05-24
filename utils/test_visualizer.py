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