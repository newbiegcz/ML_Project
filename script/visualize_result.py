import numpy as np
import utils.visualize as visualize

file_name = "img0035"
result_dice_file = "result/%s_dice.npy" % file_name
result_dice = np.load(result_dice_file)

img = np.load("result/%s_imgs.npy" % file_name)
pd_label = np.load("result/%s_pd_labels.npy" % file_name)
gt_label = np.load("result/%s_gt_labels.npy" % file_name)

print(result_dice.shape) # (13,)
print(pd_label.shape, gt_label.shape, img.shape) # (?, 1024, 1024)

img = img.transpose(3, 0, 1, 2)
print(img.shape)

visualize.initialize_window()
visualize.add_object_3d("image3D",
                    image=img,
                    pd_label=pd_label,
                    gt_label=gt_label,
                    prompt_points=None,
                    label_name=visualize.default_label_names,
                    extras=None
)